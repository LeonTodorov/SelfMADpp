import torch
from torchvision import utils
import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging

from data.frequency_artifact_generator import FrequencyArtifactGenerator
from data.pixel_artifact_generator import PixelArtifactGenerator
from data.transforms import get_gam_transforms, crop_face, reorder_landmark, hflip 
from utils.util_fun import load_data_config

class SelfMADppDataset(torch.utils.data.Dataset):

	VALID_PHASES = ("train", "test")
	VALID_SOURCES = ("FF++", "FRGC", "FRLL", "SMDD", "ONOT", "FLUXSynID")
	DEFAULT_EXTS = (".png", ".jpg", ".jpeg")

	def __init__(
		self,
		cfg,
		phase = "train",
		source_name = "SMDD",
		frequency_artifact_generator_prob = 0.1,
		extensions = DEFAULT_EXTS,
	):
		if phase not in self.VALID_PHASES:
			raise ValueError(f"Invalid phase '{phase}'. Valid: {self.VALID_PHASES}")
		if source_name not in self.VALID_SOURCES:
			raise ValueError(f"Invalid source_name '{source_name}'. Valid: {self.VALID_SOURCES}")
		if not (0.0 <= frequency_artifact_generator_prob <= 1.0):
			raise ValueError("frequency_artifact_generator_prob must be in [0, 1].")

		try:
			img_root = Path(cfg["dataset"]['train'][source_name]["path"]).expanduser().resolve()
		except Exception as e:
			raise KeyError(f"cfg missing dataset path for '{source_name}': {e}") from e
		lm_root = img_root.with_name(img_root.name + "_lm")
		lb_root = img_root.with_name(img_root.name + "_lb")
		if not lm_root.exists():
			raise FileNotFoundError(f"Landmarks directory does not exist: {lm_root}")
		if not lb_root.exists():
			raise FileNotFoundError(f"Labels directory does not exist: {lb_root}")
	
		self.phase: str = phase
		self.source_name: str = source_name
		self.image_size = tuple([cfg.get("image_size", 512)] * 2)
		self.frequency_artifact_generator_prob: float = frequency_artifact_generator_prob
		self._exts = {e.lower() for e in extensions}
		self.gam_transforms = get_gam_transforms()
		self.frequency_artifact_generator = FrequencyArtifactGenerator()
		self.pixel_artifact_generator = PixelArtifactGenerator()

		image_paths, landmark_paths, label_paths = [], [], []
		for root, _, files in os.walk(img_root):
			if not files:
				continue
			for filename in files:
				if filename.endswith(('.png', '.jpg', '.jpeg')):
					image_path = Path(root) / filename
					landmark_path = lm_root / image_path.relative_to(img_root).with_suffix('.npy')
					label_path = lb_root / image_path.relative_to(img_root).with_suffix('.npy')
					if landmark_path.exists() and label_path.exists():
						image_paths.append(image_path)
						landmark_paths.append(landmark_path)
						label_paths.append(label_path)

		if not image_paths:
			raise FileNotFoundError(
				f"No valid image/landmark/label triplets found under: {img_root} (with {lm_root}, {lb_root})"
			)
		
		self.landmark_paths = landmark_paths
		self.image_paths = image_paths
		self.label_paths = label_paths

		logging.info(f'Loaded {source_name}@{phase} with {len(image_paths)} source images.')

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		flag=True
		while flag:
			try:
				img = np.array(Image.open(self.image_paths[idx]))
				landmark = reorder_landmark(np.load(self.landmark_paths[idx]).squeeze())
				bbox = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()]).reshape(2, 2)
				face_label = np.load(self.label_paths[idx])

				if self.phase=='train' and np.random.rand() < 0.5:
					img, _, landmark, bbox, face_label = hflip(img = img, mask = None, landmark = landmark, bbox = bbox, label = face_label)

				img_bf, img_ma = self.pixel_artifact_generator(image = img, landmark = landmark, label = face_label)['image']
				if self.phase=='train':
					transformed_imgs = self.gam_transforms(image = img_ma,image_ = img_bf)
					img_ma, img_bf = transformed_imgs['image'], transformed_imgs['image_']

				if np.random.rand() < self.frequency_artifact_generator_prob:
					img_ma = self.frequency_artifact_generator(image = img_ma, label = face_label)['image']
				img_ma, _, _, _, _, (y0, y1, x0, x1) = crop_face(img = img_ma, label = face_label, landmark = landmark, bbox = bbox, phase = self.phase)		
				img_bf = img_bf[y0:y1, x0:x1]

				img_ma = cv2.resize(img_ma, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32').transpose(2, 0, 1) / 255.
				img_bf = cv2.resize(img_bf, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32').transpose(2, 0, 1) / 255.
				raw_mask = np.isin(cv2.resize(face_label[y0:y1, x0:x1], self.image_size, interpolation=cv2.INTER_NEAREST), (0, 13, 14, 15, 16, 17, 18), 0, 1)[None, ...]

				flag = False
			except Exception as e:
				print(e)
				idx = torch.randint(low = 0, high = len(self), size=(1,)).item()
		return torch.from_numpy(img_bf), torch.from_numpy(img_ma), torch.from_numpy(raw_mask)

	def collate_fn(self, batch):
		img_bf, img_ma, raw_mask = zip(*batch)
		img = torch.cat([
			torch.stack(img_bf, dim=0),
			torch.stack(img_ma, dim=0)
        ], dim=0)
		raw_mask = torch.stack(raw_mask, dim=0)
		mask = torch.cat([
			raw_mask, 
			torch.zeros_like(raw_mask)
		], dim=0)
		label = torch.cat([
			torch.zeros(len(img_bf)),
			torch.ones(len(img_ma))
		], dim=0)

		return {'img': img, 'mask': mask, 'label': label}

	def worker_init_fn(self,worker_id):                                                       
		np.random.seed(np.random.get_state()[1][0] + worker_id)

def test():
	cfg = load_data_config("data/data_config.yaml")
	image_dataset = SelfMADppDataset(cfg=cfg, phase='train', source_name="SMDD")

	bs = 16
	dl = torch.utils.data.DataLoader(
		image_dataset,
		batch_size = bs,
		shuffle = True,
		collate_fn = image_dataset.collate_fn,
		num_workers = 0,
		worker_init_fn = image_dataset.worker_init_fn
	)
	data_iter=iter(dl)
	data=next(data_iter)

	img=data['img'] # (2B, 3, H, W)
	mask=data['mask'] # (2B, 1, H, W)
	# label=data['label'] # (2B,)

	# don't save second "batch" of masks
	# masks should have shape (B, 1, H, W)
	mask = mask[:bs]
	
	img = img.view((-1, 3, image_dataset.image_size[0], image_dataset.image_size[1])) 
	mask = mask.view((-1, 1, image_dataset.image_size[0], image_dataset.image_size[1]))
	mask = mask.repeat(1, 3, 1, 1)
	utils.save_image(torch.cat([img, mask], dim=0), 'SelfMADpp_example.png', nrow=bs, normalize=False)