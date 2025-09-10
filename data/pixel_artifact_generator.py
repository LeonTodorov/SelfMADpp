import cv2
import numpy as np
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
from data.transforms import get_source_transforms, randaffine, zoom_at_point
from utils.util_fun import grouped_powerset

def get_blend_mask(mask):
	mask_resized = cv2.resize(mask, [np.random.randint(192,257)] * 2)
	mask_blured = cv2.GaussianBlur(mask_resized, [random.randrange(5, 26, 2)] * 2, 0)
	mask_blured = mask_blured / np.max(mask_blured)
	mask_blured[mask_blured < 1] = 0
	mask_blured = cv2.GaussianBlur(mask_blured, [random.randrange(5, 26, 2)] * 2, np.random.randint(5, 46))
	mask_blured = mask_blured / np.max(mask_blured)
	mask_blured = cv2.resize(mask_blured, mask.shape[::-1])[..., None]
	return mask_blured

def self_blending(img, landmark, label, source_transform):
	if np.random.rand() < 0.5:
		if np.random.rand() < 0.25:
			landmark = landmark[:68]
		mask = cv2.fillConvexPoly(np.zeros_like(img[:,:,0]), cv2.convexHull(landmark), 1.)
	else:
		mask = np.ones(img.shape[:-1], dtype=np.uint8)
		for masked_area in random.choice(grouped_powerset([0, 18, 17, 13, [8, 9]], return_empty=False)):
			mask = mask & (masked_area != label)

	source = img.copy()
	if np.random.rand() < 0.5:
		source = source_transform(image=source.astype(np.uint8))['image']
	else:
		img = source_transform(image=img.astype(np.uint8))['image']

	source, mask = randaffine(source, mask)
	blend_ratio = np.random.choice([i / 4 for i in range(1, 4)] + [1] * 3)
	mask_blurred = get_blend_mask(mask) * blend_ratio
	img_blended = (mask_blurred * source + (1 - mask_blurred) * img).astype(np.uint8)

	return img, img_blended, mask

def self_morphing(img, face_labels, landmarks):
	mask = np.ones(img.shape[:-1], dtype=np.uint8)
	for label in random.choice(grouped_powerset([0, 18, 17, 13, [8, 9]])):
		mask = mask & (face_labels != label)
	if np.random.rand() < 0.5:
		center = landmarks[30, :]
	else:
		center = np.mean(np.argwhere(mask == 1), axis=0)[::-1].astype(int)

	mask = np.stack([mask] * 3, axis=-1).astype(np.float32)
	source, target = img.copy(), img.copy()
	zoom_factor = np.random.uniform(1.0, 1.1)
	zoomed_source = zoom_at_point(source, *center, zoom_factor)
	zoomed_mask = zoom_at_point(mask, *center, zoom_factor)
	img_bf = target / 255
	zoomed_image_f = zoomed_source / 255
	blend_factor = random.choice([0.5, 0.5, 0.5, 0.375, 0.25, 0.125])
	img_ma = np.where(zoomed_mask == 0, img_bf, (1 - blend_factor) * img_bf + blend_factor * zoomed_image_f)
	img_ma = (img_ma * 255).astype(np.uint8)
	img_bf = (img_bf * 255).astype(np.uint8)

	return img_bf, img_ma, center

class PixelArtifactGenerator(ImageOnlyTransform):
	def __init__(self, p=1.0):
		super(PixelArtifactGenerator, self).__init__(p)
		self.source_tranform = get_source_transforms()
	
	@property
	def targets_as_params(self):
		return ["image", "landmark", "label"]

	def get_params_dependent_on_targets(self, params):
		return {
			"landmark": params.get("landmark", None),
			"label": params.get("label", None)
		}

	def apply(self, img, landmark, label, **kwargs):
		if np.random.rand() < 0.5:
			img_bf, img_ma, _ = self_blending(img, landmark, label, self.source_tranform)
		else:
			img_bf, img_ma, _ = self_morphing(img, label, landmark)
		return img_bf, img_ma

