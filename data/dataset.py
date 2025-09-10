from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
import torch

class MorphDataset(Dataset):
    
    VALID_DATASETS = {
        "FRLL": ("amsl", "facemorpher", "opencv", "stylegan", "webmorph"),
        "FRGC": ("facemorpher", "opencv", "stylegan"),
        "FERET": ("facemorpher", "opencv", "stylegan"),
        "MorDIFF": ("morphs_neutral", "morphs_smiling"),
        "DiMGreedy": ("dim_a", "dim_c", "fast_dim", "fast_dim_ode", "greedy_dim", "greedy_dim_s"),
        "MorphPIPE": ("morphs",),
        "MorCode": ("morphs",),
        "MIPGAN_I": ("morphs",),
        "MIPGAN_II": ("morphs",),
    }
    
    @staticmethod
    def _gather_images(directory, extensions):
        exts = {e.lower() for e in extensions}
        if not directory.exists():
            return []
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    def __init__(
        self,
        dataset_name,
        data_cfg,
        morph_type,
        extensions = ('.png', '.jpg', '.jpeg'),
    ):

        if dataset_name not in self.VALID_DATASETS:
            raise ValueError(
                f"Invalid dataset '{dataset_name}'. "
                f"Valid options: {list(self.VALID_DATASETS.keys())}"
            )
        if morph_type not in self.VALID_DATASETS[dataset_name]:
            raise ValueError(
                f"Invalid morph type '{morph_type}' for dataset '{dataset_name}'. "
                f"Valid options: {self.VALID_DATASETS[dataset_name]}"
            )
        
        try:
            root = Path(data_cfg["dataset"]["eval"][dataset_name]["path"]).expanduser().resolve()
        except Exception as e:
            raise KeyError(
                f"data_cfg missing dataset path for '{dataset_name}': {e}"
            ) from e
            
        if not root.exists():
            raise FileNotFoundError(f"Configured dataset path does not exist: {root}")
        
        self.raw_img_paths = self._gather_images(root / "raw", extensions)
        self.morph_img_paths = self._gather_images(root / morph_type, extensions)
        
        if not self.raw_img_paths:
            raise FileNotFoundError(f"No raw images found in: {root / 'raw'}")
        if not self.morph_img_paths:
            raise FileNotFoundError(f"No '{morph_type}' images found in: {root / morph_type}")

        self.image_paths = self.raw_img_paths + self.morph_img_paths
        self.labels = [0] * len(self.raw_img_paths) + [1] * len(self.morph_img_paths)
        
        img_size = data_cfg.get('image_size', 512)
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        
        self.dataset_name = dataset_name
        self.morph_type = morph_type
        self.root = root
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(path) as im:
            img = im.convert("RGB")

        img = self.transform(img)

        return {
            "img": img,
            "label": torch.tensor(label, dtype=torch.long),
            "fname": path.name,
            "path": str(path),
        }
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"dataset='{self.dataset_name}', morph_type='{self.morph_type}', "
                f"raw={len(self.raw_img_paths)}, morphs={len(self.morph_img_paths)})")

def test():
    from utils.util_fun import load_data_config
    
    data_cfg = load_data_config("data/data_config.yaml")
    datasets_to_test = [
        ('FRLL', 'stylegan'),
        ('FRLL', 'amsl'),
        ('FRLL', 'facemorpher'),
        ('FRLL', 'opencv'),
        ('FRLL', 'webmorph'),
        ('FRGC', 'stylegan'),
        ('FRGC', 'facemorpher'),
        ('FRGC', 'opencv'),
        ('FERET', 'stylegan'),
        ('FERET', 'facemorpher'),
        ('FERET', 'opencv'),
        ('MorDIFF', 'morphs_neutral'),
        ('MorDIFF', 'morphs_smiling'),
        ('DiMGreedy', 'dim_a'),
        ('DiMGreedy', 'dim_c'),
        ('DiMGreedy', 'fast_dim'),
        ('DiMGreedy', 'fast_dim_ode'),
        ('DiMGreedy', 'greedy_dim'),
        ('DiMGreedy', 'greedy_dim_s'),
        ('MorphPIPE', 'morphs'),
        ('MorCode', 'morphs'),
        ('MIPGAN_I', 'morphs'),
        ('MIPGAN_II', 'morphs'),
    ]
    for dataset_name, morph_type in datasets_to_test:
        try:
            ds = MorphDataset(dataset_name=dataset_name, morph_type=morph_type, data_cfg=data_cfg)
            print("[OK]", ds)
        except Exception as e:
            print(f"[FAIL] {dataset_name}/{morph_type}: {e}")