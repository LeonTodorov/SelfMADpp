import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import argparse

class FaceParser():
    def __init__(self, device):
        self.device = device
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.model.to(device)
        self.model.eval()

    def parse(self, img):
        inputs = self.image_processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(logits,
                size=img.size[::-1],
                mode='bilinear',
                align_corners=False)
        labels = upsampled_logits.argmax(dim=1)[0]
        labels_viz = labels.cpu().numpy()
        return labels_viz

def save_face_label(img_pth, save_path, face_parser, forget=False):
    img = np.array(Image.open(img_pth), dtype=np.uint8)
    face_labels = face_parser.parse(Image.fromarray(img))
    print(f"From {img_pth} to {save_path}, label shape: {face_labels.shape}")
    if not forget:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, face_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='Path to image daqtaset', required=True)
    parser.add_argument('-f', '--forget', action='store_true', default=False, help='If specified, do not save results', required=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp = FaceParser(device=device)
    input_path = args.input_path
    output_path = str(Path(input_path).parent / (Path(input_path).name + "_lb"))
    for root, dirs, files in os.walk(input_path):
        root_path = Path(root)
        for file in tqdm(files, desc="Processing files"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_pth = root_path / file
                # Construct output directory structure
                rel_path = img_pth.relative_to(input_path)
                save_dir = Path(output_path) / rel_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = rel_path.stem + ".npy"
                save_path = save_dir / save_file
                save_face_label(str(img_pth), str(save_path), fp, forget=args.forget)