import os
import cv2
import numpy as np
import dlib
from imutils import face_utils
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image


def crop_face(
    img,
    landmarks: np.ndarray,
    margin: float = 12.5,
    pad: bool = False,
):
    frame = img if isinstance(img, np.ndarray) else np.array(img)
    if landmarks is None:
        return img

    landmarks = np.asarray(landmarks)
    if landmarks.size == 0:
        return img

    if landmarks.ndim > 2:
        landmarks = landmarks.reshape(-1, landmarks.shape[-1])

    if landmarks.shape[-1] != 2:
        raise ValueError(
            "Expected landmarks with last dimension 2 (x, y) coordinates."
        )

    min_xy = np.min(landmarks, axis=0)
    max_xy = np.max(landmarks, axis=0)
    left, top = min_xy.astype(int)
    right, bottom = max_xy.astype(int)

    margin = max(margin, 0.0)
    box_height = bottom - top
    box_width = right - left
    pad_h = int(round(box_height * (margin / 100.0)))
    pad_w = int(round(box_width * (margin / 100.0)))

    top = top - pad_h
    bottom = bottom + pad_h
    left = left - pad_w
    right = right + pad_w

    frame_height, frame_width = frame.shape[:2]

    if pad:
        top_pad = max(0, -top)
        left_pad = max(0, -left)
        bottom_pad = max(0, bottom - frame_height)
        right_pad = max(0, right - frame_width)

        if any(v > 0 for v in (top_pad, bottom_pad, left_pad, right_pad)):
            pad_spec = ((top_pad, bottom_pad), (left_pad, right_pad))
            if frame.ndim == 3:
                pad_spec += ((0, 0),)
            frame = np.pad(frame, pad_spec, mode="constant", constant_values=0)

            top += top_pad
            bottom += top_pad
            left += left_pad
            right += left_pad

        frame_height, frame_width = frame.shape[:2]
        top = max(top, 0)
        left = max(left, 0)
        bottom = min(bottom, frame_height)
        right = min(right, frame_width)
    else:
        top = max(top, 0)
        left = max(left, 0)
        bottom = min(bottom, frame_height)
        right = min(right, frame_width)

    if top >= bottom or left >= right:
        return img

    cropped = frame[top:bottom, left:right]

    if isinstance(img, Image.Image):
        return Image.fromarray(cropped)
    return cropped

def save_crop(org_path, save_path, face_detector, face_predictor, forget=False):
    frame = cv2.imread(org_path, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None:
        raise ValueError(f"Image at path {org_path} could not be read.")
    faces = face_detector(frame, 1)
    if len(faces) == 0:
        print('No faces detected in {}'.format(org_path))
        return 
    # Select the face with the largest area and crop only that face
    largest_area = 0
    largest_landmark = None

    for face_rect in faces:
        landmark = face_predictor(frame, face_rect)
        landmark_np = face_utils.shape_to_np(landmark)
        x0, y0 = landmark_np[:, 0].min(), landmark_np[:, 1].min()
        x1, y1 = landmark_np[:, 0].max(), landmark_np[:, 1].max()
        area = (x1 - x0) * (y1 - y0)
        if area > largest_area:
            largest_area = area
            largest_landmark = landmark_np

    if largest_landmark is not None:
        frame_cropped = crop_face(frame, largest_landmark)
        landmarks = [largest_landmark]
    else:
        frame_cropped = frame
        landmarks = []
    print(f"{org_path}({frame.shape}) -> {save_path}({frame_cropped.shape})")
    if not forget:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR))
    
    if len(landmarks) == 0:
        print('No landmarks in {}'.format(org_path))
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='Path to image daqtaset', required=True)
    parser.add_argument('-f', '--forget', action='store_true', default=False, help='If specified, do not save results', required=False)
    args = parser.parse_args()
    
    
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'utils/preprocessing/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = str(Path(input_path).parent / (Path(input_path).name + "_cropped"))
    
    for root, _, files in os.walk(input_path):
        for file in tqdm(files, desc=f"Processing images in {root}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_path)
                save_dir = os.path.join(output_path, rel_path)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, file)
                save_crop(img_path, save_path, face_detector, face_predictor, forget=args.forget)
