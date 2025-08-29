import os
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import cv2
    OPENCV = True
except:
    OPENCV = False

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

IMG_SIZE = 224  # to match MobileNetV2 input

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def crop_brain_opencv(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    margin = int(0.02 * max(np_img.shape[:2]))
    x1, y1 = max(x-margin,0), max(y-margin,0)
    x2, y2 = min(x+w+margin,np_img.shape[1]), min(y+h+margin,np_img.shape[0])
    return np_img[y1:y2, x1:x2]

def center_crop(np_img):
    h, w = np_img.shape[:2]
    min_dim = min(h, w)
    start_x, start_y = (w-min_dim)//2, (h-min_dim)//2
    return np_img[start_y:start_y+min_dim, start_x:start_x+min_dim]

def preprocess_image(in_path, out_path, img_size=IMG_SIZE):
    img = Image.open(in_path).convert('RGB')
    np_img = np.array(img)
    crop = None
    if OPENCV:
        try:
            crop = crop_brain_opencv(np_img)
        except:
            crop = None
    if crop is None:
        crop = center_crop(np_img)
    pil = Image.fromarray(crop).resize((img_size, img_size))
    ensure_dir(os.path.dirname(out_path))
    pil.save(out_path, quality=95)

def preprocess_folder(src_dir, dst_dir):
    ensure_dir(dst_dir)
    for f in os.listdir(src_dir):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            preprocess_image(os.path.join(src_dir,f), os.path.join(dst_dir,f'{Path(f).stem}.jpg'))

def make_splits(prep_dir, splits_dir, val_split=0.15, test_split=0.15, seed=42):
    random.seed(seed)
    for cls in os.listdir(prep_dir):
        cls_dir = os.path.join(prep_dir, cls)
        files = [os.path.join(cls_dir,f) for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')]
        random.shuffle(files)
        n = len(files)
        n_test = int(n*test_split)
        n_val = int(n*val_split)
        n_train = n - n_val - n_test
        splits = {'train': files[:n_train], 'val': files[n_train:n_train+n_val], 'test': files[n_train+n_val:]}
        for split_name, split_files in splits.items():
            for f in split_files:
                out_dir = os.path.join(splits_dir, split_name, cls)
                ensure_dir(out_dir)
                shutil.copy2(f, os.path.join(out_dir, os.path.basename(f)))

def augment_train(train_dir, aug_dir, augment_factor=4):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=(0.85,1.15),
        fill_mode='nearest'
    )
    for cls in os.listdir(train_dir):
        src_cls = os.path.join(train_dir, cls)
        dst_cls = os.path.join(aug_dir, cls)
        ensure_dir(dst_cls)
        for f in os.listdir(src_cls):
            if f.lower().endswith('.jpg'):
                img = load_img(os.path.join(src_cls,f), target_size=(IMG_SIZE,IMG_SIZE))
                x = img_to_array(img).reshape((1,)+img_to_array(img).shape)
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=dst_cls, save_prefix='aug', save_format='jpg'):
                    i+=1
                    if i>=augment_factor:
                        break
        # copy originals
        for f in os.listdir(src_cls):
            if f.lower().endswith('.jpg'):
                shutil.copy2(os.path.join(src_cls,f), os.path.join(dst_cls,f'orig_{f}'))

def main(input_dir, output_dir, augment_factor=4, val_split=0.15, test_split=0.15):
    prep_dir = os.path.join(output_dir,'preprocessed')
    splits_dir = os.path.join(output_dir,'splits')
    aug_dir = os.path.join(output_dir,'augmented_train')

    for cls in os.listdir(input_dir):
        preprocess_folder(os.path.join(input_dir,cls), os.path.join(prep_dir,cls))

    make_splits(prep_dir, splits_dir, val_split, test_split)
    augment_train(os.path.join(splits_dir,'train'), aug_dir, augment_factor)
    print("Preprocessing and augmentation complete. Check dataset_prepared/ folder.")

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--augment_factor', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.augment_factor, args.val_split, args.test_split)
