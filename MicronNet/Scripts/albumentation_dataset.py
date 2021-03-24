import cv2
import albumentations as A
import os
import PIL.Image as Image
import numpy as np

data_transforms = A.Compose([
    A.Resize(48, 48),
    A.Compose([
        A.OneOf([A.MotionBlur(p=0.3), A.GaussianBlur(p=0.3), A.MedianBlur(p=0.3)], p=0.4),
        A.OneOf([A.GaussNoise(p=0.3), A.ISONoise(p=0.3), A.MultiplicativeNoise(p=0.3)], p=0.4),
        A.OneOf([A.ColorJitter(p=0.3), A.HueSaturationValue(p=0.3), A.RandomBrightnessContrast(p=0.3)], p=0.3),
        # A.OneOf([A.RandomFog(p=0.3), A.RandomRain(p=0.3),
        #     # RandomShadow(p=0.3),
        #     A.RandomSnow(p=0.3), A.RandomSunFlare(p=0.3)], p=0.4),
        A.OneOf([A.CoarseDropout(p=0.3), A.Cutout(p=0.3)], p=0.3),
        A.OneOf([A.GridDistortion(p=0.3), A.OpticalDistortion(p=0.3)], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=20, p=0.4, border_mode=cv2.BORDER_REPLICATE),
        # A.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.4672, 0.4564, 0.4629]),
    ], p=0.7)], p=1.0)

init_train = "../DatasetAug/train_images"
init_val = "../DatasetAug/val_images"
init_test = "../DatasetAug/test_images"
final_train = "../DatasetDiff/train_images"
final_val = "../DatasetDiff/val_images"
final_test = "../DatasetDiff/test_images"
nclasses = 48

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

os.mkdir(final_train)
os.mkdir(final_val)
os.mkdir(final_test)

def transfer(basetr, transtr): 
    for i in range(nclasses):
        fldr=("/0000" if i<10 else "/000")+str(i)
        os.mkdir(transtr+fldr)
        for path in os.listdir(basetr+fldr):
            full_path = os.path.join(basetr+fldr, path)
            if os.path.isfile(full_path) and not full_path.endswith(".csv"):
                data = data_transforms(image=np.array(pil_loader(full_path)))["image"]
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                # data = data.view(1, data.size(0), data.size(1), data.size(2))
                if cv2.imwrite(os.path.join(transtr+fldr, path), data):
                    print(full_path)

transfer(init_train, final_train)
transfer(init_val, final_val)

for path in os.listdir(init_test):
    full_path = os.path.join(init_test, path)
    if os.path.isfile(full_path) and not full_path.endswith(".csv"):
        data = data_transforms(image=np.array(pil_loader(full_path)))["image"]
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        if cv2.imwrite(os.path.join(final_test, path), data):
            print(full_path)