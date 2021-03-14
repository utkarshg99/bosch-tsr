from __future__ import print_function
import zipfile
import os

import torchvision.transforms as transforms
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
# data_transforms = transforms.Compose([
#     transforms.Resize((48, 48)),
#     # transforms.RandomApply([], p=0.2)
#     # transforms.RandomApply(, p=0.2)
#     # transforms.RandomApply(, p=0.1)
#     # transforms.RandomApply(, p=0.2)
#     transforms.ToTensor(),
#     transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
# ])
test_data_transforms = A.Compose([A.Resize(48, 48),ToTensorV2()], p=1.0)

data_transforms = A.Compose([
    A.Resize(48, 48),
    A.Compose([
        A.OneOf([A.MotionBlur(p=0.1), A.GaussianBlur(p=0.1), A.MedianBlur(p=0.1)], p=0.2),
        A.OneOf([A.GaussNoise(p=0.1), A.ISONoise(p=0.1), A.MultiplicativeNoise(p=0.1)], p=0.2),
        A.OneOf([A.ColorJitter(p=0.1), A.HueSaturationValue(p=0.1), A.ChannelShuffle(p=0.1), A.RandomBrightnessContrast(p=0.1)], p=0.1),
        A.OneOf([A.RandomFog(p=0.1), A.RandomRain(p=0.1),
            # RandomShadow(p=0.1),
            A.RandomSnow(p=0.1), A.RandomSunFlare(p=0.1)], p=0.2),
        A.OneOf([A.CoarseDropout(p=0.1), A.Cutout(p=0.1), A.GridDropout(p=0.1)], p=0.1),
        A.OneOf([A.GridDistortion(p=0.1), A.OpticalDistortion(p=0.1)], p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2, border_mode=cv2.BORDER_REPLICATE),
        # A.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]),
        ToTensorV2(),
    ], p=0.5)], p=1.0)



def initialize_data(folder):
    nclasses = 48
    train_folder = folder + '/train_images'
    test_folder = folder + '/test_images'

    totImgs = np.zeros(nclasses)

    for i in range(nclasses):
        fldr=("/0000" if i<10 else "/000")+str(i)
        for path in os.listdir(train_folder+fldr):
            full_path = os.path.join(train_folder+fldr, path)
            if os.path.isfile(full_path) and not full_path.endswith(".csv"):
                totImgs[i]+=1

    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    fdash = int(f[0:len(f)-4])
                    if fdash<totImgs[int(dirs)]/10:
                    # if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
