import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2
import os

data_transforms = A.Compose([A.Resize(48, 48),ToTensorV2()], p=1.0)

train_data_transforms = A.Compose([
    A.Resize(48, 48),
    A.Compose([
        A.OneOf([A.MotionBlur(p=0.1), A.GaussianBlur(p=0.1), A.MedianBlur(p=0.1)], p=0.2),
        A.OneOf([A.GaussNoise(p=0.1), A.ISONoise(p=0.1), A.MultiplicativeNoise(p=0.1)], p=0.2),
        A.OneOf([A.ColorJitter(p=0.1), A.HueSaturationValue(p=0.1), A.RandomBrightnessContrast(p=0.1)], p=0.1),
        A.OneOf([A.CoarseDropout(p=0.1), A.Cutout(p=0.1)], p=0.1),
        A.OneOf([A.GridDistortion(p=0.1), A.OpticalDistortion(p=0.1)], p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2, border_mode=cv2.BORDER_REPLICATE),
        ToTensorV2(),
    ], p=0.5)
], p=1.0)


# Custom DataLoader
class ADataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(os.path.normpath(image_filepath).split(os.path.sep)[-2])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


# Transformer
def transform(options):
    p_noise = 1
    p_blur = 1
    p_cut = 1
    if not options["tabs"]["Noise"]:
        p_noise = 0
    if not options["tabs"]["Brightness & Contrast"]:
        options["brightness"] = 0
        options["contrast"] = 0
    if not options["tabs"]["Rotate"]:
        options["rotate"] = 0
    if not options["tabs"]["Blur & Distort"]:
        p_blur = 0
        options["distort"] = 0
    if not options["tabs"]["Hue & Saturation"]:
        options["hue"] = 0
        options["jitter"] = 0
    if not options["tabs"]["Dropout & Cutout"]:
        p_cut = 0
    if not options["tabs"]["Affine & Perspective"]:
        options["affine"] = 0
        options["perspective"] = 0

    return A.ReplayCompose([
        A.RandomBrightness(limit=options["range_brightness"], p=options["brightness"]),
        A.RandomContrast(limit=options["range_contrast"], p=options["contrast"]),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=options["range_rotate"],
                           p=options["rotate"], border_mode=cv2.BORDER_REPLICATE),
        A.OneOf([
            A.MotionBlur(p=options["mblur"]), A.GaussianBlur(p=options["gblur"]), A.MedianBlur(p=options["mdblur"])
        ], p=p_blur),
        A.OpticalDistortion(p=options["distort"]),
        A.OneOf([
            A.GaussNoise(p=options["gnoise"]), A.ISONoise(p=options["inoise"]),
            A.MultiplicativeNoise(p=options["mnoise"])
        ], p=p_noise),
        A.HueSaturationValue(hue_shift_limit=options["range_hue"], sat_shift_limit=options["range_sat"],
                             p=options["hue"]),
        A.ColorJitter(p=options["jitter"]),
        A.OneOf([A.CoarseDropout(p=options["dropout"]), A.Cutout(p=options["cutout"])], p=p_cut),
        A.IAAAffine(p=options["affine"]),
        A.IAAPerspective(p=options["perspective"])
    ])