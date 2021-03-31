import PIL.Image as Image
import cv2
import base64
import shutil
import numpy as np
import pickle
import os


# For model stats
stats = {
    "val_loss": [],
    "val_accuracy": 0,
    "train_epoch": 0,
    "train_loss": 0,
    "avg_train_loss": [],
    "running": False,
    "saved": False
}

# For evaluation stats
eval_stats = {'eval_curr': 0, 'eval_tot': 0}


# Image loader
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# Read image from Base64
def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# Convert image to Base64
def writeb64(file):
    img = cv2.imread(file)
    string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
    return string


# Reset stats
def reset_stats():
    with open('stats.bin', 'wb') as f:
        pickle.dump(stats, f)


# Reset eval stats
def reset_eval_stats(i):
    with open('eval_stats' + str(i) + '.bin', 'rb+') as f:
        info = pickle.load(f)
        info['eval_curr'] = 0
        info['eval_tot'] = 0
        f.seek(0)
        pickle.dump(info, f)


# Generate file paths for DataLoader
def generateImgFilePaths(base_path, orig):
    imgpths = []
    if orig:
        nclasses = 43
    else:
        nclasses = max([int(i) for i in os.listdir(base_path)]) + 1
    for i in range(nclasses):
        for path in os.listdir(base_path + str(i)):
            imgpths.append(os.path.join(base_path + str(i), path))
    return imgpths


# Generate file paths for DataLoader (Increment loading)
def generateImgFilePathsInc(base_path, prev_classes):
    imgpths = []
    nclasses = max([int(i) for i in os.listdir(base_path)]) + 1
    for i in range(prev_classes, nclasses):
        for path in os.listdir(base_path + str(i)):
            imgpths.append(os.path.join(base_path + str(i), path))
    return imgpths


# For scanning temp folder and processing
def scan_dir(fldr, num, transformer, cls, mongo, additional_temp):
    for name in os.listdir(fldr):
        path = os.path.join(fldr, name)
        if os.path.isfile(path):
            augment(mongo, path, num, transformer, cls, additional_temp)
            name = path.split('/')[-1]
            shutil.move(path, os.path.join(additional_temp, name))
            mongo.db.transformations.insert({"name": name, "class": cls, "transformations": []})
        else:
            scan_dir(path, num, transformer, cls, mongo, additional_temp)


# For applying augmentations
def augment(mongo, path, num, transformer, cls, additional_temp):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i in range(num):
        x = transformer(image=image)
        transformed = x["image"]
        data = x["replay"]
        name = path.split('/')[-1]
        name = name.split('.')
        name[-2] = name[-2] + str(i)
        name = ".".join(name)
        listOfTransforms = []
        for x in data["transforms"]:
            if x["applied"]:
                if x["__class_fullname__"].split(".")[-1] == "OneOf":
                    for y in x["transforms"]:
                        if y["applied"]:
                            listOfTransforms.append(y["__class_fullname__"].split(".")[-1])
                else:
                    listOfTransforms.append(x["__class_fullname__"].split(".")[-1])
        mongo.db.transformations.insert({"name": name, "class": cls, "transformations": listOfTransforms})
        transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(additional_temp, name), transformed)