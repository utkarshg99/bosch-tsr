import glob
import os
import random as rn
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as kl
from albumentations import (
    ShiftScaleRotate, GaussianBlur, MotionBlur, IAASharpen,
    HueSaturationValue, HorizontalFlip, OneOf, Compose
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def get_sign_names_mapping(filepath):
    idx2sign_name = {}
    with open(filepath, 'r') as f:
        for idx, sign_name in enumerate(f):
            idx2sign_name[idx] =  sign_name.strip()
    return idx2sign_name

TRAIN_IMAGES_DIR = 'data/GTSRB/Final_Training/Images'
NUM_CLASSES = 43

n_rows, n_cols = 11, 4

idx2sign_name = get_sign_names_mapping(filepath='data/GTSRB/sign_names.txt')

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,18), constrained_layout=True)
fig.subplots_adjust(hspace=0.5, wspace=0.0)

for idx, class_path in enumerate(sorted(glob.glob(TRAIN_IMAGES_DIR + '/*'))):
    img_path = np.random.choice(glob.glob(class_path + '/*.ppm'))
    raw_img = cv2.imread(img_path)
#     rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    axs[idx // n_cols, idx % n_cols].set_title(idx2sign_name[idx])
    axs[idx // n_cols, idx % n_cols].grid(False)
    axs[idx // n_cols, idx % n_cols].axis('off')
    axs[idx // n_cols, idx % n_cols].imshow(rgb_img)
# axs[NUM_CLASSES // n_cols, NUM_CLASSES % n_cols].grid(False)
# axs[NUM_CLASSES // n_cols, NUM_CLASSES % n_cols].axis('off')
# plt.show()

annotation_paths = sorted(glob.glob('data/GTSRB/Final_Training/*/*/*csv'))

annotation_df = pd.concat([pd.read_csv(path, sep=';') for path in annotation_paths], ignore_index=True)
annotation_df.head()

annotation_df['unique_sign'] = annotation_df['ClassId'].astype(str).str.zfill(5) + "/" + annotation_df['Filename'].str.split('_', expand=True)[0]
annotation_df.head()



sign_class_number = 1

n_rows, n_cols = 2, 5

sign_samples = annotation_df.query('ClassId == @sign_class_number')  # pd.Dataframe
# print(sign_samples.head())
sign_samples = sign_samples['unique_sign'].sample(n=n_rows*n_cols, random_state=42) # pd.Series

fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,6), constrained_layout=True)
fig.subplots_adjust(hspace=0.3, wspace=0.5)
fig.suptitle(idx2sign_name[sign_class_number])

for idx, sign in enumerate(sign_samples.values):
    filename = np.random.choice(annotation_df.query('unique_sign == @sign')['Filename'].values, size=1)[0]
    img_path = os.path.join(TRAIN_IMAGES_DIR, '{0:0>5}'.format(sign_class_number), filename) #train_images_dir
    raw_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    axs[idx // n_cols, idx % n_cols].grid(False)
    axs[idx // n_cols, idx % n_cols].axis('off')
    axs[idx // n_cols, idx % n_cols].imshow(rgb_img)
plt.show()

train_df = annotation_df.set_index(['ClassId','Filename'])
train_df.loc[0].loc['00000_00000.ppm']



def get_img_class(img_path):
    return int(img_path.split('/')[-2])
    
get_img_class(os.path.join(TRAIN_IMAGES_DIR, '00011/00000_00000.ppm'))

def get_sign_coords(img_path, is_train=True):
    im_name = img_path.split('/')[-1]
    
    if is_train:
        pic_class = get_img_class(img_path)
        df_row = train_df.loc[pic_class].loc[im_name]
    else:
        df_row = test_df.loc[im_name]
    
    x1, y1, x2, y2 = df_row['Roi.X1'], df_row['Roi.Y1'], df_row['Roi.X2'], df_row['Roi.Y2']
    return (x1, y1, x2, y2)
    
get_sign_coords(os.path.join(TRAIN_IMAGES_DIR, '00011/00000_00000.ppm'))

def get_aug(p=0.5):
    return Compose([
        OneOf([
            MotionBlur(p=0.1),
            GaussianBlur(p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20,
                         p=0.2, border_mode=cv2.BORDER_REPLICATE),
        IAASharpen(p=0.2),
        HueSaturationValue(p=0.3),
    ], p=p)

augmentation = get_aug(p=0.8)

def preprocess_img(img_path):
    bgr_img = cv2.imread(img_path)
    
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_img[...,2] = cv2.equalizeHist(hsv_img[...,2])
    
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb_img

def crop_n_resize_img(img, pt1, pt2, dsize):
    (x1, y1), (x2, y2) = pt1, pt2
    cropped_img = img[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, (dsize, dsize))
    
    return resized_img
    

def augment_img(augmentation, img):
    res_img = augmentation(image=img)['image']
    return res_img

img_path = os.path.join(TRAIN_IMAGES_DIR, '00017/00000_00015.ppm')
x1, y1, x2, y2 = get_sign_coords(img_path)

processed_img = preprocess_img(img_path)
resized_img = crop_n_resize_img(processed_img, (x1,y1), (x2,y2), dsize=48)
augmented_img = augment_img(augmentation, resized_img)

plt.imshow(augmented_img)
plt.show()

class DataGenerator(Sequence):

    def __init__(self, path_list, labels, n_classes, batch_size, img_size,
                 n_channels, shuffle=True, use_augs=False, is_train=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.labels = labels
        self.path_list = path_list
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_augs = use_augs
        self.is_train = is_train
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.path_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, idx in enumerate(indexes):
            img_path = self.path_list[idx]
            processed_img = preprocess_img(img_path)
            x1, y1, x2, y2 = get_sign_coords(img_path, is_train=self.is_train)
            resized_img = crop_n_resize_img(processed_img, (x1,y1), (x2,y2), dsize=self.img_size)
            augmented_img = augment_img(augmentation, resized_img)
            
            X[i] = augmented_img / 255
            y[i] = self.labels[idx]

        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

sign_list = annotation_df['unique_sign'].unique()
labels = [get_img_class(sign_name) for sign_name in sign_list]

train_signs, dev_signs = train_test_split(sign_list, train_size=0.75, 
                                          stratify=labels, random_state=123)

train_paths = []
for sign_name in train_signs:
    train_paths.extend(glob.glob(os.path.join(TRAIN_IMAGES_DIR, sign_name + '*.ppm')))
    
dev_paths = []
for sign_name in dev_signs:
    dev_paths.extend(glob.glob(os.path.join(TRAIN_IMAGES_DIR, sign_name + '*.ppm')))

train_labels = [get_img_class(sign_name) for sign_name in train_paths]
dev_labels = [get_img_class(sign_name) for sign_name in dev_paths]
len(train_labels), len(dev_labels), len(train_paths), len(dev_paths)

def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = np.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


labels_dict = Counter(labels)

class_weight = create_class_weight(labels_dict)

IMG_SIZE = 48
l2_reg_rate = 1e-5
eps = 1e-6

def get_micronet():
    input_ = kl.Input(shape=(IMG_SIZE,IMG_SIZE,3), name='data')
    # 1-part
    x = kl.Conv2D(filters=1, kernel_size=(1,1), padding='same', kernel_regularizer=l2(l2_reg_rate))(input_)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    # 2-part
    x = kl.Conv2D(filters=29, kernel_size=(5,5), kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling2D(pool_size=3, strides=2)(x)
    # 3-part
    x = kl.Conv2D(filters=59, kernel_size=(3,3), padding='same', kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling2D(pool_size=3, strides=2)(x)
    # 4-part
    x = kl.Conv2D(filters=74, kernel_size=(3,3), padding='same', kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling2D(pool_size=3, strides=2)(x)
    # 5-part
    x = kl.Flatten()(x)
    x = kl.Dense(300, kernel_regularizer=l2(l2_reg_rate))(x)
    x = kl.BatchNormalization(epsilon=eps)(x)
    x = kl.ReLU()(x)
    x = kl.Dense(300)(x)
    x = kl.ReLU()(x)
    x = kl.Dense(NUM_CLASSES)(x)
    x = kl.Softmax()(x)
    return Model(inputs=input_, outputs=x)

BATCH_SIZE = 50

train_generator = DataGenerator(
    path_list=train_paths,
    labels=train_labels,
    n_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    n_channels=3,
    shuffle=True,
    use_augs=True,
)

dev_generator = DataGenerator(
    path_list=dev_paths,
    labels=dev_labels,
    n_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    n_channels=3,
    shuffle=True,
    use_augs=False,
)

def create_lr_decay(decay_step_size, lr_decay_rate):
    def lr_decay(epoch, lr):
        if epoch % decay_step_size == 0:
            lr = lr * lr_decay_rate
        return lr
    return lr_decay

lr = 0.0007
lr_decay = create_lr_decay(decay_step_size=2, lr_decay_rate=0.9996)

weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

model_name = '2019-08-07-micronnet-model'

monitor = 'val_categorical_accuracy'
model_path = os.path.join(weights_path, model_name + '-epoch-{epoch:02d}-val_cat_acc-{val_categorical_accuracy:.3f}.hdf5')

model_checkpoint = ModelCheckpoint(model_path, monitor=monitor, save_best_only=True, verbose=1)
lr_sched = LearningRateScheduler(lr_decay)

callbacks = [
    model_checkpoint,
    lr_sched,
]

K.clear_session()

model = get_micronet()

opt = tensorflow.keras.optimizers.SGD(
    learning_rate=lr,
    momentum=0.9,
    nesterov=True,
)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit_generator(
    generator=train_generator,
    epochs=100,
    callbacks=callbacks,
    validation_data=dev_generator,
    class_weight=class_weight,
)

TEST_IMAGES_DIR = 'data/GTSRB/Final_Test/Images'

test_df = pd.read_csv('data/GT-final_test.csv', sep=';')
test_df.head()

test_paths = TEST_IMAGES_DIR + os.sep + test_df['Filename']
test_labels = test_df['ClassId']

(test_paths[2], test_labels[2]), (test_paths[4], test_labels[4]),

BATCH_SIZE = 50
IMG_SIZE = 48

test_generator = DataGenerator(
    path_list=test_paths,
    labels=test_labels,
    n_classes=NUM_CLASSES,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    n_channels=3,
    shuffle=False,
    use_augs=False,
    is_train=False
)

test_loss, test_cat_acc = loaded_model.evaluate_generator(test_generator)
print('Точность работы классификатора на тестовой выборке: {:.4%}'.format(test_cat_acc))