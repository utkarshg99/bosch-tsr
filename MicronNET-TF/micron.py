import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
from collections import Counter

import cv2
from albumentations import (
    Compose, OneOf, ShiftScaleRotate,
    GaussianBlur, MotionBlur, MedianBlur,
    RandomFog, RandomRain, RandomShadow, RandomSnow, RandomSunFlare,
    RandomBrightnessContrast, HueSaturationValue, ColorJitter,
    ChannelShuffle, GaussNoise, ISONoise, MultiplicativeNoise,
    CoarseDropout, ChannelDropout, GridDropout, Cutout, GridDistortion, OpticalDistortion
)

import tensorflow.keras.backend as backend
import tensorflow.keras.layers as layers
from tensorflow.keras import models, utils, regularizers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

MODEL_NAME = '2019-08-07-micronnet-model'
TRAIN_IMAGES_DIR = '../data/GTSRB/Final_Training/Images'
TEST_IMAGES_DIR = '../data/GTSRB/Final_Test/Images'
LABEL_NAMES_FILE = '../data/GTSRB/sign_names.txt'
WEIGHTS_PATH = 'weights'
MODEL_DIR = 'models'
WEIGHT_DIR = '2019-08-07-micronnet-model-epoch-07-val_cat_acc-0.988980'
LOAD_CHKPT = '2019-08-07-micronnet-model-epoch-08-val_cat_acc-0.990204'


NUM_CLASSES = 43
NUM_CHANNELS = 3
IMG_SIZE = 48
BATCH_SIZE = 50
NUM_EPOCHS = 10

L2_REG_RATE = 1e-5
EPSILON = 1e-6
LR = 7e-4
DECAY_STEP_SIZE = 2
LR_DECAY_RATE = 0.9996

def get_img_class(img_path):
    return int(img_path.split('/')[-2])

class DataGenerator(utils.Sequence):

    def __init__(self, path_list, labels, n_classes, data_df, batch_size, img_size,
                 shuffle=True, use_augs=False, is_train=True):
        self.img_size = img_size
        self.batch_size = batch_size
        self.labels = labels
        self.path_list = path_list
        self.n_channels = NUM_CHANNELS
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.use_augs = use_augs
        self.on_epoch_end()
        self.is_train = is_train
        self.data_df = data_df
        self.augmentation = self.get_aug(p=0.8)

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

    def get_sign_coords(self, img_path, is_train=True):
        im_name = img_path.split('/')[-1]
        
        if self.is_train:
            pic_class = get_img_class(img_path)
            df_row = self.data_df.loc[pic_class].loc[im_name]
        else:
            df_row = self.data_df.loc[im_name]
        
        x1, y1, x2, y2 = df_row['Roi.X1'], df_row['Roi.Y1'], df_row['Roi.X2'], df_row['Roi.Y2']
        return (x1, y1, x2, y2)

    def get_aug(self, p=0.5):
        # return Compose([
        #     OneOf([
        #         MotionBlur(p=0.1),
        #         GaussianBlur(p=0.1),
        #         MedianBlur(p=0.1),
        #     ], p=0.2),
        #     OneOf([
        #         GaussNoise(p=0.1),
        #         ISONoise(p=0.1),
        #         MultiplicativeNoise(p=0.1),
        #     ], p=0.2),
        #     OneOf([
        #         ColorJitter(p=0.1),
        #         HueSaturationValue(p=0.1),
        #         ChannelShuffle(p=0.1),
        #         RandomBrightnessContrast(p=0.1),
        #     ], p=0.1),
        #     OneOf([
        #         RandomFog(p=0.1),
        #         RandomRain(p=0.1),
        #         # RandomShadow(p=0.1),
        #         RandomSnow(p=0.1),
        #         RandomSunFlare(p=0.1),
        #     ], p=0.2),
        #     OneOf([
        #         CoarseDropout(p=0.1),
        #         Cutout(p=0.1),
        #         GridDropout(p=0.1),
        #     ], p=0.1),
        #     OneOf([
        #         GridDistortion(p=0.1),
        #         OpticalDistortion(p=0.1),
        #     ], p=0.1),
        #     ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20,
        #         p=0.2, border_mode=cv2.BORDER_REPLICATE),
        # # IAASharpen(p=0.2),
        # # HueSaturationValue(p=0.3),
        # ], p=p)
        return Compose([
            OneOf([
                    MotionBlur(p=0.1),
                    GaussianBlur(p=0.1),
                ], p=0.2),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20,
                                p=0.2, border_mode=cv2.BORDER_REPLICATE),
                # IAASharpen(p=0.2),
                HueSaturationValue(p=0.3),
            ], p=p)

    def preprocess_img(self, img_path):
        bgr_img = cv2.imread(img_path)
        
        # hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        # hsv_img[...,2] = cv2.equalizeHist(hsv_img[...,2])
        
        # rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
        yuv_img[...,0] = cv2.equalizeHist(yuv_img[...,0])
        
        rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
        
        return rgb_img

    def crop_n_resize_img(self, img, pt1, pt2, dsize):
        (x1, y1), (x2, y2) = pt1, pt2
        cropped_img = img[y1:y2, x1:x2]
        resized_img = cv2.resize(cropped_img, (dsize,dsize))
        return resized_img

    def augment_img(self, img):
        res_img = self.augmentation(image=img)['image']
        return res_img

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.img_size, self.img_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, idx in enumerate(indexes):
            img_path = self.path_list[idx]
            processed_img = self.preprocess_img(img_path)
            x1, y1, x2, y2 = self.get_sign_coords(img_path, is_train=self.is_train)
            resized_img = self.crop_n_resize_img(processed_img, (x1, y1), (x2, y2), dsize=self.img_size)
            if self.use_augs:
                augmented_img = self.augment_img(resized_img)
            else:
                augmented_img = resized_img

            X[i] = augmented_img / 255
            y[i] = self.labels[idx]

        return X, utils.to_categorical(y, num_classes=self.n_classes)

def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = np.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

def prepare_data():
    annotation_paths = sorted(glob.glob('data/GTSRB/Final_Training/*/*/*csv'))
    annotation_df = pd.concat([pd.read_csv(path, sep=';') for path in annotation_paths], ignore_index=True)
    annotation_df['unique_sign'] = annotation_df['ClassId'].astype(str).str.zfill(5) + "/" + annotation_df['Filename'].str.split('_', expand=True)[0]
    
    train_df = annotation_df.set_index(['ClassId', 'Filename'])
    
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

    labels_dict = Counter(labels)
    class_weight = create_class_weight(labels_dict)

    train_generator = DataGenerator(
        path_list=train_paths,
        labels=train_labels,
        n_classes=NUM_CLASSES,
        data_df=train_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=True,
        use_augs=True,
    )

    dev_generator = DataGenerator(
        path_list=dev_paths,
        labels=dev_labels,
        n_classes=NUM_CLASSES,
        data_df=train_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=True,
        use_augs=False,
    )

    test_df = pd.read_csv('data/GT-final_test.csv', sep=';')

    test_paths = TEST_IMAGES_DIR + os.sep + test_df['Filename']
    test_labels = test_df['ClassId']

    test_df = test_df.set_index('Filename')

    test_generator = DataGenerator(
        path_list=test_paths,
        labels=test_labels,
        n_classes=NUM_CLASSES,
        data_df=test_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=False,
        use_augs=False,
        is_train=False
    )

    return train_generator, dev_generator, test_generator, class_weight

def get_micron():
    input_ = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3), name='data')
    # 1-part
    x = layers.Conv2D(filters=1, kernel_size=(1,1), padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(input_)
    x = layers.BatchNormalization(epsilon=EPSILON)(x)
    x = layers.ReLU()(x)
    # 2-part
    x = layers.Conv2D(filters=29, kernel_size=(5,5), kernel_regularizer=regularizers.l2(L2_REG_RATE))(x)
    x = layers.BatchNormalization(epsilon=EPSILON)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    # 3-part
    x = layers.Conv2D(filters=59, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(x)
    x = layers.BatchNormalization(epsilon=EPSILON)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    # 4-part
    x = layers.Conv2D(filters=74, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(L2_REG_RATE))(x)
    x = layers.BatchNormalization(epsilon=EPSILON)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    # 5-part
    x = layers.Flatten()(x)
    x = layers.Dense(300, kernel_regularizer=regularizers.l2(L2_REG_RATE))(x)
    x = layers.BatchNormalization(epsilon=EPSILON)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(300)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(NUM_CLASSES)(x)
    x = layers.Softmax()(x)
    return models.Model(inputs=input_, outputs=x)

def create_lr_decay(decay_step_size, lr_decay_rate):
    def lr_decay(epoch, lr):
        if epoch % decay_step_size == 0:
            lr = lr * lr_decay_rate
        return lr
    return lr_decay

def f1(y_true, y_pred):
    y_pred = backend.round(y_pred)
    tp = backend.sum(backend.cast(y_true*y_pred, 'float'), axis=0)
    tn = backend.sum(backend.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = backend.sum(backend.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + EPSILON)
    r = tp / (tp + fn + EPSILON)

    f1 = 2*p*r / (p+r+EPSILON)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return backend.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = backend.sum(backend.cast(y_true*y_pred, 'float'), axis=0)
    tn = backend.sum(backend.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = backend.sum(backend.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2*p*r / (p+r+backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - backend.mean(f1)

class Metrics(callbacks.Callback):

    def __init__(self, val_data):
        super(Metrics, self).__init__()
        self.validation_data = val_data

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_test_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs):
        # 5.4.1 For each validation batch
        for batch_index in range(0, len(self.validation_data)):
            # 5.4.1.1 Get the batch target values
            temp_data, temp_targ = self.validation_data.__getitem__(batch_index)
            # 5.4.1.2 Get the batch prediction values
            temp_predict = (np.asarray(self.model.predict(
                                temp_data))).round()
            # 5.4.1.3 Append them to the corresponding output objects
            if(batch_index == 0):
                val_targ = temp_targ
                val_predict = temp_predict
            else:
                val_targ = np.vstack((val_targ, temp_targ))
                val_predict = np.vstack((val_predict, temp_predict))

        val_f1 = round(f1_score(val_targ, val_predict, average='macro'), 4)
        val_recall = round(recall_score(val_targ, val_predict, average='macro'), 4)
        val_precis = round(precision_score(val_targ, val_predict, average='macro'), 4)

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precis)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["val_f1"] = val_f1
        logs["val_recall"] = val_recall
        logs["val_precis"] = val_precis

        print("— val_f1: {} — val_precis: {} — val_recall {}".format(
                 val_f1, val_precis, val_recall))
        return
    
    def on_test_end(self, logs):
                # 5.4.1 For each validation batch
        for batch_index in range(0, len(self.validation_data)):
            # 5.4.1.1 Get the batch target values
            temp_data, temp_targ = self.validation_data.__getitem__(batch_index)
            # 5.4.1.2 Get the batch prediction values
            temp_predict = (np.asarray(self.model.predict(
                                temp_data))).round()
            # 5.4.1.3 Append them to the corresponding output objects
            if(batch_index == 0):
                val_targ = temp_targ
                val_predict = temp_predict
            else:
                val_targ = np.vstack((val_targ, temp_targ))
                val_predict = np.vstack((val_predict, temp_predict))

        val_f1 = round(f1_score(val_targ, val_predict, average='macro'), 4)
        val_recall = round(recall_score(val_targ, val_predict, average='macro'), 4)
        val_precis = round(precision_score(val_targ, val_predict, average='macro'), 4)

        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precis)

        # Add custom metrics to the logs, so that we can use them with
        # EarlyStop and csvLogger callbacks
        logs["test_f1"] = val_f1
        logs["test_recall"] = val_recall
        logs["test_precis"] = val_precis

        print("— test_f1: {} — test_precis: {} — test_recall {}".format(
                 val_f1, val_precis, val_recall))
        return

    # def on_test_end(self, logs):
    #     print("— test_f1: {} — test_precis: {} — test_recall {}".format(
    #              np.mean(np.array(val_f1s)), np.mean(np.array(val_precisions)), np.mean(np.array(val_recalls))))
    #     return

def train_micron(train_generator, dev_generator, class_weight):
    # model = get_micron()
    model = models.load_model('weights/'+WEIGHT_DIR + os.sep + LOAD_CHKPT+'.hdf5')
    lr_decay = create_lr_decay(decay_step_size=2, lr_decay_rate=0.9996)
    weights_path = os.path.join(WEIGHTS_PATH, LOAD_CHKPT)
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    model_path = os.path.join(weights_path, MODEL_NAME + '-epoch-{epoch:02d}-val_cat_acc-{val_categorical_accuracy:.6f}.hdf5')
    model_acc_ckpt = callbacks.ModelCheckpoint(model_path, monitor='val_categorical_accuracy', save_best_only=True, verbose=1)
    lr_sched = callbacks.LearningRateScheduler(lr_decay)
    callback_list = [
        Metrics(dev_generator),
        model_acc_ckpt,
        lr_sched,
    ]
    backend.clear_session()
    opt = optimizers.SGD(
        learning_rate=LR,
        momentum=0.9,
        nesterov=True,
    )
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
    history = model.fit(
        train_generator,
        epochs=NUM_EPOCHS,
        callbacks=callback_list,
        validation_data=dev_generator,
        class_weight=class_weight,
    )
    val_acc_index = np.argmax(history.history['val_categorical_accuracy'])
    max_val_acc = history.history['val_categorical_accuracy'][val_acc_index]
    max_val_f1 = history.history['val_f1'][val_acc_index]
    savepath = os.path.join(MODEL_DIR, MODEL_NAME + str(max_val_acc) +' ' + str(max_val_f1) + ' ' + str(NUM_EPOCHS))
    model.save(savepath)

def eval_micron(test_generator):
    model = models.load_model('weights/'+WEIGHT_DIR + os.sep + LOAD_CHKPT+'.hdf5')
    test_loss, test_cat_acc = model.evaluate(test_generator, callbacks=[Metrics(test_generator)])
    print("Test results: ", test_loss, test_cat_acc)


if __name__ == "__main__":
    train_gen, dev_gen, test_gen, class_w = prepare_data()
    is_train = int(input("Train? (1/0): "))
    if is_train == 1:
        train_micron(train_gen, dev_gen, class_w)
        eval_micron(test_gen)
    else:
        eval_micron(test_gen)
        