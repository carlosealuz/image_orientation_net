import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import csv
import argparse

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.callbacks import ModelCheckpoint


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
args = vars(ap.parse_args())

dataset_folder = args["dataset"]

train_images_folder = dataset_folder + '/train/'
images = glob(os.path.join(train_images_folder, "*.jpg"))
images_names = [os.path.basename(fn) for fn in images]

validation_images_folder = dataset_folder + '/test/'
validation_images = glob(os.path.join(validation_images_folder, "*.jpg"))
test_images_names = [os.path.basename(fn) for fn in validation_images]

print("{} TRAIN IMAGES".format(len(images)))
print("{} VALIDATION IMAGES".format(len(validation_images)))

ground_truth = {}
truth_file = open(dataset_folder + '/train_truth.csv', "r")
reader = csv.reader(truth_file)
for idx,line in enumerate(reader):
    if idx>0:
        ground_truth[line[0]] = line[1]

print("LOADING DATASET")

val_imgs = np.asarray([cv2.imread(img) for img in validation_images])

train_x_img_names = images[:44116]
train_x = np.asarray([cv2.imread(img) for img in train_x_img_names])
train_y = [ground_truth[i.split('/')[-1]] for i in train_x_img_names]

test_x_img_names = images[44116:]
test_x = np.asarray([cv2.imread(img) for img in test_x_img_names])
test_y = [ground_truth[i.split('/')[-1]] for i in test_x_img_names]
print("DATASET LOADED")
print("----------")
print("TRAIN SHAPE {}. TEST SHAPE {}".format(train_x.shape, test_x.shape))
print("{} VALIDATION SHAPE".format(val_imgs.shape))

lb = LabelBinarizer()

train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_x.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('softmax'))

opt = SGD(lr=0.001)
checkpoint = ModelCheckpoint('best_rot_checkpoint.hdf5', monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=10, callbacks=callbacks, verbose=1)

labelNames = ['rotated_left', 'rotated_right' ,'upright', 'upside_down']

predictions = model.predict(test_x, batch_size=32)

print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

pred = model.predict(val_imgs)
predicted_argmax = pred.argmax(axis=1)
prediction_labels = []
for i in predicted_argmax:
  prediction_labels.append(labelNames[i])

with open('test.preds3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_images_names, prediction_labels))
