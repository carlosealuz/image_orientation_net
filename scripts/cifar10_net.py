from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import cv2
import numpy as np
import os
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import csv


images_folder = '../dataset/train/'
images = glob(os.path.join(images_folder, "*.jpg"))
images_names = [os.path.basename(fn) for fn in images]

test_images_folder = '../dataset/test/'
test_images = glob(os.path.join(test_images_folder, "*.jpg"))
test_images_names = [os.path.basename(fn) for fn in test_images]

print("LOADING TRAIN DATASET")

test_imgs = np.asarray([cv2.imread(img) for img in test_images])

ground_truth = {}
truth_file = open('../dataset/train_truth.csv', "r")
reader = csv.reader(truth_file)
for idx,line in enumerate(reader):
    if idx>0:
        ground_truth[line[0]] = line[1]

train_x_img_names = images[:44116]
train_x = np.asarray([cv2.imread(img) for img in train_x_img_names])
train_y = [ground_truth[i.split('/')[-1]] for i in train_x_img_names]
test_x_img_names = images[44116:]
test_x = np.asarray([cv2.imread(img) for img in test_x_img_names])
test_y = [ground_truth[i.split('/')[-1]] for i in test_x_img_names]

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

pred = model.predict(test_imgs)
predicted_argmax = pred.argmax(axis=1)
prediction_labels = []
for i in predicted_argmax:
  prediction_labels.append(labelNames[i])

with open('test.preds3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(test_images_names, prediction_labels))
