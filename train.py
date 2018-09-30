from __future__ import print_function

import keras
import numpy as np
import scipy as sp
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input, Add, MaxPooling2D, LeakyReLU
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG19
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import regularizers
from keras import initializers

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skimage import io

import csv

subset = 57382
#subset = 1000

with open('tag_list.csv') as f:
    labels=[tuple(line) for line in csv.reader(f)]
	
labels = labels[0:subset]


mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)



#print(mlb.classes_)

#print(mlb.transform([("wild", "yum")]))

images = []
for i in range(subset):
	image = io.imread("processed/"+str(i)+".jpg")
	if (len(image.shape) < 3):
		image = np.expand_dims(image, axis=2)
		image = np.repeat(image, 3, axis=2)
	if i%10000 == 0:
		print("Loading Images...", i)
	images.append(image)

images = np.array(images)
labels = np.array(labels)	

#print(images)
	
seed = 42

(x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.1, random_state=seed)

#x_train = np.array(x_train)
#x_test = np.array(x_test)
#y_train = np.array(y_train)
#y_test = np.array(y_test)

batch_size = 64
num_classes = len(mlb.classes_)
epochs = 100

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(y_test[0:10])

img_rows, img_cols, img_chans = 256, 256, 3

# if K.image_data_format() == 'channels_first':
    # x_train = x_train.reshape(x_train.shape[0], img_chans, img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], img_chans, img_rows, img_cols)
    # input_shape = (img_chans, img_rows, img_cols)
# else:
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chans)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chans)
    # input_shape = (img_rows, img_cols, img_chans)

print(x_test.shape)
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_train /= 255
x_test /= 255
x_train -= 0.5
x_test -= 0.5

y_train = y_train.astype('float16')
y_test = y_test.astype('float16')
y_train *= 2
y_test *= 2
y_train -= 1
y_test -= 1
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

reg = 0.001

input = Input(shape=(256, 256, 3))
x0 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(input)
x = x0
x = LeakyReLU(alpha = 0.1)(x)

#for i in range(4):
#	x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
#	x = LeakyReLU(alpha = 0.1)(x)
#	x = Add()([x0, x])
	
x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

#x = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid')(x)
x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.1)(x)

x = Flatten()(x)

x = Dense(2048, kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.2)(x)
#x = Dropout(0.2)(x)

x = Dense(2048, kernel_initializer=initializers.lecun_normal(seed), kernel_regularizer=regularizers.l2(reg))(x)
x = LeakyReLU(alpha = 0.2)(x)
#x = Dropout(0.2)(x)

output = Dense(num_classes, activation='tanh')(x)

model = Model(inputs=input, outputs=output)

model.summary()

#model.compile(loss='mean_squared_error',
model.compile(loss='hinge',
              optimizer=SGD(lr=0.05, clipvalue=0.0001, decay=1e-7, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

filepath="model-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
					callbacks=callbacks_list)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])