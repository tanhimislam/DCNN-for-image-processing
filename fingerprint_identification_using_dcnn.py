# -*- coding: utf-8 -*-
"""20thMarch_final_fingerprint identification_CNN.ipynb

"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage.filters import threshold_otsu
import numpy as np
from glob import glob
from scipy import misc
from matplotlib.patches import Circle,Ellipse
from matplotlib.patches import Rectangle
import os
from PIL import Image
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
# %matplotlib inline
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

trainlist = '101_1.tif', '101_2.tif', '101_3.tif', '101_4.tif', '101_5.tif', '101_6.tif', '101_8.tif', '101_8.tif' 
testlist = '102_1.tif', '102_2.tif',

data_train = trainlist

data_test = testlist

images = []
def read_images(data_train):
    for i in range(len(data_train)):
        img = misc.imread(data_train[i])
        img = misc.imresize(img,(224,224))
        images.append(img)
    return images

imagesTest = []
def read_imagesTest(data_test):
    for i in range(len(data_test)):
        img1 = misc.imread(data_test[i])
        img1 = misc.imresize(img1,(224,224))
        imagesTest.append(img1)
    return imagesTest

images = read_images(data_train)
images_arr = np.asarray(images)
images_arr = images_arr.astype('float32')
images_arr.shape

imagesTest = read_imagesTest(data_test)
images_arrTest = np.asarray(imagesTest)
images_arrTest = images_arrTest.astype('float32')
images_arrTest.shape

print("Dataset (images) shape: {shape}".format(shape=images_arr.shape))

print("Dataset (images) shape: {shape}".format(shape=images_arrTest.shape))

images_arr = images_arr.reshape(-1, 224,224, 1)
images_arr.shape

images_arrTest = images_arrTest.reshape(-1, 224,224, 1)
images_arrTest.shape

images_arr.dtype

np.max(images_arr)

np.max(images_arrTest)

images_arrTest = images_arrTest / np.max(images_arr)

test_Y_y = images_arrTest

images_arr = images_arr / np.max(images_arr)

np.max(images_arrTest), np.min(images_arrTest)

np.max(images_arr), np.min(images_arr)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(images_arr,
                                                             images_arr,
                                                             test_size=0.2,
                                                             random_state=13)

batch_size = 128
epochs = 200
inChannel = 1
x, y = 224, 224
input_img = Input(shape = (x, y, inChannel))

def fingerprint(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

fingerprint = Model(input_img, fingerprint(input_img))
fingerprint.compile(loss='mean_squared_error', optimizer = RMSprop())

fingerprint.summary()

fingerprint_train = fingerprint.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

loss = fingerprint_train.history['loss']
val_loss = fingerprint_train.history['val_loss']
epochs = range(200)
plt.figure(figsize=(20,10))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

fingerprint = fingerprint.save_weights('fingerprint.h5')



fingerprint.load_weights('fingerprint.h5')

fingerprint.compile(loss='mean_squared_error', optimizer = RMSprop())

pred = fingerprint.predict(valid_X)



