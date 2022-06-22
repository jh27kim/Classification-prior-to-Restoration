import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras import backend as K


# Set some parameters
im_width = 160
im_height = 160
border = 5
path_train = 'C:/dataset2/DEHAZE/X/'
path_test = 'C:/dataset2/DEHAZE/y/'


def sobel(x):
        weight =  tf.Variable(tf.constant([[-1.0,-1.0,-1.0],  [0,0,0],  [1.0,1.0,1.0],
                                    [-2.0,-2.0,-2.0], [0,0,0],  [2.0,2.0,2.0],
                                    [-1.0,-1.0,-1.0], [0,0,0],  [1.0,1.0,1.0]],shape = [3, 3, 3, 1])) 

        frame=tf.nn.conv2d(x, weight, [1,1,1,1], padding='SAME')
        return frame

def Laplacian(x):
        weight=tf.constant([
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
        ])
        frame=tf.nn.conv2d(x,weight,[1,1,1,1],padding='SAME')
        return frame
        
   
  # Get and resize train images and masks
def get_data(path,path_test, train=True):
    ids = next(os.walk(path))[2]
    id_test = next(os.walk(path_test))[2]
    
    X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    X_half = np.zeros((len(ids), im_height//2, im_width//2, 3), dtype=np.float32)
    
    if train:
        y = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
        y_half = np.zeros((len(ids), im_height//2, im_width//2, 3), dtype=np.float32)
    print('Getting and resizing images ... ')

    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + id_, grayscale=False)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_width, im_height, 3), mode='constant', preserve_range=True)
        x_img_half = resize(x_img, (im_width//2, im_height//2, 3), mode='constant', preserve_range=True)
        X[n] = x_img/255
        X_half[n] = x_img_half/255

    for nt, id_test in tqdm_notebook(enumerate(id_test), total=len(id_test)):
        for j in range(4):
            idx = 4*nt + j
            img = load_img(path_test + id_test, grayscale=False)
            y_img = img_to_array(img)
            y_img = resize(y_img, (im_width, im_height, 3), mode='constant', preserve_range=True)
            y_img_half = resize(y_img, (im_width//2, im_height//2, 3), mode='constant', preserve_range=True)
            y[idx] = y_img / 255
            y_half[idx] = y_img_half / 255
        
    print('Done!')
    if train:
        return X, X_half, y, y_half
    else:
        return X, X_half
    
X, X_half, y, y_half = get_data(path_train, path_test, train=True)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2018)
X_train_half, X_valid_half, y_train_half, y_valid_half = train_test_split(X_half, y_half, test_size=0.2, random_state=2018)
print(X_train.shape, X_valid.shape)
print(X_train_half.shape, X_valid_half.shape)
#Check if the features and the target match 

sel = random.randint(0, len(X_train)-1)

plt.imshow(X_train[sel])
plt.show()
plt.imshow(y_train[sel])
plt.show()

plt.imshow(X_train_half[sel])
plt.show()
plt.imshow(y_train_half[sel])
plt.show()


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
    
    
def unet_en(input_img, n_filters=16, dropout=0.3, batchnorm=True):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    return [c1, c2, c3, c4, c5]
    
    # expansive path
def unet_dec(input_img, layers, n_filters=16, dropout=0.4, batchnorm=True):
    c1, c2, c3, c4, _ = layers
    
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (input_img)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    return c9

def unet_half_en(input_img, n_filters=16, dropout=0.3, batchnorm=True):
    # contracting pat
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)
   
    c4 = conv2d_block(p3, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    return [c1, c2, c3, c4]

def unet_half_dec(input_img, n_filters=16, dropout=0.4, batchnorm=True):
    c1,c2,c3,x = input_img
    
    u4 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (x)
    u4 = concatenate([u4, c3])
    u4 = Dropout(dropout)(u4)
    c4 = conv2d_block(u4, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u5 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c2], axis=3)
    u5 = Dropout(dropout)(u5)
    c5 = conv2d_block(u5, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    u6 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c1])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    u7 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c6)
    
    return u7
    
def dilated_Conv(en1, en2):
    input_img = add([en1, en2])
    dilate3 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(input_img)
    b9 = BatchNormalization()(dilate3)
    b9 = Dropout(rate=0.2)(b9)
    
    dilate4 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=8, kernel_initializer='he_normal')(b9)
    b10 = BatchNormalization()(dilate4)
    b10 = Dropout(rate=0.2)(b10)
    
    dilate5 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=16, kernel_initializer='he_normal')(b10)
    b11 = BatchNormalization()(dilate5)
    b11 = Dropout(rate=0.2)(b11)
    
    dilate6 = Conv2D(64,3, activation='relu', padding='same', dilation_rate=32, kernel_initializer='he_normal')(b11)
    return b11

def reconstruct(dec, dec_half):
    out = concatenate([dec, dec_half])
    out = Conv2D(3, (1, 1), activation='sigmoid')(out)
    return out
    
def loss(img_true, img_pred):
    alpha = 1
    beta = 0
    gamma = 0.05
    def perceptual_loss(img_true, img_pred):
        image_shape = (128, 128, 3)
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
        loss_block3 = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_block3.trainable = False
        loss_block2 = Model(inputs=vgg.input, outputs=vgg.get_layer('block2_conv2').output)
        loss_block2.trainable = False
        loss_block1 = Model(input=vgg.input, outputs = vgg.get_layer('block1_conv2').output)
        loss_block1.trainable = False
        return K.mean(K.square(loss_block1(img_true) - loss_block1(img_pred))) + 2*K.mean(K.square(loss_block2(img_true)
                - loss_block2(img_pred))) + 5*K.mean(K.square(loss_block3(img_true) - loss_block3(img_pred)))

    def ssim_loss(y_true, y_pred):
        return 1-tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    def mae(y_true, y_pred):
        return tf.keras.losses.mean_absolute_error(y_true, y_pred)

    def inference_mse_loss(frame_hr, frame_sr):
        content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2+(1e-3)**2))
        return tf.reduce_mean(content_base_loss)

    derain_loss = inference_mse_loss(img_true, img_pred)
    x_edge = Laplacian(img_true)
    imitation_edge = Laplacian(img_pred)
    edge_loss = inference_mse_loss(img_true, img_pred)
    train_loss = alpha*derain_loss + beta*edge_loss
    
    return train_loss + gamma*ssim_loss(img_true, img_pred)

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

input_img_half = Input((im_height//2, im_width//2, 3), name='img_half')
input_img = Input((im_height, im_width, 3), name='img')

encoding = unet_en(input_img, n_filters=16, dropout=0.05, batchnorm=True)
half_encoding = unet_half_en(input_img_half, n_filters=16, dropout=0.05, batchnorm=True)

bottleneck = dilated_Conv(encoding[-1], half_encoding[-1])

decoding = unet_dec(bottleneck, encoding, n_filters=16, dropout=0.05, batchnorm=True)
half_decoding = unet_half_dec(half_encoding, n_filters=16, dropout=0.05, batchnorm=True)

output = reconstruct(decoding, half_decoding)

model = Model(inputs=[input_img, input_img_half], outputs = [output])
model.compile(optimizer=Adam(), loss=loss, metrics=[PSNR])
model.summary()

#combined_model = Model(inputs=[input_img, input_img_half], outputs=[outputs])

callbacks = [ EarlyStopping(patience=5, verbose=1),
             ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
             ModelCheckpoint('dehze_1st.h5', verbose=1, save_best_only=True, save_weights_only=True) ]

results = model.fit([X_train, X_train_half], 
                    y_train, batch_size=16, epochs=10, 
                    callbacks=callbacks, validation_data=([X_valid, X_valid_half], y_valid), verbose = 1)

model.save('Dehaze_1st.h5')


