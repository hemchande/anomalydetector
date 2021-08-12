# Dense adds layers to the model via Sequential Model
#conv3D creates a convolutionla kernel that produces a tensor of outputs and is over a 3 dimensional space

import keras
from keras import layers
from keras.models import Sequential 

encoding_dim = 32 # 32 neurons in a layer

input_img = keras.Input(shape = (784,))

encoded = layers.Dense(encoding_dim, activation = 'relu')(input_img)
decoded = layers.Dense(784, activation = 'sigmoid')(encoded)


autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)
#decoder = eras.Model(encoded,decoded)


encoded_input = keras.Input(shape =(encoded_dim,))
decoder_layer = autoencoder.layers[-1]#feature weights

decoder = keras.Model(encoded_input, decoder_layer(encoded_input))



autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy')


from keras.datasets import mnist

import numpy as np

(x_train, _),(x_test,_) = mnist.load_data()


x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_test, epochs = 50, batch_size = 256, shuffle = True, validation_data = (x_test, x_train))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize = (20,4))
for i in range(n):
    ax = plt.
