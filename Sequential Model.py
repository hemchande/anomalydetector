mport keras
from keras import layers
from keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Dense(32, input_dim = 784))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(LSTM(32, input_shape(10,64)))

#input dim/length is supported by 2D and 3D layers 
# input shape is (Input length, input dimension)
#batch input shape is the amount of picture data you're inputting 
#LSTM needs input shape arg 

model.compile(loss = 'binary_crossentropy',
             optimizer = 'rmsprop',
             metrics =metrics = ['accuracy'])

#generating random data

data = np.random.random((1000,784))
labels = np.random.randint(2, size = (1000,1))

model.fit(data,labels, nb_epoch = 10, batch_size = 32)
