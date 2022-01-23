import pandas as pd
import keras
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 10
epochs = 10

def cnn_model(x_train,y_train,x_test,y_test):
    # define the keras model
    model = Sequential()
    model.add(Dense(5, input_dim=98, activation='sigmoid'))
    model.add(Dense(3, activation='relu'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])
    model.summary()


    model_train= model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    test_eval = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])



