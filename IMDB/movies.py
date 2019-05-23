# Code derived from Deep Learning with Python textbook
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras import Sequential

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

x_train = sequence.pad_sequences(x_train, maxlen=500)
x_test = sequence.pad_sequences(x_test, maxlen=500)

model = Sequential()
model.add(Embedding(10000, 32))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
print model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

result = model.fit(x_train, y_train, batch_size = 128, epochs = 20, validation_split = 0.2)

score, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print 'Test score:', score
print 'Test accuracy', accuracy

acc = result.history['acc']
val_acc = result.history['val_acc']
loss = result.history['loss']
val_loss = result.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
