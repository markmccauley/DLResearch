import os
import numpy
import h5py
import functions
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.models import model_from_json
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

path = '/home/mark/Research'
data_dir = path + '/data'

train = True
load_all = True

weight_matrix, word_index = functions.load_embeddings(data_dir + '/glove.6B.100d.txt')

data = functions.read_data(data_dir)
train, test, val = functions.split_data(data, .8, data_dir)

train = train.reset_index()
test = test.reset_index()
val = val.reset_index()

#max_length, avg_words, seq_length = functions.maxLen(data)
train_x = functions.pipeline(train, word_index, weight_matrix)
test_x = functions.pipeline(test, word_index, weight_matrix)
val_x = functions.pipeline(val, word_index, weight_matrix)

train_y = functions.labels(train)
test_y = functions.labels(test)
val_y = functions.labels(val)

print 'Training data: '
print train_x.shape
print train_y.shape

print 'Classes: '
print numpy.unique(train_y.shape[1])

model = Sequential()
model.add(Embedding(len(weight_matrix), 100, weights=[weight_matrix], input_length=1, trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print model.summary()

best_model = keras.callbacks.ModelCheckpoint(data_dir + '/best.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

result = model.fit(train_x, train_y, batch_size=5, epochs=10, validation_data=(val_x, val_y), callbacks=[best_model])

model.save_weights(data_dir + '/best.hdf5')

score, accuracy = model.evaluate(test_x, test_y, batch_size=5)
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