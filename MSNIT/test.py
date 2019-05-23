import tensorflow as flow
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = flow.keras.datasets.mnist.load_data()

# Keras API only accepts 4 dimensional arrays so we must reshape data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# set up model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=flow.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=flow.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train,y=y_train, epochs=1)
loss, acc = model.evaluate(x_test, y_test)
print(acc) 

plt.imshow(x_test[4444].reshape(28, 28), cmap='Greys')
pred = model.predict(x_test[4444].reshape(1, 28, 28, 1))
print(pred.argmax())

plt.show()
