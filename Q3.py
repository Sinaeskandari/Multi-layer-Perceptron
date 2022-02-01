# Q3_graded
# Do not change the above line.

from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np

# Q3_graded
# Do not change the above line.
(x_train_1, y_train_1), (x_test_1, y_test_1) = datasets.fashion_mnist.load_data()
y_train_1 = to_categorical(y_train_1, num_classes=10)
y_test_1 = to_categorical(y_test_1, num_classes=10)
image_size = x_train_1.shape[1]
# converting image array from 3d to 2d
x_train_1 = np.reshape(x_train_1, [-1, image_size*image_size])
x_test_1 = np.reshape(x_test_1, [-1, image_size*image_size])
# normalizing image array
x_train_1 = x_train_1 / 255
x_test_1 = x_test_1 / 255

# Q3_graded
# Do not change the above line.

# initalizing model and add layers
model = Sequential()
model.add(layers.Input(shape=(image_size, image_size)))
model.add(layers.Flatten())
# model.add(layers.Dense(128))
# model.add(layers.Activation('relu'))
model.add(layers.Dense(units=128, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Activation('relu'))
model.add(layers.Dense(10))
model.add(layers.Activation('softmax'))

# Q3_graded
# Do not change the above line.

# initializing gradient descent optimizer
sgd_optimizer = SGD(
    learning_rate=0.01,
    momentum=0,
)

loss = 'categorical_crossentropy'

# Q3_graded
# Do not change the above line.

# adding optimizer and loss function to model
model.compile(
    loss=loss,
    optimizer=sgd_optimizer,
    metrics=['accuracy'],
)

# Q3_graded
# Do not change the above line.

# training the model
history = model.fit(
            x=x_train_1,
            y=y_train_1,
            batch_size=64,
            epochs=20,
            validation_split=0.2,
            shuffle=True,
)

# Q3_graded
# Do not change the above line.
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['training', 'validation'], loc='lower right')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['training', 'validation'], loc='upper right')

# Q3_graded
# Do not change the above line.
model.evaluate(
    x=x_test_1,
    y=y_test_1,
    batch_size=64,
)

