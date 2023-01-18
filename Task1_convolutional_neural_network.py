from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt


# information of the images and iteration
height = 128
width = 128
channels = 3
num_class = 10
batch = 64
epochs = 50


# location of the sets
train_set = './archive/training/training'
validation_set = './archive/validation/validation'

train_data = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

train_generator = train_data.flow_from_directory(
    train_set,
    target_size=(height, width),
    class_mode='categorical',
    batch_size=batch,
    seed=8,
    shuffle=True,
)

validation_data = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

validation_generator = validation_data.flow_from_directory(
    validation_set,
    target_size=(height, width),
    batch_size=batch,
    seed=8,
    shuffle=False,
    class_mode='categorical'
)

train_num = train_generator.samples
validation_num = validation_generator.samples

# build the model
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32,
                        kernel_size=3,
                        padding='same',
                        activation='relu',
                        input_shape=[width, height, channels]),
    keras.layers.Conv2D(filters=32,
                        kernel_size=3,
                        padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=64,
                        kernel_size=3,
                        padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=64,
                        kernel_size=3,
                        padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_class, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# training
history = model.fit(
    train_generator,
    steps_per_epoch= train_num // batch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= validation_num // batch
)


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8,5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history, 'accuracy', epochs, 0, 1.1)
plot_learning_curves(history, 'loss', epochs, 0, 5)

