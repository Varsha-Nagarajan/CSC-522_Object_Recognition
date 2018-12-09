# Plain Classifier model for Caltech256
from __future__ import print_function
import numpy as np
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)
import keras
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import losses
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

img_x, img_y = 128, 128
batch_size = 32
epochs = 50
num_classes = 256
sample_datagen = ImageDataGenerator()

#Extracting 10000 samples to fit the train,test,validation sets for performing featurewise_std_normalization and featurewise_center
def generate_Samples(datagen):
    for Sample_X, Sample_Y in datagen.flow_from_directory(directory="train/",
    target_size=(img_x, img_y),
    batch_size = 10000,
    class_mode='categorical'):
        x = Sample_X
        break
    return x
  
Sample_X = generate_Samples(sample_datagen)

#Generators for Train, Validation and Test Set
train_datagen = ImageDataGenerator(rescale=1./255, featurewise_std_normalization=True, featurewise_center=True, rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
train_datagen.fit(Sample_X)
validate_datagen = ImageDataGenerator(rescale=1./255, featurewise_std_normalization=True, featurewise_center=True, rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
validate_datagen.fit(Sample_X)
test_datagen = ImageDataGenerator(rescale=1./255, featurewise_std_normalization=True, featurewise_center=True, rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen.fit(Sample_X)


train_generator = train_datagen.flow_from_directory(
    directory="train/",
    target_size=(img_x, img_y),
    batch_size = batch_size,
    class_mode='categorical', seed = 42)

validation_generator = validate_datagen.flow_from_directory(
        'val/',
        target_size=(img_x, img_y),
        batch_size=batch_size,
        class_mode='categorical', seed = 42)

test_generator = test_datagen.flow_from_directory(
        'test/',
        target_size=(img_x, img_y),
        batch_size=1, seed = 42)



K.set_image_data_format('channels_last')
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), padding="same", strides=(1,1), input_shape=(img_x,img_y,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

print(model.summary())

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

#Fits the model on batches with real-time data augmentation
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN+1,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID+1,
                    epochs=epochs, callbacks=[history])

#Saving model weights
modeldir="model"
model.save_weights(modeldir+"/PlainClassifier_Caltech256.h5")

#Evaluating the model
scores = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST+1)
print(scores)

#Uncomment if you need to plot the training accuracy against the number of epochs
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.plot(range(1, epochs+1), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
#print(history.acc)
