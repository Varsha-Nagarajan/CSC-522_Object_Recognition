#import all that is needed
import numpy as np
np.random.seed(7)
from tensorflow import set_random_seed
set_random_seed(7)
import pandas as pd
import keras
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import losses
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.metrics import top_k_categorical_accuracy
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

img_x, img_y = 224, 224
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

batch_size = 24
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

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

		
base_model = ResNet50(weights='imagenet', include_top=False)

x = GlobalAveragePooling2D()(x)

# get layers and add average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add fully-connected layer
x = Dense(512, activation='relu')(x)

# add output layer
predictions = Dense(len(np.unique(train_generator.classes)), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
	

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

#Fits the model on batches with real-time data augmentation
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN+1,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID+1,
                    epochs=10)
					
layer_num = len(model.layers)
for layer in model.layers[:int(layer_num * 0.9)]:
    layer.trainable = False
for layer in model.layers[int(layer_num * 0.9):]:
    layer.trainable = True
		
model.compile(loss=losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

			  
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN+1,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID+1,
                    epochs=20, callbacks=[history])

#Saving model weights					
modeldir="model"
model.save_weights(modeldir+"/fineTuned_resnet.h5")

#Evaluating the model
scores = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST+1)
print(scores)
