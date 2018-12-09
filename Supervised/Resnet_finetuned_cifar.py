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
from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# resize data to be fed into ResNet50 which accepts only 224x224
def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 224, 224, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

# resize train and  test data
x_train = resize_data(x_train)
x_test = resize_data(x_test)
x_train = x_train / 255.0
x_test = x_test / 255.0

num_classes = y_test.shape[1]


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

#Finetuning ResNet50	
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

#Fitting the finetuned model
model.fit(x_train, y_train, epochs = 10, validation_split=0.25, callbacks=[history])
			
#Identufying and setting layers as trainable and non-trainable. Ideally, we do not want to retrain the earlier layers as they learn only high level features
#like edge and corners and we need this information. So we specifically train the later layers to make them learn more specific features.			
layer_num = len(model.layers)
for layer in model.layers[:int(layer_num * 0.9)]:
    layer.trainable = False
for layer in model.layers[int(layer_num * 0.9):]:
    layer.trainable = True
		
model.compile(loss=losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

			  
model.fit(x_train, y_train, epochs = 20, validation_split=0.25, callbacks=[history])

#Saving the model
modeldir="model"
model.save_weights(modeldir+"fineTuned_resnet_cifar10.h5")
#Evaluating the model performance
scores = model.evaluate(x_test, y_test)
print(scores)

#Uncomment if you need to plot the training accuracy against the number of epochs
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.plot(range(1, epochs+1), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
#print(history.acc)
