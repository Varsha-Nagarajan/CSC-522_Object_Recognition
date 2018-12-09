#[2.558237690734863, 0.1, 0.5]
# val accuarcy 0.7268 - val_top_k_categorical_accuracy: 0.9752
#[0.16813333340644837, 0.24679999944448472, 0.3005333327317238, 0.34263999973535536, 0.37986666697263716, 0.4105333338499069, 0.43090666827201846, 0.4558666682672501, 0.47877333515167236, 0.4989600015449524, 0.5163200017261506, 0.5335200012874604, 0.5448000013160705, 0.5528266673755646, 0.5719733347225189, 0.5801600012969971, 0.5910400011634827, 0.600773333902359, 0.6026400011539459, 0.6152800007247925, 0.6055466677093506, 0.6247200008964539, 0.6319466673088073, 0.6387200008964539, 0.6379200001907349, 0.6452000008773804, 0.6459466667079925, 0.6533066679954529, 0.6581066665554046, 0.6582133338737488, 0.6693600006866455, 0.6690133334064484, 0.6747200009536743, 0.6779733335494995, 0.6853599995422364, 0.6876266665649414, 0.6936000004959106, 0.6947199999046326, 0.6678133332824707, 0.6582933338928223, 0.6862400004959106, 0.6940533340263366, 0.6982133325004578, 0.6990400001335144, 0.7003733336067199, 0.707626666469574, 0.7162133333587647, 0.7184000000762939, 0.7201333335113526, 0.7189866662979126]

# [0.955399033650637, 0.6734449760765551, 0.969896331738437] - accuracy when augmentation was done in test too.

# Cifar 100 - [5.312142794799804, 0.01, 0.05]
# val accuracy : 0.4469 - val_top_k_categorical_accuracy: 0.7526
#[0.029386666666666665, 0.055440000002384186, 0.07544000000238418, 0.09392000000317892, 0.11834666667143504, 0.13882666666666665, 0.15765333333651224, 0.18002666666984557, 0.1956, 0.2128533333349228, 0.22888000000953673, 0.2435733333301544, 0.2589866666698456, 0.2725066666634878, 0.27898666666348776, 0.2921600000047684, 0.3029866666762034, 0.3171999999968211, 0.32586666666348774, 0.3385333333269755, 0.34786666666666666, 0.3557600000031789, 0.35600000000317894, 0.3612266666762034, 0.37426666666030883, 0.38021333334287005, 0.3900799999968211, 0.3965866666730245, 0.40800000000953673, 0.40282666666984557, 0.4174399999968211, 0.41085333334287005, 0.4283999999968211, 0.43200000000635785, 0.43818666666666667, 0.4410933333269755, 0.4484533333237966, 0.4552533333269755, 0.454506666653951, 0.4544, 0.4659200000031789, 0.4693333333269755, 0.4735466666603088, 0.47559999999364216, 0.48146666665712995, 0.46946666666030884, 0.4799466666698456, 0.48744000000317894, 0.49424, 0.4981333333333333]
#[2.5082122032341965, 0.37559808612440193, 0.6840111642743222] -  accuracy when augmentation was done in test too.


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
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

batch_size = 24
epochs=50
num_classes = 10 
#num_classes = 100
img_x, img_y = 32, 32
(x_train, y_train), (X_test, y_test) = cifar10.load_data()
#(x_train, y_train), (X_test, y_test) = cifar100.load_data()

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=17)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# Defining ImageDataGenerators for real-time data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
		
val_datagen = ImageDataGenerator(rescale=1./255, rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
		
test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

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

#Fits the model on batches with real-time data augmentation
model.fit_generator(generator=train_datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=(len(X_train) / batch_size) + 1, validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size), validation_steps=(len(X_val) / batch_size) + 1, epochs=epochs,  callbacks=[history])


#Saving model weights
modeldir="model"
model.save_weights(modeldir+"/PlainClassifier_Cifar10.h5")
#model.save_weights(modeldir+"/PlainClassifier_Cifar100.h5")

#Evaluating the model
scores = model.evaluate_generator(generator=test_datagen.flow(X_test, y_test, batch_size=1), steps = (len(X_test) / batch_size))
print(scores)

#Uncomment if you need to plot the training accuracy against the number of epochs
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.plot(range(1, epochs+1), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
#print(history.acc)
