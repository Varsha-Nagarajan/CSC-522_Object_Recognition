from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator



input_img = Input(shape=(128, 128, 3))

x = Conv2D(100, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(250, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(400, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(600, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(600, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(400, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(250, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(100, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create models with above architecture
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

# Generator to read in and rescale the data
datagen = ImageDataGenerator(
        rescale=1./255)

generator = datagen.flow_from_directory(
        'data/256_ObjectCategories',
        target_size=(128, 128),  # all images will be resized to 128 x 128
        batch_size=16,
        class_mode='input')  # predicting the input

batch_size = 128
n_images = 29780

autoencoder.fit_generator(generator,
                steps_per_epoch=n_images / batch_size,
                epochs=300,
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# Save trained models
autoencoder.save_weights('auto_encoder_weights_caltech256.h5')
autoencoder.save('auto_encoder_caltech256.h5')
encoder.save_weights('encoder_weights_caltech256.h5')
encoder.save('encoder_caltech256.h5')
