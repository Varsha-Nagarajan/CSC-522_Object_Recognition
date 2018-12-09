from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

n_images = 29780

# Plots a random subset of feature maps
def plot_feature_maps(encoded_imgs):

    fig, ax = plt.subplots(2, 10)

    fig.subplots_adjust(left=0.12, bottom=0.3, right=0.9, top=.55, wspace=0.1, hspace=0.01)

    # Choose 20 random images to display a feature map from
    idx = np.random.choice(n_images, 20)
    images = encoded_imgs[idx]

    for i in range(10):
        # Choose random feature map from each image
        map_idx = np.random.choice(images.shape[-1], 1)[0]

        # Plot feature map
        ax[0][i].imshow(images[i, :, :, map_idx])
        ax[0][i].axis('off')

    for i in np.arange(10) + 10:
        # Choose random feature map from each image
        map_idx = np.random.choice(images.shape[-1], 1)[0]

        # Plot feature map
        ax[1][i-10].imshow(images[i, :, :, map_idx])
        ax[1][i - 10].axis('off')


def main():
    model_name = 'encoder_caltech256.h5'

    # Load in pre-trained model
    encoder = load_model('encoder_' + model_name)

    encode_datagen = ImageDataGenerator(rescale=1. / 255)
    predict_generator = encode_datagen.flow_from_directory(
        'data/256_ObjectCategories',
        target_size=(128, 128),
        batch_size=1,
        class_mode='input', shuffle=False)

    # Encode all images in data set
    encoded_imgs = encoder.predict_generator(predict_generator, n_images, verbose=1)

    # Visualize random subset of 20 feature maps
    plot_feature_maps(encoded_imgs)






if __name__ == '__main__':
    main()