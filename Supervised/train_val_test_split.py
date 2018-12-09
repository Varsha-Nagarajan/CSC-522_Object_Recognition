import os
import numpy as np
from shutil import copyfile


def main():
    data_path = 'data/256_ObjectCategories/'
    train_path = 'data/train/'
    val_path = 'data/val/'
    test_path = 'data/test/'
    os.chdir(data_path)
    folders = os.listdir()

    # First create the folders for the train, val and test data
    for folder in range(len(folders)):
        try:
            os.makedirs(train_path + folders[folder] + '/')
            print('Directory ' + train_path + folders[folder] + '/ Created')
        except FileExistsError:
            print('Directory ' + train_path + folders[folder] + '/ Already exists')
        try:
            os.makedirs(val_path + folders[folder] + '/')
            print('Directory ' + val_path + folders[folder] + '/ Created')
        except FileExistsError:
            print('Directory ' + val_path + folders[folder] + '/ Already exists')
        try:
            os.makedirs(test_path + folders[folder] + '/')
            print('Directory ' + test_path + folders[folder] + '/ Created')
        except FileExistsError:
            print('Directory ' + test_path + folders[folder] + '/ Already exists')

    for folder in range(len(folders)):
        folder_path = data_path + str(folders[folder]) + str('/')
        os.chdir(folder_path)
        images_in_folder = os.listdir()
        # Only copy images (there was an empty folder in the dog directory named 'greg'?)
        images_in_folder = [image for image in images_in_folder if image[-4:] == '.jpg']

        # Generate indices for train, validation, and test data
        indices = np.random.choice([1, 2, 3], len(images_in_folder),
                                   replace=True, p=[0.6, 0.15, 0.25])

        # Do split
        train_images = np.array(images_in_folder)[indices == 1]
        val_images = np.array(images_in_folder)[indices == 2]
        test_images = np.array(images_in_folder)[indices == 3]

        # Copy files to the folder corresponding to their split
        for train_image in train_images:
            print('from: %s\nto: %s' % (folder_path + train_image, train_path + folders[folder] + '/' + train_image))
            print()
            copyfile(folder_path + train_image, train_path + folders[folder] + '/' + train_image)

        for val_image in val_images:
            print('from: %s\nto: %s' % (folder_path + val_image, val_path + folders[folder] + '/' + val_image))
            print()
            copyfile(folder_path + val_image, val_path + folders[folder] + '/' + val_image)

        for test_image in test_images:
            print('from: %s\nto: %s' % (folder_path + test_image, test_path + folders[folder] + '/' + test_image))
            print()
            copyfile(folder_path + test_image, test_path + folders[folder] + '/' + test_image)

        print()
    print()


if __name__ == '__main__':
    main()
