# CSC522 Object Recognition

## Dataset
The dataset is available at [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)

For supervised classification, we made a train-val-test split to generate 17803 train, 4665 validation and 7312 test images. The file named train_val_test_split.py contains code to do the split. Please ensure that the path to the image folders are correct to ensure the split goes through fine.

Packages needed:
1. Keras (use TensorFlow backend)
2. TensorFlow
3. sklearn
4. pandas
5. matplotlib
6. numpy
7. scipy
8. time

## 1. Executing the Supervised Code
For all files in supervised scetion, the code assumes the training images to be present in /train, test images in /test and validation images in /validation folders. Please ensure the corresponsing folders are present and this file is in the same directory as those folders or modify the file to point to the correct folders.

For HD-CNN, we create directory structure for all coarse and fine categories, so ensure that the target directory has necessary permissions.

### 1.1 Plain Classifier 
PlainClassifier.py contains the code for this model as mentioned in the report.
PlainClassifier_Cifar.py contains code for this model for evaluating them in CIFAR-10/100 datasets. Ensure that the correct dataset is loaded and num_classes reflects the correct number of images categories while working with these datasets. 

To run, type: ```python <filename>```

Example: ```python PlainClassifier.py```
  
The files have elaborate comments that will help you with your experiments. Please reach out to the authors if you still face any issues.

### 1.2 ResNet50

We included files for both finetuned and new models as proposed in the report. Since the way we work with images is different in case of Caltech256, we have seperate files for Caltech256 and CIFAR-10/100.

Ensure that the directory path to the training, validation and test data is updated to reflect the correct path in your system. The codes have elaborate comments to help you modify it to suit your purposes. Kindly go through the comments before making any modifications.

To run, type: ```python <filename>```
  
### 1.3 HD-CNN

Insutructions for running this file is same as above. Starting off with correct paths to image folder would be the first step. Ensure that you save your model weights in your first run and load those weights in subsequent runs as these models have a huge training time often spanning to days. Code for all this in already in place, so it just needs to be commented/uncommented as needed.

There is a section where we implement overlapping coarse categories as discussed in the report. The code that does that is currently commented out (due to computational limitations). Please uncomment it for better results. Should you choose to include overlapping categories, uncomment the code that generates the new training, testing and validation directories for fine categories within each coarse category folder. 

To run, type: ```python <filename>```
  
Feel free to experiment with the hyper-parameters for spectral clustering (number of clusters, dimensions, value of t), the threshold and cost for the maximum number of fine categories within a cluster and the models (mini batch size, optimizers - decay, momentum, learning rate). 

## 2. Executing the Unsupervised Code 
Create a **"data/"** directory in the same directory where the tsne.py and clustering_metrics.py exist. The downloaded dataset has to be extracted in the data directory which will create **"data/256_ObjectCategories"** where the actual dataset exists.
### 2.1 Convolutional Autoencoder
To Train the autoencoder model: ```python autoencoder_training.py```

To evaluate the entropy and purity of the k-means clusters using the features extracted by the autoencoder: ```python evaluate_model.py```

To create the visualization of the feature maps extracted by the encoder: ```python visualize_feature_maps.py ```

To perform and evaluate the k-means baseline approach(which trains k-means with raw pixel values): ```python kmeans_baseline.py```
### 2.2 t-Distributed Stochastic Neighbor Embedding (tsne)
To run tsne on the provided data: ```python tsne.py```
