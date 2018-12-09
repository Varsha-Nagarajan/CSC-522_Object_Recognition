# CSC522 Object Recognition

## Dataset
The dataset is available at [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)

For supervised classification, we made a train-val-test split 

## 1. Executing the Supervised Code
### 1.1 Plain Classifier
## 2. Executing the Unsupervised Code 
Please make sure the following packages are installed.
1. pandas
2. numpy
3. matplotlib
4. sklearn
5. tensorflow
6. keras
7. time

Also, create a **"data/"** directory in the same directory where the tsne.py and clustering_metrics.py exist. The downloaded dataset has to be extracted in the data directory which will create **"data/256_ObjectCategories"** where the actual dataset exists.

### 2.1 Auto Encoder
run autoencoder_training.py to train the autoencoder model.

run evaluate_model.py to evaluate the entropy and purity of the k-means clusters using the features extracted by the autoencoder.

run visualize_feature_maps.py to create the visualization of the feature maps extracted by the encoder.

run kmeans_baseline.py to perform and evaluate the k-means baseline approach, which trains k-means with raw pixel values.

### 2.2 t-Distributed Stochastic Neighbor Embedding (tsne)

Execute the below command to run tsne on the provided data.

```python tsne.py```
