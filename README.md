# CSC522 Object Recognition

## Dataset
The dataset is available at [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)

## 1. Executing the Unsupervised Code

## 2. Executing the Supervised Code 
### 2.1 Auto Encoder
### 2.2 t-Distributed Stochastic Neighbor Embedding (tsne)
Please make sure the following packages are installed.
1. pandas
2. numpy
3. matplotlib
4. sklearn
5. tensorflow
6. keras

Also, create a **"data/"** directory in the same directory where the tsne.py and clustering_metrics.py exist. The downloaded dataset has to be extracted in the data directory which will create **"data/256_ObjectCategories"** where the actual dataset exists.

Execute the below command to run tsne on the provided data.

```python tsne.py```
