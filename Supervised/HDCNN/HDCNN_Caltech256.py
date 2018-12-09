import keras
import numpy as np
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
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.cluster import AffinityPropagation
from keras.layers import Input
from keras.models import Model
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import shutil

img_size = 128
batch_size=24

train_datagen = ImageDataGenerator(rescale=1./255, zca_whitening = False, featurewise_center=False)
validate_datagen = ImageDataGenerator(rescale=1./255, zca_whitening =False, featurewise_center=False)
test_datagen = ImageDataGenerator(rescale=1./255, zca_whitening =False, featurewise_center=False)


train_generator = train_datagen.flow_from_directory(
    directory="train/",
    target_size=(img_size, img_size),
    batch_size = batch_size,
    class_mode='categorical', seed = 42)

validation_generator = validate_datagen.flow_from_directory(
        'val/',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical', seed = 42)

test_generator = test_datagen.flow_from_directory(
        'test/',
        target_size=(img_size, img_size),
        batch_size=1,class_mode='categorical',seed = 42)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

print(validation_generator.classes)

STEP_SIZE_VALID=validation_generator.n//batch_size
STEP_SIZE_TEST=test_generator.n//batch_size

#SimpleClassifier model to generate similarity matrix for spectral clustering
K.set_image_data_format('channels_last')
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same", strides=(1,1), input_shape=(img_size,img_size,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(1,1), padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation = 'relu'))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation = 'relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(256, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.summary()

#loading pre-trained model weights. Fit the model when running for the first time.
modeldir="model/"
model.load_weights(modeldir+"hdcnn.h5")

#Making predictions. These will be used to generate the similarity matrix
values = model.predict_generator(generator=test_generator, steps=STEP_SIZE_TEST)
num_classes = values.shape[1]

#predicted labels
y_pred_on_val_data = [np.argmax(x) for x in values]
#groundtruth labels
y_val = test_generator.classes
#putting them into a dataframe
df = pd.DataFrame([y_val, y_pred_on_val_data],dtype=int)
df = df.transpose()
df.columns =['t','p']

#generating confusion matrix
conf_mat = np.zeros((num_classes,num_classes))
for i in range(df.shape[0]):
    t = df.loc[i]['t']
    p = df.loc[i]['p']
    conf_mat[t][p] = conf_mat[t][p] +1

#Saving space by freeing up dataframe as it is no longer needed
df = 0

for i in range(num_classes):
    conf_mat[i] = conf_mat[i]/sum(conf_mat[i])

#print(conf_mat)
#plt.imshow(conf_mat, cmap=plt.cm.Blues)
#plt.colorbar()

dist_mat = 1 - conf_mat
#set diagonal elements to 0
dist_mat[range(num_classes),range(num_classes)] = 0
dist_mat = 0.5 * (dist_mat + dist_mat.T)
#plt.figure()
#plt.title('distance matrix on validation set')
#plt.imshow(dist_mat, cmap= plt.cm.Purples_r)
#plt.colorbar()

#Laplacian eigenmap dimensionality reduction construct adjacency graph W (symmetric) using k-NN
W=np.zeros((num_classes,num_classes))

# t is used for generating weight matrix for spectral clustering. knn, t, and dim are hyperparameters
# Todo: can be tuned further with different k values.
k_nn, t, dim = 3, 0.9, 4

for i in range(num_classes):
    idx=np.argsort(dist_mat[i,:])[1:k_nn+1]
    W[i,idx]=np.exp(-dist_mat[i,idx] / t)
    W[idx,i]=W[i,idx]
D=np.zeros(W.shape)
for i in range(num_classes):
    D[i,i]=np.sum(W[i,:])

L=D-W

eigen_values, eigen_vectors = scipy.linalg.eig(L,D)

ftr=eigen_vectors[:,1:dim+1]
L = 0
D= 0
W = 0
dist_mat = 0
conf_mat = 0

#Affinity Propagation clustering
affinity_propagation_cluster = AffinityPropagation(damping=0.75, max_iter=15000, convergence_iter=50, copy=True)
cluster_labels = affinity_propagation_cluster.fit_predict(ftr)
unique_cluster_label = np.unique(cluster_labels)
n_cluster = unique_cluster_label.shape[0]
cluster_members=[None]*n_cluster
print ('%d clusters' % n_cluster)

label_names=range(num_classes)
for i in range(n_cluster):
    idx = np.nonzero(cluster_labels == unique_cluster_label[i])[0]
    cluster_members[i]=list(idx)
    print ('cluster %d size %d ' % (i, len(idx)))
    for j in range(len(idx)):
        print ('%s,' % label_names[idx[j]],)
    print (' ')
print(cluster_members)

#=====================================================================================================================#
# Uncomment if you want overlapping clusters in coarse categories. That is, each fine category can belong to multiple coarse category
# train_val_img_labels = validation_generator.classes
# exp_cluster_members=[None]*n_cluster
# if 1:
    # all_mb=range(num_classes)
    # # for 5 clusters v0.0
    # gamma=2.0
    # score_thres=1.0/(gamma*n_cluster)
    
    
    # max_exp_clu_size=80
    # extra_cluster_members=[None]*n_cluster

    # for i in range(n_cluster):
        # non_member = np.asarray(np.setdiff1d(range(num_classes),cluster_members[i]))
    # #     print non_member.shape
        # score=np.zeros((non_member.shape[0]))
        # for j in range(non_member.shape[0]):
            # idx=np.nonzero(train_val_img_labels==non_member[j])[0]
            # lc_prob=values[idx,:][:,cluster_members[i]]
            # score[j]=np.mean(np.sum(lc_prob,axis=1))
        # score_sorted=np.sort(score)[::-1]
        # idx_sort=np.argsort(score)[::-1]
        # idx2=np.nonzero(score_sorted>=score_thres)[0]
        # if len(idx2)+len(cluster_members[i])> max_exp_clu_size:
            # idx2=idx2[:(max_exp_clu_size-len(cluster_members[i]))]
        # extra_cluster_members[i]=[non_member[idx_sort[id]] for id in idx2]
        # exp_cluster_members[i]=cluster_members[i]+extra_cluster_members[i]
        # #assert len(exp_cluster_members[i])==np.unique(np.asarray(exp_cluster_members[i])).shape[0]
# else:
    # '''disjoint coarse category'''
    # for i in range(n_cluster):
        # exp_cluster_members[i]=cluster_members[i]
        

# f2cmap1=[None]*num_classes
# for i in range(num_classes):
    # f2cmap1[i]=[]
# for i in range(len(exp_cluster_members)):
    # for j in range(len(exp_cluster_members[i])):
        # f2cmap1[exp_cluster_members[i][j]] += [i]

# for i in range(num_classes):
	# print(f2cmap1[i])
#====================================================================================================================#

f2cmap = {}
for coarse in range(len(cluster_members)):
    for fine in cluster_members[coarse]:
        f2cmap[fine] = coarse

# The number of coarse categories
coarse_categories = n_cluster
# The number of fine categories
fine_categories = num_classes
#print(coarse_categories)
#print(fine_categories)


#generating directory structure for coarse categories
#we will have all fine categories as directories within each coarse category (most will be empty)
print("Partitioning in directories")
modeldir="model"
traindict = train_generator.class_indices
traindest = modeldir+"/ctrain"
trainsrc = "train/"
valdict = validation_generator.class_indices
valdest = modeldir+"/cvalidation"
valsrc = "val/"

testdict = test_generator.class_indices
testdest = modeldir+"/ctest"
testsrc = "test/"

#===============================================================================================================================#
# Uncomment if you want overlapping clusters in coarse categories. That is, each fine category can belong to multiple coarse category
# This will create directory structure based on new fine -> coarse mapping
# for c in range(coarse_categories):
    # if not os.path.exists(traindest+"/"+str(c)):
        # os.makedirs(traindest+"/"+str(c))
    # if not os.path.exists(valdest+"/"+str(c)):
        # os.makedirs(valdest+"/"+str(c))

# for key in traindict:
    # val = traindict[key]
    # c1 = f2cmap1[val]
    # for c in c1:
    	# print("course " + str(c) + " key " + str(key) + " "+ str(traindest))
    	# shutil.copytree(trainsrc+key, traindest+"/"+ str(c)+"/"+key)


# for key in valdict:
    # val = valdict[key]
    # c1 = f2cmap1[val]
    # for c in c1:
    	# print("course " + str(c) + " key " + str(key) + " "+ str(valdest))
    	# shutil.copytree(valsrc+key, valdest+"/"+ str(c)+"/"+key)
	
# for key in testdict:
    # val = testdict[key]
    # c1 = f2cmap1[val]
    # for c in c1:
    	# print("course " + str(c) + " key " + str(key) + " "+ str(testdest))
    	# shutil.copytree(testsrc+key, testdest+"/"+ str(c)+"/"+key)

# for c in range(coarse_categories):
    # for key in traindict:
        # if not os.path.exists(traindest+"/"+str(c)+"/"+key):
            # os.makedirs(traindest+"/"+str(c)+"/"+key)
        # if not os.path.exists(valdest+"/"+str(c)+"/"+key):
            # os.makedirs(valdest+"/"+str(c)+"/"+key)
        # if not os.path.exists(testdest+"/"+str(c)+"/"+key):
            # os.makedirs(testdest+"/"+str(c)+"/"+key)
#===============================================================================================================================#
for c in range(coarse_categories):
    if not os.path.exists(traindest+"/"+str(c)):
        os.makedirs(traindest+"/"+str(c))
    if not os.path.exists(valdest+"/"+str(c)):
        os.makedirs(valdest+"/"+str(c))

for key in traindict:
    val = traindict[key]
    c = f2cmap[val]
    print("course " + str(c) + " key " + str(key) + " "+ str(traindest))
    shutil.copytree(trainsrc+key, traindest+"/"+ str(c)+"/"+key)


for key in valdict:
    val = valdict[key]
    c = f2cmap[val]
    print("course " + str(c) + " key " + str(key) + " "+ str(valdest))
    shutil.copytree(valsrc+key, valdest+"/"+ str(c)+"/"+key)
	
for key in testdict:
    val = testdict[key]
    c = f2cmap[val]
    print("course " + str(c) + " key " + str(key) + " "+ str(testdest))
    shutil.copytree(testsrc+key, testdest+"/"+ str(c)+"/"+key)

for c in range(coarse_categories):
    for key in traindict:
        if not os.path.exists(traindest+"/"+str(c)+"/"+key):
            os.makedirs(traindest+"/"+str(c)+"/"+key)
        if not os.path.exists(valdest+"/"+str(c)+"/"+key):
            os.makedirs(valdest+"/"+str(c)+"/"+key)
        if not os.path.exists(testdest+"/"+str(c)+"/"+key):
            os.makedirs(testdest+"/"+str(c)+"/"+key)
                                            

#single classifier training (shared)
input_shape = (32,32,3)
in_layer = Input(shape=input_shape, dtype='float32', name='main_input')

net = Conv2D(384, 3, strides=1, padding='same', activation='elu')(in_layer)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(384, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(384, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(640, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(640, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.2)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(640, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.3)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(768, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(896, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(896, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.4)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(896, 3, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1024, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1024, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.5)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.6)(net)
net = MaxPooling2D((2, 2), padding='same')(net)

net = Flatten()(net)
net = Dense(1152, activation='elu')(net)
net = Dense(fine_categories, activation='softmax')(net)

model = Model(inputs=in_layer,outputs=net)
sgd_coarse = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer= sgd_coarse, loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255, zca_whitening = False, featurewise_center=False)
validate_datagen = ImageDataGenerator(rescale=1./255, zca_whitening =False, featurewise_center=False)
test_datagen = ImageDataGenerator(rescale=1./255, zca_whitening =False, featurewise_center=False)


batch_size=24
img_size = 32

train_generator = train_datagen.flow_from_directory(
    directory="train/",
    target_size=(img_size, img_size),
    batch_size = batch_size,
    class_mode='categorical', seed = 42)

validation_generator = validate_datagen.flow_from_directory(
        'val/',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical', seed = 42)

test_generator = test_datagen.flow_from_directory(
        'test/',
        target_size=(img_size, img_size),
        batch_size=1, seed = 42)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(train_generator,
          epochs=30,steps_per_epoch = STEP_SIZE_TRAIN+1,
          verbose=1,
          validation_data=validation_generator,validation_steps= STEP_SIZE_VALID+1
          )
model.save_weights('model/model_coarse.h5')
#model.load_weights('model/model_coarse.h5')

sgd_fine = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

for i in range(len(model.layers)):
    model.layers[i].trainable=False

#fine-tuning for coarse classifier
net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(model.layers[-8].output)
net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.6)(net)
net = MaxPooling2D((2, 2), padding='same')(net)

net = Flatten()(net)
net = Dense(1152, activation='elu')(net)
out_coarse = Dense(coarse_categories, activation='softmax')(net)

model_c = Model(inputs=in_layer,outputs=out_coarse)
model_c.compile(optimizer= sgd_coarse, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

for i in range(len(model_c.layers)-1):
    model_c.layers[i].set_weights(model.layers[i].get_weights())

ctrain_generator = train_datagen.flow_from_directory(
        'model/ctrain',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')
cvalidation_generator = validate_datagen.flow_from_directory(
        'model/cvalidation',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')
ctest_generator = test_datagen.flow_from_directory(
        'model/ctest',
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')
		
		
STEP_SIZE_TRAIN_C=ctrain_generator.n//ctrain_generator.batch_size
STEP_SIZE_VALID_C=cvalidation_generator.n//cvalidation_generator.batch_size
STEP_SIZE_TEST_C=ctest_generator.n//ctest_generator.batch_size

model_c.fit_generator(ctrain_generator, steps_per_epoch  = STEP_SIZE_TRAIN_C+1,
          epochs=20,verbose=1,
          validation_data = cvalidation_generator, validation_steps =STEP_SIZE_VALID_C+1,
          )
model_c.save_weights('model/model_c_sgdcoarseoptimizer.h5')
#model_c.load_weights('model/model_c_sgdcoarseoptimizer.h5')
model_c.compile(optimizer=sgd_fine, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
model_c.fit_generator(ctrain_generator, steps_per_epoch  = STEP_SIZE_TRAIN_C+1,
        epochs=20,verbose=1, validation_steps =STEP_SIZE_VALID_C+1,
          validation_data = cvalidation_generator
         )

model_c.save_weights('model/model_c_sgdfineoptimizer.h5')
#model_c.load_weights('model/model_c_sgdfineoptimizer.h5')
coarse_predictions=model_c.predict_generator(ctest_generator,steps=STEP_SIZE_TEST_C+1)
final_coarse_predictions=model_c.evaluate_generator(ctest_generator,steps=STEP_SIZE_TEST_C+1)
print("Accuracy for course: "+ str(final_coarse_predictions))

#constructing fine classifiers
def fine_model():
    net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(model.layers[-8].output)
    net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
    net = Dropout(.6)(net)
    net = MaxPooling2D((2, 2), padding='same')(net)

    net = Flatten()(net)
    net = Dense(1152, activation='elu')(net)
    out_fine = Dense(fine_categories, activation='softmax')(net)
    model_fine = Model(inputs=in_layer,outputs=out_fine)
    model_fine.compile(optimizer= sgd_coarse,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    for i in range(len(model_fine.layers)-1):
        model_fine.layers[i].set_weights(model.layers[i].get_weights())
    return model_fine
	
fine_models = {'models' : [{} for i in range(coarse_categories)], 'yhf' : [{} for i in range(coarse_categories)]}
for i in range(coarse_categories):
    model_i = fine_model()
    fine_models['models'][i] = model_i
	
def get_error(t,p):
    #TODO add confidence score
    return accuracy_score(t,p)
	
modeldir="model"
traindict = train_generator.class_indices
traindest = modeldir+"/ctrain"
trainsrc = "train/"
valdict = validation_generator.class_indices
valdest = modeldir+"/cvalidation"
valsrc = "val/"

testdict = test_generator.class_indices
testdest = modeldir+"/ctest"
testsrc = "test/"

for c in range(coarse_categories):
    for key in traindict:
        if not os.path.exists(traindest+"/"+str(c)+"/"+key):
            os.makedirs(traindest+"/"+str(c)+"/"+key)
        if not os.path.exists(valdest+"/"+str(c)+"/"+key):
            os.makedirs(valdest+"/"+str(c)+"/"+key)

#creating generators for fine classifiers
traingenlist = []
valgenlist = []
train_epochs = []
val_epochs =[]
for i in range(coarse_categories):
    tgen = train_datagen.flow_from_directory(
            'model/ctrain/'+str(i),
            target_size=(img_size,img_size),
            batch_size=batch_size,
            class_mode='categorical')
    vgen = validate_datagen.flow_from_directory(
            'model/cvalidation/'+str(i),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical')
    traingenlist.append(tgen)
    valgenlist.append(vgen)
    train_epochs.append(tgen.n//tgen.batch_size)
    val_epochs.append(vgen.n//vgen.batch_size)
	
#training fine classifiers on corresponding data
for cat in range(coarse_categories):
    print("Start For coarse category : "+ str(cat))
    index= 0
    step = 5
    stop = 15
    while index < stop:
        fine_models['models'][cat].fit_generator(traingenlist[cat], steps_per_epoch = train_epochs[cat]+1,
          epochs=index+step,initial_epoch=index,
          verbose=1,
          validation_data = valgenlist[cat], validation_steps = val_epochs[cat]+1
          )
        index += step
    
    fine_models['models'][cat].compile(optimizer=sgd_fine, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
    stop = 30

    while index < stop:
        fine_models['models'][cat].fit_generator(traingenlist[cat], steps_per_epoch = train_epochs[cat]+1,
          epochs=index+step,verbose=1,initial_epoch=index,
          validation_data = valgenlist[cat], validation_steps = val_epochs[cat]+1
          )
	index += step
    fine_models['models'][cat].save_weights('model/finemodel'+str(cat)+'.h5')
    traingenlist[cat] = 0 
    valgenlist[cat] = 0
    print("End For coarse category : "+ str(cat))


fine_predictions = [] #dim:  n_classes_coarse X n_images_predict X n_classes_fine
for c in range(coarse_categories):
    score_fine = fine_models['models'][c].predict_generator(test_generator,steps=STEP_SIZE_TEST+1)
	score_fine1 = fine_models['models'][c].evaluate_generator(test_generator,steps=STEP_SIZE_TEST+1)
    print("Fine prediction for "+ str(c)+" coarse category: "+ str(score_fine1))
    fine_predictions.append(score_fine)

prediction_size = len(coarse_predictions)
predictions = []
for img in range(prediction_size):
    proba = [0]*fine_categories
    for finec in range(fine_categories):
        for coarsec in range(coarse_categories):
            proba[finec] += coarse_predictions[img][coarsec]*fine_predictions[coarsec][img][finec]
    predicted = np.argmax(proba)
    predictions.append(predicted)

print("Final predictions : "+ str(predictions))	
print("Accuracy for course: "+ str(final_coarse_predictions))

truelabels = test_generator.classes
print(get_error(truelabels,predictions))
	

