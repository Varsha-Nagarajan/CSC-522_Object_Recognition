import numpy as np
import keras
from keras.models import Model, load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras import models

## load the model and add this code to tvisualize the model's activation on a given layer
img_path = '256_ObjectCategories/test/056.dog/056_0037.jpg'

img = image.load_img(img_path, target_size=(128, 128)) 
img_arr = image.img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0) 
img_arr /= 255.

#show real_image
plt.imshow(img_arr[0])
plt.show()
print(img_arr.shape)


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])


layer_outputs = [layer.output for layer in model.layers[1:]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

layer_activation = activations[3] # the layer activation that needs to be visualized
print(layer_activation.shape)

plt.matshow(layer_activation[0, :, :, 2], cmap='viridis')
    
