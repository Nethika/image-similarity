"""
Image Similarity by Reverse image search and retrieval with Keras

Nethika Suraweera
TinMan Kinetics
01/11/2018


Inspired by :https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-search.ipynb


This script uses previously-trained neural network ResNet50 
from Keras to search through a large collection of images. 
Specifically, it will show you how you can retrieve a set 
of images which are similar to a query image, 
returning you its n nearest neighbors in terms of image content.
It removes the last classification layer from the network, 
leaving the last fully-connected layer as the new output layer. 
The way we do this is by instantiating a new model called 
feature_extractor which takes a reference to the desired 
input and output layers in our ResNet50 model. 
Thus, feature_extractor's output is the layer just before the classification, 
the last 2048-neuron fully connected layer.
With ResNet50 model, a distance of 20.0 seems to be a good threshold to filter similar images.
"""

import os
import random
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from keras.applications.resnet50 import ResNet50
from PIL import ExifTags
from PIL import Image


def get_image(path,input_shape):
    """
    get_image will return a handle to the image itself, and a numpy array of its pixels to input the network.
    This function preprocesses the images to be in the correct orientation using exif tags data. 
    """
    img = image.load_img(path)
    exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
    #print (exif)
    if 'Orientation' in exif:
        if exif['Orientation'] == 6:
            img=img.rotate(-90, expand=True)
        if exif['Orientation'] == 8:
            img=img.rotate(90, expand=True)    
        if exif['Orientation'] == 3:
            img=img.rotate(180, expand=True) 
    img=img.resize(input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(query_image_idx, num_results=5):
    """
    returns the indexes and the distances of the similar images for the given queary image.
    """
    distances = [ distance.euclidean(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    dis_closest = sorted(distances)[1:num_results+1]
    return (idx_closest, dis_closest)


#load the model
model = ResNet50(weights='imagenet')  #threshold =20  get_layer("flatten_1")

# input shape
input_shape = model.input_shape[1:3]

#remove the last layer
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("flatten_1").output)  #ResNet50

#load Images
max_num_images = 10000

## from folder
"""
images_path = './dupes'
images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
"""

#from input file
lines = [line.rstrip('\n') for line in open('input.txt')]
q_image = lines[1]
images = lines[3:]
images.insert(0,q_image)


if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]

print("\nKeeping %d images to analyze" % len(images))



#get features
features = []
for image_path in images:
    img, x = get_image(image_path,input_shape)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)


features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)

# do a query on an image
#query_image_idx = int(len(images) * random.random())
query_image_idx = 0
#print("Query image index", query_image_idx)
print("\nQuery image:")
print(images[query_image_idx])
print("")
(idx_closest,dis_closest) = get_closest_images(query_image_idx)
#print("Closest image indexes  :", idx_closest)
#print("Closest image distances:", dis_closest)
print("Closest images:")
for idx,dis in zip(idx_closest,dis_closest):
    print("image:",images[idx])
    print("distance:",dis)
    print("")

print ("Similar images are the ones that the distance is less than 20.0 (threshold)")


