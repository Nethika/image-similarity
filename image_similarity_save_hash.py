"""
Image Similarity by Reverse image search and retrieval with Keras
A new image uploaded is only to be comapred with a number of "n_hashes" recently uploaded images.

Nethika Suraweera
TinMan Kinetics
02/05/2018

New Addition:
comapring with a number of "n_hashes" recently uploaded images
==============================================================

`image_hashes.json` file save the hashes calculated for each uploaded image. 
If the new image is similar to a previously uploaded image, a mean hash will be calculated and 
`image_hashes.json` will be updated accordingly.
A new image uploaded is will only be comapred with a number of "n_hashes" recently uploaded images.

ResNet50 Model: 
===============

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
import sys
import glob
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
import json
import string


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


# Image Location
new_images_path = "dupes"

# Read new Images
new_images = glob.glob(os.path.join(new_images_path, "*.jpg"))

# write/read to/from json
json_file= 'image_hashes.json'

# number of hashes to save
n_hashes = 5

# Match with new image
## In a loop:
#for new_image_path in new_images:
   ######################## 

## One at a time:
#new_image_path=os.path.join(user, 'maroon_bells.jpg')
new_image_path=new_images[0]

print("Processing file: {}".format(new_image_path))

img, x = get_image(new_image_path,input_shape)
feat = feat_extractor.predict(x)[0]

# Read from json file
if os.path.exists(json_file):
    image_data_all = json.load(open(json_file))
else:
    image_data_all = []

# Threshold set to identify different images
threshold = 20.0

#Comapre the image with the images in json file

match_dict = {}


if  len(image_data_all) > n_hashes:
    image_data = image_data_all[-n_hashes:]
else:
    image_data = image_data_all

for j in range(len(image_data)):
    image_id = image_data[j]['image_id']
    image_freq = image_data[j]['frequency']
    image_hash = json.loads(image_data[j]['hash'])
    dist = distance.euclidean(feat,image_hash)
    print(image_id , image_freq, dist)
    if dist < threshold:
        match_dict[j] = dist
if match_dict:
    indx = min(match_dict, key=match_dict.get)
    min_hash = json.loads(image_data[indx]['hash'])
    # find New Mean for hash
    new_mean = np.mean([feat,min_hash],axis=0)
    #update hash
    image_data[indx]['hash'] = str(new_mean.tolist())
    #update frequency
    image_data[indx]['frequency'] += 1  
    print("Matched with:")
    print(image_data[indx]['image_id'])
else:     #new image
    print("No match! -> New Image:")
    image_id = str(len(image_data_all)+1).zfill(4)
    print(image_id)
    tempt_dict={'image_id': image_id, 'frequency': 1,'hash':str(feat.tolist())}
    image_data_all.append(tempt_dict)
            
#update json file
with open(json_file, 'w') as imagefile:
    json.dump(image_data_all, imagefile)


####################################################################################

