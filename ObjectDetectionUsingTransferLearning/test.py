import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from os import walk
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

model_path = './model/image_classifier_model.h5'
model_weights_path = './model/image_classifier_model_weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

test_path = 'Dataset/test_set'
training_path = 'Dataset/training_set'

labels = list()

for (dirpath, dirnames, filenames) in walk(training_path):
	for dirname in dirnames:
		labels.append(dirname)

labels = sorted(labels)
print(labels)
picture_array = list()

for (dirpath, dirnames, filenames) in walk(test_path):
    for filename in filenames:
    	picture_array.append(test_path + "/" + filename)
    break

for picture in picture_array:
	image = load_img(picture, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	yhat = model.predict_classes(image)
	print(picture + " : " + labels[yhat[0]])
