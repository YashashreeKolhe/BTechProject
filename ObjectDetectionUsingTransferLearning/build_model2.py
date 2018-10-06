import numpy as np
import keras
import os
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.optimizers import Adam

train_path = 'Dataset/training_set'
valid_path = 'Dataset/validation_set'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224, 224), classes = ['car', 'cat', 'dog', 'flower', 'fruit', 'person'], batch_size = 10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224, 224), classes = ['car', 'cat', 'dog', 'flower', 'fruit', 'person'], batch_size = 10)

vgg16_model = keras.applications.vgg16.VGG16()
model = Sequential()
for layer in vgg16_model.layers[0: len(vgg16_model.layers) - 1]:
	model.add(layer)

for layer in model.layers:
	layer.trainable = False      

model.add(Dense(6, activation = 'softmax'))
                                      
model.compile(Adam(lr = 0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch = 30, validation_data = valid_batches, epochs=10, verbose = 2)

target_dir = './model/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./model/image_classifier_model.h5')
model.save_weights('./model/image_classifier_model_weights.h5')
