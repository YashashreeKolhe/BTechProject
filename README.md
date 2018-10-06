# BTechProject

#Object Detection using vgg16

- This program directly classifies images into one of the 1000 categories on which vgg16 model is pretrained

How to use: python3 test_images.py

Paste the images you want the labels of, in the folder named 'single_prediction'

#Object Detection using Transfer Learning

- This program allows user to use her own dataset to train the model and then classify test-images into one of the categories for which the model is trained
- It contains fine tuning of VGG 16 Image Classifier with keras by modification of some layers of VGG16 model

How to use: 

To build the model and save it: python3 build_model2.py
To load the saved model and test it: python3 test.py

Dataset:
- The dataset contains 3 folders - training_set, validation_set and test_set
- training_set and validation_set contain images in the appropriately labelled folders
- Heirarchy of Dataset:

        Dataset
             -training_set
                  -car
                    -images of cars
                  -cat
                    -images of cats
                  -dog
                    -images of dogs
                  -flower
                    -images of flowers
                  -fruit
                    -images of fruits
                  -person
                    -images of people
            -validation_set
                   -car
                    -images of cars
                  -cat
                    -images of cats
                  -dog
                    -images of dogs
                  -flower
                    -images of flowers
                  -fruit
                    -images of fruits
                  -person
                    -images of people
             -test_set
                  -images to be classified
                  
    
