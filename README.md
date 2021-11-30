# ITU-AI-ML-PS-009-RF-Sensor-based-HAR

This repository contains the code and description of our solution as well as the report and the final presentation slides for ITU AI/ML Challenge PS-009 problem statement.

test_individualSensor.py:

Assumptions:
1) Spectrogram images are provided for classification.
2) The type of sensor is known apriori (77GHz/ 24GHz/ Xethru)

Testing:
1) Provide the appropriate information to the variable sensor in line number 29 (needs to be chosen among 77GHz/24GHz/Xethru).
2) In line number 31 declare whether a single image or a folder containing images needs to be tested for (0 for single image and 1 folder containing images).
3) Provide the path to image / folder containing images in line number 33.

Output:
The predicted class integer value obtained in variable y_pred is displayed(printed out) for the image/ images in the folder.

The class integer value can be mapped to the activity as given below.

Prediction	Activity
0		Walking towards
1		Walking away
2		Picking
3		Bending
4		Sitting
5		Kneeling
6		Crawling
7		Walking both toes
8		Limping
9		Short Step
10		Scissor gait



I_Q_to_image.py:
The above file contains codes for converting I/Q samples to spectogram images. The above code has been taken from https://github.com/ci4r/CI4R-Activity-Recognition-datasets. 

train_individualSensor.py

Assumptions:
1) We have 3 different folders corresponding to each sensor in the following format:
	1. Folder 1: Containing 2 subfolders with all the images of classes 2 and 3.
	2. Folder 2: Containing 3 subfolders with all the images of classes 8,9 and 10.
	3. Folder 3: Containing 8 subfolders in which one subfolder contains all images of both classes 2 and 3 and one subfolder contains all images of classes 8,9 and 10.

Training:
1)Provide the appropiate path to the folder on line number 31.
2)Provide the appropriate information to the variable classes on line number 32.(needs to be chosen among 2/3/8).
3)Based on the value of classes variable, the number of filter size is determined.
4)Provide appropriate name to the checkpoint_path variable on line number 99.

Output:
The model weights are stored in the filename provided by checkpoint_path variable.

