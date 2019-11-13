# Camshift for hand action recognition
The main code is obtained from python samples in OpenCV, camshift.py and facedetect.py

### Directory Structure
```
|- README.md explains the implementation, requirement and how to run the code.
|- `camshift_handdetect.py` the implementation of the hand action detection as described below.
```

### Camshift Hand Detect Implementation
The program will first detect the face using the pretrained model with Haar Cascade Classifier in OpenCV. Following the detected face. It will then calculate the color histogram representation of the face and perform Continuously Adaptive Meanshift (Camshift) to find parts of images which corresponds to the similar color (i.e. colour of the face) in the entire image. The original face is first obfuscated to prevent Camshift from returning the original area of the face.

The detected area which should correspond to the hand is then saved according to the hand action that is being performed by manually selecting the key on the keyboard. The images saved are labelled according to it and this will serve as the initial preparation stage for the next step of performing machine learning on it in order to determine the label.


### Running the code
Make a copy of the python file `camshift_handdetect.py` in OpenCV python samples directory in order to satisfy the dependencies and run the following command:
```
python camshift_handdetect.py
```
