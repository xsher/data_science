# Detecting letters from hand action recognition
The main code is obtained from python samples in OpenCV, camshift.py,  letter_recog.py and facedetect.py

### Directory Structure
|- README.md explains the implementation, requirement and how to run the code.
|- `camshift_handdetect.py` the implementation of the hand action detection with predictions of the letter as described below.
|- `project4_feature_generation.ipynb` the notebook ran in Google Colab to generate the VGG16 features to train the MLP model for the letter recognition.

### Camshift Hand Detect Implementation
The program will first detect the face using the pretrained model with Haar Cascade Classifier in OpenCV. Following the detected face. It will then calculate the color histogram representation of the face and perform Continuously Adaptive Meanshift (Camshift) to find parts of images which corresponds to the similar color (i.e. colour of the face) in the entire image. The original face is first obfuscated to prevent Camshift from returning the original area of the face.

The detected area which should correspond to the hand is then saved according to the hand action that is being performed by manually selecting the key on the keyboard. The images saved are labelled according to it.

The saved images are used to train letter recognition model with the `letter_recog.py`.
Three models are trained in this case: 
- MultiLayer Perceptron (MLP)
- Support Vector Machine (SVM)
- Boost

This iteration is trained with the VGG16 features of the hand images of size 224 by 224.
We utilise the pretrained model on ImageNet in order to generate these features. Details are in the `.ipynb`.
The following are the test accuracies of each model trained on VGG16 features.
_**Test Accuracy for VGG16 based features**_
- SVM : 44.864 %
- MLP: 97.297 %
- Boost: 93.513 %

_**Test Accuracy for pixel based features**_
- SVM : 44.155 %
- MLP: 97.727 %
- Boost: 89.286 %

The model MLP gives the best performance and hence we will be utilizing it for predictions. We can also note that changing the features from using pixels to VGG16 features does not improve the MLP model but it does gives a performance improvement to the Boost model.

To train the models, follow the instructions in the section of `# Running the code`

### Running the code
Make a copy of the python file `camshift_handdetect.py` in OpenCV python samples directory in order to satisfy the dependencies and run the following command to run the detection:
```
python camshift_handdetect.py
```

To train the models:
```
python letter_recog.py --data <path_to_data> --model <type_of_model> --save <save_path>
```
- path_to_data: path to the file characterized by "<letter>,<VGG16 features>" per line
- model: type of model to train for, in this case we experiment with MLP, SVM and Boost.
- save_path: path to output the model so that we can load it for predictions later.

Note that the file letter_recog.py should be in the opencv directory with the other files required. 

To predict:
```
python camshift_handdetect.py --model <path_to_model> --predict T
```
- path_to_model: the path to the trained model, note that the naming for the saved model should be prefixed by mlp, svm or boost for it to pick up the model type for predictions.

Note that in order to be able to detect the letters, turn on the back project mode by utilising the key 'b'. The predicted letters will be shown on the screen.

To generate the VGG features:
Run the `project4_feature_generation.ipynb` and the change the path to the letter images accordingly. The features will be saved to a file which can then be used for the training of model signified by variable `<path_to_data>`.


##### Final Note:
The procedures and processing steps are commented within the `camshift_handdetect.py` file.
