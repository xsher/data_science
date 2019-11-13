# OpenCV Face Detect
Disclaimer: the code base used in this is from OpenCV python sample

### Directory Structure
|- README.md explains the implementation, requirement and how to run the code.
|- facedetect.py is the python program file that runs the face detection algorithm
|- common.py is a replica from OpenCV sample as the utility to run facedetect.py
|- video.py is a replica from OpenCV sample as the utility to run facedetect.py

### Face Detection Implementation
This programs performs the frontal face detection using the pretrained model with Haar Cascade Classifier in OpenCV. Haar features which works similarly to convolutional kernel are extracted from the images. Each feature is the subtraction of the sum of pixels in the white region from the sum of pixels from the black region. Integral images are used to improve the performance of the feature generation. Cascade of classifiers are then performed.

To run the face detection program:
```bash
python facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
```
Parameters denoted in `[]` are optional.

Implementation:
1. Capture the image from camera
2. If there are no previously detected frames with frontal face:
    - perform the detection of frontal face with cascade classifier on the whole image (`function detect()`)
3. If there are previously detected frames with frontal face: (in order to speed up the detection)
    - for each frame:
        - scale the image by a scale e.g. 1.5 and perform detection on that frame (`function detect()`)

To speed up the task, the part of the code which performs eye detection is being removed or commented.


Several improvements that can be considered:
- Efficient subwindow search / branch and bound on the whole image
- Better prior knowledge
- Vary the scale size but it is a trade off between the accuracy and the time

### Requirements
- Library: _numpy_
- Latest version of OpenCV or there will be code incompatibilities when running facedetect.py
