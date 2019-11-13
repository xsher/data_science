#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
import time
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str

from letter_recog import Boost, MLP, SVM

from keras.layers import Flatten, Input
from keras.models import Model
from keras.applications.vgg16 import VGG16


def vgg_model():
    _input = Input(shape=(224, 224, 3), name = 'image_input')
    # output dimension for VGG16 is a tensor of 7 x 7 x 512
    vgg_model = VGG16(weights='imagenet', include_top=False)

    our_feats = vgg_model(_input)
    our_feats = Flatten()(our_feats)

    model = Model(inputs=input, outputs=our_feats)
    model.summary() # See our model

    return model


def detect(img, cascade):
    """
    The function `detect` performs the detection of frontal faces with CascadeClassifier

    Args:
        img: An image objects to perform detection on
        cascade: A cascade classifier object with pre-trained classifier
    Returns:
        A list of rectangles (x1, y1, x2, y2) which is being detected as faces
    """
    to_detect = img.copy()
    rects = cascade.detectMultiScale(to_detect, scaleFactor=1.3, minNeighbors=4,
                                minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) > 0:
        # changes the (x, y , w, h) to (x1, y1, x2, y2)
        rects[:,2:] += rects[:,:2]

    return rects

def draw_rects(img, rects, color):
    """
    Takes in the original image and draws rectangles on it

    Args:
        img: the original image for the drawing of rectangles
        rects: the list of rectangle coordinates to perform the drawing
        color: the colour for the rectangle to be drawn
    Returns:
        Nothing
    """
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def show_hist(hist):
    """
    Generating the colour histogram
    """
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in xrange(bin_count):
        h = int(hist[i])
        cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imshow('hist', img)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade=',
        'model=', 'predict='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))

    models = [Boost, SVM, MLP] # NBayes
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )

    # control variable to see if we want to do predictions
    predict = True if args.get('--predict', "T") == "T" else False
    model = None
    # load model only if we want to do prediction
    if predict:
        model_path = args.get('--model', "mlp_model")
        Model = models[model_path.split("/")[-1].split("_")[0]]
        model = Model()
        model.load(model_path)

        vgg_imgnet = vgg_model()

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(
                        cv.samples.findFile('samples/data/lena.jpg')))
    f = open("letter_recog_matrix.txt", "a")
    # to_detect = []
    detected = None
    show_backproj = False


    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # (0, 60, 30) is a dark colour
        # (180. 255, 255) is cyan
        # masking is thresholding the HSV image to get only colours in
        # the range specified
        mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        vis = img.copy()
        if detected is None:
            """
            If there is no past detected face i.e the first run, or unable
            to detect the last image
            """
            rects = detect(gray, cascade)
            if len(rects) > 0:
                detected = rects[0]
            else:
                detected = None
        else:
            """
            If there is past detected faces, perform camshift algorithm
            """
            x0, y0, x1, y1 = detected
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            # initialise the track window i.e. (xmin, ymin, w, h)
            track_window = (x0, y0, x1 - x0, y1 - y0)

            # generating histogram for the blue channel
            # for color range [0, 180] as masked above
            # since we are only doing it for a region, we need to pass mask
            hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )

            # using a min max normalizer
            cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            hist = hist.reshape(-1)
            show_hist(hist)

            prob = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)
            prob &= mask
            term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
            track_box, track_window = cv.CamShift(prob, track_window, term_crit)

            # trackbox is in the format of ((cx, cy), (w, h), ?)
            (cx, cy), (w, h), _ = track_box
            
            tx1 = int(cx - w/2) - 20
            tx2 = int(cx + w/2) + 20

            # applying on entire y due to the coloration of the neck
            prob[:, tx1:tx2] = 0

            # perform the detection for hand on the entire image instead
            # of only on the track window
            entire_window = (0, 0, np.shape(vis)[0], np.shape(vis)[1])
            hand_track_box, _ = cv.CamShift(prob, entire_window, term_crit)
            (hcx, hcy), (hw, hh), _ = hand_track_box

            # prevent it from spilling out of window by doing min max
            hx1 = max(int(hcx - hw/2), 0)
            hy1 = max(int(hcy - hh/2), 0)
            hx2 = min(int(hcx + hw/2), np.shape(vis)[0])
            hy2 = min(int(hcy + hh/2), np.shape(vis)[1])

            if show_backproj:
                vis[:] = prob[...,np.newaxis]

                # only enable capture if there is an actual hand detected
                if hw > 0 and hh > 0:
                    hand_img = vis.copy()
                    hand_img = hand_img[hy1:hy2, hx1:hx2]
                    hand_img_16 = cv.resize(hand_img, (16, 16))
                    hand_img_224 = cv.resize(hand_img, (224, 224))
                    ts = int(time.time())
                    """
                    We do not need to set the image to gray anymore
                    as the image that we train our model on is based on the
                    original image. Here we can just get the VGG16 features
                    for the original image
                    """
                    vgg_feats = vgg_model.predict([[hand_img_224]])
                    if predict:
                        # The model is trained with VGG features,
                        # so we predict with that
                        predicted = model.predict(vgg_feats.flatten()) + ord('A')
                    
                    # save image on key detection
                    if ch == ord('o'):
                        cv.imwrite('O/O_16x16_{}.jpg'.format(ts), hand_img_16)
                        cv.imwrite('O/O_224x224_{}.jpg'.format(ts), hand_img_224)

                    if ch == ord('c'):
                        cv.imwrite('C/C_16x16_{}.jpg'.format(ts), hand_img_16)
                        cv.imwrite('C/C_224x224_{}.jpg'.format(ts), hand_img_224)

                    if ch == ord('l'):
                        """
                        saving out images as pixels of 16x16 in gray colour
                        grayImg = cv.cvtColor(hand_img_224, cv.COLOR_BGR2GRAY)
                        img_arr = [float(x) for x in np.array(grayImg).flatten()]
                        f.write("L,")
                        f.write(",".join(img_arr))
                        f.write("\n")
                        """
                        cv.imwrite('L/L_16x16_{}.jpg'.format(ts), hand_img_16)
                        cv.imwrite('L/L_224x224_{}.jpg'.format(ts), hand_img_224)
            try:
                # draw ellipse over the face
                cv.ellipse(vis, track_box, (0, 0, 255), 2)
                # draw rectangle over the hand
                cv.rectangle(vis, (hx1, hy1),
                    (hx2, hy2),
                    (0, 255, 0), 2)
                if predict:
                    draw_str(vis, (50, 50), 'Letter shown is:  {}'.format(chr(predicted)))
            except:
                print(track_box)

        cv.imshow('facedetect', vis)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('b'):
            show_backproj = not show_backproj


    f.close()
    cv.destroyAllWindows()
