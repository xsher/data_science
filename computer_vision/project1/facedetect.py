#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# '''
# Several improvements that can be considered:
#   - Use the detected frame from the current frame as a region of interest to
#     detect for the next frame rather than recalculating it all over again
#   - Efficient subwindow search / branch and bound on the whole image
#   - Better prior knowledge
#   - Vary the scale size? but it is a trade off between the accuracy and the time
# '''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str


def detect(imgs, cascade):
    """
    The function `detect` performs the detection of frontal faces with CascadeClassifier

    Args:
        imgs: A list of image objects to perform detection on
        cascade: A cascade classifier object with pre-trained classifier
    Returns:
        A list of rectangles (x1, y1, x2, y2) which is being detected as faces
    """
    detected_rects = []
    # for each image in the list of images, perform a cascade detection of the
    # frontal faces
    for img in imgs:
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                                    minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
        if len(rects) > 0:
            # changes the (x, y , w, h) to (x1, y1, x2, y2)
            rects[:,2:] += rects[:,:2]
        detected_rects += list(rects)

    return detected_rects

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

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(
                        cv.samples.findFile('samples/data/lena.jpg')))

    to_detect = []
    imgScale = 1.5

    while True:
        ret, img = cam.read()
        if len(to_detect) == 0:
            """
            If there is no past detected face i.e the first run, or unable
            to detect the last image
            """
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            to_detect.append(cv.equalizeHist(gray))
        else:
            """
            If there is past detected faces, perform the face detection
            only on the regions around it.
            """
            for idx, r in enumerate(to_detect):
                _img = img[r[1]:r[3], r[0]:r[2]]
                newX = _img.shape[1] * imgScale
                newY = _img.shape[0] * imgScale
                to_detect[idx] = cv.resize(_img, (int(newX), int(newY)))

        t = clock()
        rects = detect(to_detect, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        to_detect = rects

        """
        Eyes detection is not required, relevant cascade initialisations removed
        """
        # if not nested.empty():
        #     for x1, y1, x2, y2 in rects:
        #         roi = gray[y1:y2, x1:x2]
        #         vis_roi = vis[y1:y2, x1:x2]
        #         subrects = detect(roi.copy(), nested)
        #         draw_rects(vis_roi, subrects, (255, 0, 0))

        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()
