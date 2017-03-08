import cv2
import numpy as np
from math import (pi, sqrt)

window = 'CV2 test rig'

class Param:
  def __init__(self, name, max_value, value):
    self.name = name
    self.max = max_value
    self.value = value

class CV2Test:
  def __init__(self, img, params):
    self.img = img
    self.params = params

    cv2.namedWindow(window)

    for param in params:
      cv2.createTrackbar(param.name, window, param.value, param.max, self.onChange)

    self.show()

  def show(self):
    preview = self.img.copy()

    dst = cv2.cornerHarris(gray,
      self.params[0].value,
      (self.params[1].value * 2) + 1,
      max(self.params[2].value, 1) / 100)
    
    dst = cv2.dilate(dst, None)
    corner = preview.copy()
    corner = cv2.cvtColor(corner, cv2.COLOR_GRAY2BGR)
    corner[dst > 0.01 * dst.max()] = (0, 0, 255)

    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    preview = np.hstack((img, preview, corner))

    # preview = cv2.resize(
    #     preview, 
    #     dsize = None, 
    #     fx = 1.8, 
    #     fy = 1.8, 
    #     interpolation = cv2.INTER_AREA)

    n = 30
    for param in self.params:
      text = ("%s: %s" % (param.name, param.value))
      cv2.putText(preview, text, (10, n), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
      n = n + 30

    cv2.imshow(window, preview)
    cv2.waitKey(0)

  def onChange(self, value):

    for param in self.params:
      param.value = cv2.getTrackbarPos(param.name, window)

    self.show()

img = cv2.imread('./2016-10-08-01-01-01_pilot01_6166_occupancy.pgm', 
                 cv2.CV_LOAD_IMAGE_COLOR)
width, height, channels = img.shape
img = img[100:, :, :]
# img[np.where((img >= (250, 250, 250)).all(axis = 2))] = (0, 0, 0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray[np.where((img >= (250, )).all(axis = 2))] = 0
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

SM_KERNEL_SIZE = 5
SM_KERNEL = np.ones((SM_KERNEL_SIZE, SM_KERNEL_SIZE), np.uint8)

thresh = cv2.dilate(thresh, SM_KERNEL)
thresh = cv2.erode(thresh, SM_KERNEL)
# params = [
#   {'name': 'theta', 'max': 100, 'value': 10},
#   {'name': 'threshold', 'max': 100, 'value': 10},
#   {'name': 'minLineLength', 'max': 100, 'value': 10},
#   {'name': 'maxLineGap', 'max': 100, 'value': 10}
# ]

params = [
  Param('blockSize', 100, 2),
  Param('ksize', 14, 3),
  Param('k', 200, 4)
]

CV2Test(thresh, params)