import cv2
import numpy as np

def get(img, dilate_size = 5, erode_size = 5):
  """Isolate the exterior of an image

  Args:
    img (array_like): ndarray (height, width, channels)

  Returns:
    (array_like): ndarray (height, width, 1) cv2.COLOR_GRAY
  """
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # TODO: This can probably be done with a slice.
  gray[np.where(gray < (10, ))] = (255, )
  ret, gray = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

  gray = cv2.dilate(gray, np.ones((dilate_size, dilate_size)))
  gray = cv2.erode(gray, np.ones((erode_size, erode_size)))

  gray = cv2.GaussianBlur(gray, (3, 3), 0)
  ret, gray = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
  
  return gray