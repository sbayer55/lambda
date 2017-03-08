from map_maker import MapMaker
import sys, json, traceback
import util
import cv2
import numpy as np


try:
  # Parse command line args
  cov_path = sys.argv[1]
  cov_path_name = sys.argv[2]
  occ_path = sys.argv[3]
  occ_path_name = sys.argv[4]

  # Load occupancy map and coverage
  occ = cv2.imread("2016-10-08-01-01-01_pilot01_6166_occupancy.png", cv2.CV_LOAD_IMAGE_COLOR)
  cov = cv2.imread("2017-01-23-18-18-14_pilot12_6166_coverage.pgm", cv2.CV_LOAD_IMAGE_COLOR)

  # Make black and white
  cv2.threshold(cov, 1, 255, cv2.THRESH_BINARY, cov)

  # Flip vertically (0)
  cv2.flip(cov, 0, cov)

  # Grow image by 8x8 pixels
  kernel_size = 8
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  cv2.dilate(cov, kernel, cov)

  # Apply 12 pixel Gaussian Blue
  blur = 12
  cv2.GaussianBlur(cov, (blur * 2 + 1, blur * 2 + 1), 0, cov)

  # Shrink by 13x13 pixels
  kernel_size = 13
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  cv2.erode(cov, kernel, cov)

  # Remove blurred pixels / make image black and white
  cv2.threshold(cov, 127, 255, cv2.THRESH_BINARY_INV, cov)

  # Render preview
  result = cv2.addWeighted(occ, 0.6, cov, 0.4, 0.5)
  util.show(3000, result)

  # Write results to disk
  util.save(cov, temp_file + '-coverage', temp_file_name[0:-15] + 'coverage.png')

except:
  # On exception tell NodeJS watcher and exit gracefully
  e_type, value, tb = sys.exc_info()
  stack = ''.join(traceback.format_tb(tb))
  bridge.error('Unhandled exception" %s\nDescription: %s\nStack: %s' % (e_type, value, stack))