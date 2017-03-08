import numpy as np
import matplotlib.pyplot as plt
import cv2
from draw.colors import COLORS
from file_processor import util
from manipulate import hull
from cluster.line_cluster import LineCluster
import geometry as geo
from draw.draw_points import drawPoints

def process(file, files, path):
  img = cv2.imread(file)
  # img = img[130:, :, :]

  # Get outer hull
  img_hull = hull.get(img)

  # Find contour points:
  cnt = util.largestContour(img_hull)

  # Cluster contour points:
  cluster = LineCluster(points = cnt, 
    sampleSize = 5, 
    avgThresh = 5, 
    minThresh = 5,
    img = img,
    path = path)

import os
def run(file, files):
  # path = '/Users/steven/maidbot_ros/maidbot_mapping/maps/marriott_houston_airport/1008/1008.pgm'
  # process(path, [path], '/Users/steven/Desktop/maps/00.png')

  path = '/Users/steven/maidbot_ros/maidbot_mapping/maps/marriott_houston_airport'
  todo = []
  for root, dirs, files in os.walk(path):
    for name in files:
      if '.pgm' in name and not 'keepout' in name:
        path = os.path.join(root, name)
        todo.append(path)

  for n, file in enumerate(todo):
    out = '/Users/steven/Desktop/maps/%s.png' % n
    print('%s of %s %s' % (n, len(todo), file))
    img = process(file, files, out)