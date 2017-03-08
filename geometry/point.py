import numpy as np
import numpy.linalg as LA
from cluster.cluster_bin import ClusterBin
import matplotlib.pyplot as plt
from draw.colors import COLORS
import cv2

def drawPoint(img, point, color=COLORS.red, thickness=1, label=None):
  p = tuple(point[0])
  if thickness == 0:
    img[p[1],p[0]] = color
  else:
    cv2.circle(img, p, 3, color, thickness)
    if label:
      cv2.putText(img, 
        label, 
        (p[0] + 5, p[1] + 2), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.3, 
        COLORS.cyan)

def drawPoints(img, points, color=COLORS.red, thickness=1):
  for n, point in enumerate(points):
    drawPoint(img, point, color, thickness, label=str(n))

def filterNoise(points, thresh, sampleSize = 4):
  """Filter out redundant or misleading points

  Args:
    points (array_like): ndarray(n, 1, 2)
    thresh (float): max avg dist point to point in sample range(-1, 1)
    sampleSize (int): number of points in sample
  """
  diff = points[:-1] - points[1:]
  mask = np.zeros((diff.shape[0] + 1), 'bool')
  avg_dist = np.zeros((diff.shape[0] + 1))
  
  for i in range(diff.shape[0]):
    cbin = ClusterBin(i, i + sampleSize)
    sample = cbin.getFrom(diff)
    avg_dist[i] = np.average(LA.norm(sample, axis=2))
    if (avg_dist[i] < thresh):
      mask[i:i + sampleSize] = True
          
  # noise = points[mask]
  plt.plot(avg_dist)
  plt.grid(True)
  plt.show()

  # noise_avg = np.average(noise, axis=0)
  # points[mask] = 0
  # first_noise = np.argmax(mask)
  # points[first_noise] = noise_avg
  # mask[first_noise] = False
  points = points[mask == False]
  return points