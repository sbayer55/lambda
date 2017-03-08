import cv2, os, math
import numpy as np
from nodejs import bridge
import matplotlib.pyplot as plt

# Hardcoded switch to disable rendering in production
SHOW = True
SCREEN_SIZE = (1440, 860)
EXT = '.png'

def same_size(img_a, img_b):
  return img_a.shape == img_b.shape

def map_val(x, min, max, out_min, out_max):
  if (max - min) == 0 or (out_max - out_min) == 0:
    return 0
  slope = float(out_max - out_min) / float(max - min)
  # log("Slope: %s" % slope)
  val = out_min + (slope * (x - min))
  return int(round(val))

def getDistanceUnit(img):
  """Get unit of visible distance from image

  Args:
    img (array_list): OpenCV img matrix

  Returns:
    int: unit
  """
  width, height, channels = img.shape
  size = math.sqrt(width ** 2 + height ** 2)
  unit = (size / 75)
  return unit

def rotationMatrix(theta):
    """Matrix to rotate point
    
    Args:
        theta (int): Radians to rotate point.
        
    Returns:
        array_like: [[cos, -sin], [sin, cos]]
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def show(img, cvt = cv2.COLOR_BGR2RGB, size = 3, time = -1, text = None):
    img = img.copy()
    if cvt is not None:
        img = cv2.cvtColor(img, cvt)
        
    
    if text is not None:
        cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        
    if time < 0:
        plt.figure(figsize = (size, size))
        plt.axis("off")
        plt.imshow(img, interpolation='nearest')
        plt.show()
    else:
        img = cv2.resize(img, None, fx=size, fy=size, interpolation = cv2.INTER_AREA)
        cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("preview", img)
        cv2.waitKey(time)
        cv2.destroyWindow("preview")

def scale(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_AREA)

def save(img, physical_path, alias, file_type):
  """Save OpenCV2 image matrix to a file.

  Write the file to /tmp using the temp_file(tpath) to resolve location. Also
  notifies NodeJS threads monitoring this processes stdout.

  Args:
    img (numpy.ndarray): The image data being saved
    temp_file (string): tpath for the source image
    temp_file_name (string): extName for the source image
    output_type (string): the suffix to the output file
        "2017-01-23-18-18-14_pilot12_6166_<output_type>"

  Returns:
    None
  
  Raises:
    None
  """
  # '2017-01-23-18-18-14_pilot12_6166_coverage.pgm' => 
  #                 '2017-01-23-18-18-14_pilot12_6166'
  prefix = alias[0:alias.rfind('_')]

  # '2017-01-23-18-18-14_pilot12_6166' + '_' + 'coverage' + '.png'
  out_alias = prefix + '_' + file_type + EXT

  # '/var/folders/yl/31853pjx2lj80f8z5cqsbdww0000gn/T/upload_d041886e18e780a07ed751a46ee8183a' + '_PYTHON_' + 'coverage' + '.png'
  out_path = physical_path + '_PYTHON_' + file_type + EXT
  try:
    cv2.imwrite(out_path, img)
    bridge.fileMade(out_path, alias)
  except:
    e_type, value, tb = sys.exc_info()
    stack = ''.join(traceback.format_tb(tb))
    bridge.error('Unhandled exception (when calling cv2.imwrite)" %sDescription: %sStack: %s' % (e_type, value, stack))

def largestContour(img):
  contours, hierarchy = cv2.findContours(img,
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

  largest = contours[0]
  largest_size = cv2.contourArea(largest)
  for cnt in contours:
    size = cv2.contourArea(cnt)
    if (size > largest_size):
      largest = cnt
      largest_size = size
  return largest

def factor(img, factor):
  """Used to scale an image linearly.

  Args:
    img (array_like): A list of points (ndarray)
    factor (float): new_rgb = rgb * factor = (0, 0, 255) * (0.5)

  Returns:
    array_like: A mutated copy of img
  """
  temp = img.copy()
  temp = temp * factor
  temp = temp.astype('uint8')
  return temp

def about(points):
  """Calculates stats based on a series of points.

  Args:
    points (array_like): A list of points (ndarray)
  """
  points
  line = cv2.fitLine(points, cv2.NORM_L2, 0, 0.01, 0.01)
  diff = points[1:] - points[:-1]
  avg_diff = np.linalg.norm(diff)
  thetas = np.arctan2(diff[:, :, 0], diff[:, :, 1])
  theta_max = np.argmax(np.abs(thetas))
  theta = np.average(thetas)
  sample_dist = np.linalg.norm(points[-1, 0] - points[1, 0])