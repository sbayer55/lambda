import math, cv2
import numpy as np
import numpy.linalg as LA
from draw.colors import COLORS
from file_processor import util

def asPoints(line):
  line = line.copy()
  line[0, 0:2] += line[0, 2:4]
  return line[0]

def drawLine(img, line, color=COLORS.yellow, thickness=1):
  size = 1000
  vx, vy, x, y = line[0]
  p1 = (int(x + vx * -size), int(y + vy * -size))
  p2 = (int(x + vx * size), int(y + vy * size))
  cv2.line(img, p1, p2, color, thickness)

def drawLines(img, lines, color=COLORS.yellow, thickness=1):
  for line in lines:
    drawLine(img, line, color, thickness)

def drawLineSegment(img, line, color=COLORS.yellow, thickness=1):
  p1 = tuple(line[0, 0:2].astype(np.int))
  p2 = tuple(line[0, 2:4].astype(np.int))
  cv2.line(img, p1, p2, color, thickness)

def drawUnitVector(img, uvline, color=COLORS.yellow, thickness=1):
  unit = util.getDistanceUnit(img)

  p1 = tuple(uvline[0, 2:4].astype(np.int))

  uv = uvline[0, 0:2]
  target = p1 + uv * (unit * 5)
  p2 = tuple(target.astype(np.int))

  rot = util.rotationMatrix(np.radians(210))
  tip = target + np.matmul(rot, uv) * unit
  p3 = tuple(tip.astype(np.int))
  
  rot = util.rotationMatrix(np.radians(150))
  tip = target + np.matmul(rot, uv) * unit
  p4 = tuple(tip.astype(np.int))
  
  cv2.line(img, p1, p2, color, thickness)
  cv2.line(img, p2, p3, color, thickness)
  cv2.line(img, p2, p4, color, thickness)

def distFrom(line, point):
  """A Points distance from a line

  Line is defined by UVLine but is extended infinitly for the calculation.

  Args:
    line (array_like): ndarray (1, 4)
    point (array_like): ndarray (1, 2)

  Returns:
    int: distance
  """
  x2, y2, x1, y1 = line[0]
  x2 = x1 + x2
  y2 = y1 + y2
  x0, y0 = point[0]
  return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def intersection(l1, l2):
  """Determins the theoretical intersection of two line segments

  Args:
    l1 (array_like): UVLine / ndarray (1, 4)
    l2 (array_like): UVLine / ndarray (1, 4)

  Returns:
    array_like: ndarray (1, 2)
  """
  x2, y2, x1, y1 = asPoints(l1)
  x4, y4, x3, y3 = asPoints(l2)
  
  d = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
  
  if d == 0:
    return np.array([[x1, y1]], np.int32)
  
  x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
  y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
  
  return np.array([[x, y]], np.int32)


def length(line):
  return LA.norm(line[0, 2:4] - line[0, 0:2])

def isColinear(uvLine1, uvLine2, thresh = 0.45):
  uv1 = uvLine1[0, 0:2]
  uv2 = uvLine2[0, 0:2]
  return np.dot(uv1, uv2) > thresh

def asIntersections(lines, thresh = 10):
  """Get intersections from an array of lines

  Args:
    lines (array_like): ndarray (n, 1, 4) of UVLines
  
  Returns:
    array_like: ndarray (n, 1, 2) of Points
  """
  points = np.empty((lines.shape[0], 1, 2), np.int32)
  for n in range(lines.shape[0] - 1):
    l1 = lines[n]
    l2 = lines[n + 1]
    p = intersection(l1, l2)
    points[n] = p
  
  # Get wrapping point
  l1 = lines[-1]
  l2 = lines[0]
  p = intersection(l1, l2)
  points[-1] = p
  
  return points