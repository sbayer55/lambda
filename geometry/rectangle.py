import numpy as np

class Rectangle:
  @staticmethod
  def expand(rect, amount):
    return np.hstack((
        rect[0, 0:2] - amount, 
        rect[0, 2:4] + amount)
      )[np.newaxis, :]
  
  @staticmethod
  def isInside(point, rect):
    (x, y), = point
    (left, top, right, bottom), = rect

    return left < x < right and top < y < bottom