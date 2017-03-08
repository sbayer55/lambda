import cv2
import numpy as np
from key_point import Point, TranslationPoint

BLACK = np.asarray([0, 0, 0])
PINK = np.asarray([255, 0, 255])
YELLOW = np.asarray([0, 255, 255])

# Dell
SCREEN_SIZE = (1920, 1040)
# Mac
# SCREEN_SIZE = (1440, 860)

class DistortionMatrix:
  def __init__(self, grid_size, grid_offset):
    self.__points = []
    self.__grid_size = grid_size
    self.__grid_offset = grid_offset
    self.top_left = Point(99999, 99999)
    self.bottom_right = Point(0, 0)

  def __show_progress(self, cv_image):
    img = self.drawOn(cv_image)
    img = cv2.resize(img, SCREEN_SIZE, interpolation = cv2.INTER_AREA)
    # info = 'i %s welding %s points ' % (i, len(weld_targets))
    # cv2.putText(img, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    window_name = 'PROGRESS'

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, img)
    # cv2.moveWindow(window_name, 1680, -400)
    cv2.moveWindow(window_name, 0, 0)
    cv2.waitKey(1)
    cv2.destroyWindow(window_name)

  def __calculate_bounds(self):
    for point in self.__points:
      if point.dst.x < self.top_left.x:
        self.top_left.x = point.dst.x
      if point.dst.y < self.top_left.y:
        self.top_left.y = point.dst.y
      if point.dst.x > self.bottom_right.x:
        self.bottom_right.x = point.dst.x
      if point.dst.y > self.bottom_right.y:
        self.bottom_right.y = point.dst.y

    if self.top_left.x < 0:
      self.top_left.x = 0
    if self.top_left.y < 0:
      self.top_left.y = 0

  def __auto_weld(self, weld_distance = 4, cv_image = None):
    # print('Before weld: %s' % self.__points)
    i = 0
    while i < len(self.__points):
      a = self.__points[i].src.copy()
      weld_targets = []

      # Compair i to points > i:
      for j in range(i + 1, len(self.__points)):
        b = self.__points[j].src
        if a.distance(b) < weld_distance:
          weld_targets.append(j)

      if len(weld_targets) > 0:
        for index in reversed(weld_targets):
          a.add(self.__points[index].src.copy())
          del self.__points[index]
        a.divide(len(weld_targets) + 1)
        self.__points[i] = TranslationPoint(a.x, a.y, self.__grid_size, self.__grid_offset)
        # i = 0
        # i += 1
      else:
        i += 1
      
      if cv_image is not None:
        self.__show_progress(cv_image)
        
    self.__calculate_bounds()
    # print('After weld: %s' % self.__points)
  
  # def __make_distortion_matrix(self):
  #   self.__matrix = np.ndarray(shape = self.__size.asTuple(), dtype = float)

  def add_point(self, x, y, value = 0):
    point = TranslationPoint(x, y, self.__grid_size, self.__grid_offset, value)
    self.__points.append(point)

  def add_harris_corners(self, harris_corners, cv_image = None):
    threshold = 0.05 * harris_corners.max()
    for i in range(len(harris_corners)):
        for j in range(len(harris_corners[i])):
            if harris_corners[i][j] > threshold:
              self.add_point(i, j, harris_corners[i][j])
    self.__auto_weld(cv_image = cv_image)

  def get_perspective_transform(self, img_width, img_height):
    top_left_point = Point(0, 0)
    top_right_point = Point(img_width, 0)
    bottom_left_point = Point(0, img_height)
    bottom_right_point = Point(img_width, img_height)

    center = Point(round(img_width / 2), round(img_width / 2))

    top_left_ref = (top_left_point.distance(center), None)
    top_right_ref = (top_right_point.distance(center), None)
    bottom_left_ref = (bottom_left_point.distance(center), None)
    bottom_right_ref = (bottom_right_point.distance(center), None)

    for point in self.__points:
      top_left_dist = top_left_point.distance(point.src)
      top_right_dist = top_right_point.distance(point.src)
      bottom_left_dist = bottom_left_point.distance(point.src)
      bottom_right_dist = bottom_right_point.distance(point.src)

      if top_left_dist < top_left_ref[0]:
        top_left_ref = (top_left_dist, point)
      if top_right_dist < top_right_ref[0]:
        top_right_ref = (top_right_dist, point)
      if bottom_left_dist < bottom_left_ref[0]:
        bottom_left_ref = (bottom_left_dist, point)
      if bottom_right_dist < bottom_right_ref[0]:
        bottom_right_ref = (bottom_right_dist, point)

    src = np.float32([[top_left_ref[1].src.x, top_left_ref[1].src.y], 
      [top_right_ref[1].src.x, top_right_ref[1].src.y],
      [bottom_left_ref[1].src.x, bottom_left_ref[1].src.y],
      [bottom_right_ref[1].src.x, bottom_right_ref[1].src.y]])

    dst = np.float32([[top_left_ref[1].dst.x, top_left_ref[1].dst.y], 
      [top_right_ref[1].dst.x, top_right_ref[1].dst.y],
      [bottom_left_ref[1].dst.x, bottom_left_ref[1].dst.y],
      [bottom_right_ref[1].dst.x, bottom_right_ref[1].dst.y]])
    
    matrix = cv2.getPerspectiveTransform(src, dst)

    print('Transfrom based on %s to %s\n%s' % (src, dst, matrix))

    return matrix

  def drawOn(self, cv_image, src_color = YELLOW, dest_color = PINK):
    points = cv_image.copy()
    points[:] = BLACK
    for point in self.__points:
      points[point.src.x, point.src.y] = src_color
      points[point.dst.x, point.dst.y] = dest_color
    points = cv2.dilate(points, None)

    return cv2.addWeighted(cv_image, 0.5, points, 1.0, 0)