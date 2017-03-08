import cv2
import numpy as np
from distortion_matrix import DistortionMatrix
from key_point import Point
import math

WHITE = np.asarray([255, 255, 255])
BLACK = np.asarray([0, 0, 0])
GREY = np.asarray([216, 216, 216])
BLUE = np.asarray([255, 175, 100])
GREEN = np.asarray([0, 255, 0])
RED = np.asarray([0, 0, 255])
YELLOW = np.asarray([0, 255, 255])
ORANGE = np.asarray([0, 69, 255])
WALL_FLAG = np.asarray([255, 175, 100])


LG_KERNEL_SIZE = 20
LG_KERNEL = np.ones((LG_KERNEL_SIZE, LG_KERNEL_SIZE), np.uint8)
SM_KERNEL_SIZE = 5
SM_KERNEL = np.ones((SM_KERNEL_SIZE, SM_KERNEL_SIZE), np.uint8)
XS_KERNEL_SIZE = 2
XS_KERNEL = np.ones((XS_KERNEL_SIZE, XS_KERNEL_SIZE), np.uint8)

SOFT_KERNEL = np.asarray([
  [0.2, 0.4, 0.6, 0.4, 0.2],
  [0.4, 0.6, 1.0, 0.6, 0.4],
  [0.6, 1.0, 1.0, 1.0, 0.6],
  [0.4, 0.6, 1.0, 0.6, 0.4],
  [0.2, 0.4, 0.6, 0.4, 0.2],
])


# Dell
# SCREEN_SIZE = (1920, 1040)
# Mac
SCREEN_SIZE = (1440, 860)

GRID_SIZE = 5
GRID_OFFSET = 5

RELEVANT_LINE_LENGTH = 4


class MapMaker:
  def __init__(self, in_path, grid_size = GRID_SIZE, grid_offset = GRID_OFFSET):
    self.__img = cv2.imread(in_path)
    self.__in_path = in_path
    self.__grid_size = grid_size
    self.__grid_offset = grid_offset
    self.__distortion_matrix = DistortionMatrix(grid_size, grid_offset)
    # self.calculate_harris_corners()
  
  def __draw_points(self, points, img, size = 1):
    p1 = points[0].copy().add(Point(-size, -size)).asTuple()
    p2 = points[0].copy().add(Point(size, size)).asTuple()
    cv2.rectangle(img, p1, p2, RED)
    del points[0]

    for point in points:
      p1 = point.copy().add(Point(-size, -size)).asTuple()
      p2 = point.copy().add(Point(size, size)).asTuple()
      cv2.rectangle(img, p1, p2, YELLOW)
    return img

  def __get_square_neighbors(self, point, dot_img):
    height, width, channels = self.__img.shape
    neighbors = [point]

    right = point.copy().add(Point(GRID_SIZE, 0))
    if right.inBounds(width, height) and (dot_img[right.y, right.x] == WALL_FLAG).all():
      neighbors.append(right)

    down = point.copy().add(Point(0, GRID_SIZE))
    if down.inBounds(width, height) and (dot_img[down.y, down.x] == WALL_FLAG).all():
      neighbors.append(down)

    down_right = point.copy().add(Point(GRID_SIZE, GRID_SIZE))
    if down_right.inBounds(width, height) and (dot_img[down_right.y, down_right.x] == WALL_FLAG).all():
      neighbors.append(down_right)

    # print('Found point %s with %s neightbors' % (point, len(neighbors)))
    
    return neighbors

  def __get_t_neighbors(self, point, dot_img, show_preview = False):
    height, width, channels = self.__img.shape
    neighbors = [point]

    left = point.copy().add(Point(-GRID_SIZE, 0))
    if left.inBounds(width, height) and (dot_img[left.y, left.x] == WALL_FLAG).all():
      neighbors.append(left)

    right = point.copy().add(Point(GRID_SIZE, 0))
    if right.inBounds(width, height) and (dot_img[right.y, right.x] == WALL_FLAG).all():
      neighbors.append(right)

    down = point.copy().add(Point(0, GRID_SIZE))
    if down.inBounds(width, height) and (dot_img[down.y, down.x] == WALL_FLAG).all():
      neighbors.append(down)

    up = point.copy().add(Point(0, -GRID_SIZE))
    if up.inBounds(width, height) and (dot_img[up.y, up.x] == WALL_FLAG).all():
      neighbors.append(up)
    
    if show_preview:
      preview = dot_img.copy()
      preview = self.__draw_points(neighbors, img = preview)
      self.show_image(1, fullscreen = True, img = preview)

    return neighbors

  def __get_x_neighbors(self, point, dot_img):
    height, width, channels = self.__img.shape
    neighbors = [point]

    top_left = point.copy().add(Point(-GRID_SIZE, -GRID_SIZE))
    if top_left.inBounds(width, height) and (dot_img[top_left.y, top_left.x] == WALL_FLAG).all():
      neighbors.append(top_left)

    top_right = point.copy().add(Point(GRID_SIZE, -GRID_SIZE))
    if top_right.inBounds(width, height) and (dot_img[top_right.y, top_right.x] == WALL_FLAG).all():
      neighbors.append(top_right)

    down_left = point.copy().add(Point(-GRID_SIZE, GRID_SIZE))
    if down_left.inBounds(width, height) and (dot_img[down_left.y, down_left.x] == WALL_FLAG).all():
      neighbors.append(down_left)

    down_right = point.copy().add(Point(GRID_SIZE, GRID_SIZE))
    if down_right.inBounds(width, height) and (dot_img[down_right.y, down_right.x] == WALL_FLAG).all():
      neighbors.append(down_right)
    
    return neighbors


  def __get_tx_neighbors(self, point, dot_img):
    height, width, channels = self.__img.shape
    neighbors = [point]

    left = point.copy().add(Point(-GRID_SIZE, 0))
    if left.inBounds(width, height) and (dot_img[left.y, left.x] == WALL_FLAG).all():
      neighbors.append(left)

    right = point.copy().add(Point(GRID_SIZE, 0))
    if right.inBounds(width, height) and (dot_img[right.y, right.x] == WALL_FLAG).all():
      neighbors.append(right)

    down = point.copy().add(Point(0, GRID_SIZE))
    if down.inBounds(width, height) and (dot_img[down.y, down.x] == WALL_FLAG).all():
      neighbors.append(down)

    up = point.copy().add(Point(0, -GRID_SIZE))
    if up.inBounds(width, height) and (dot_img[up.y, up.x] == WALL_FLAG).all():
      neighbors.append(up)

    top_left = point.copy().add(Point(-GRID_SIZE, -GRID_SIZE))
    if top_left.inBounds(width, height) and (dot_img[top_left.y, top_left.x] == WALL_FLAG).all():
      neighbors.append(top_left)

    top_right = point.copy().add(Point(GRID_SIZE, -GRID_SIZE))
    if top_right.inBounds(width, height) and (dot_img[top_right.y, top_right.x] == WALL_FLAG).all():
      neighbors.append(top_right)

    down_left = point.copy().add(Point(-GRID_SIZE, GRID_SIZE))
    if down_left.inBounds(width, height) and (dot_img[down_left.y, down_left.x] == WALL_FLAG).all():
      neighbors.append(down_left)

    down_right = point.copy().add(Point(GRID_SIZE, GRID_SIZE))
    if down_right.inBounds(width, height) and (dot_img[down_right.y, down_right.x] == WALL_FLAG).all():
      neighbors.append(down_right)
    
    return neighbors

  def __remove_if_on_white(self, points, src_img, dot_img):
    for point in points:
      if not (src_img[point.y, point.x] == BLACK).all():
        # print('Killing off %s that is %s' % (point, src_img[point.y, point.x]))
        dot_img[point.y, point.x] = WHITE
    return dot_img

  def __kill_shorty(self, points, src_img, dot_img):
    line_length_sums = np.ndarray(len(points))
    for n in range(len(points)):
      line_length_sums[n] = self.__line_lengths(points[n], dot_img)

    min_index = np.argmin(line_length_sums)

    dot_img[points[min_index].y, points[min_index].x] = WHITE

    return dot_img

  def __line_lengths(self, point, dot_img, show_preview = False):
      h_length = self.__line_length_horizontal(point, dot_img, show_preview)
      v_length = self.__line_length_vertical(point, dot_img, show_preview)
      return(h_length + v_length)

  def __four_corners(self, dot_img, show_preview = False):
    height, width, channels = self.__img.shape
    for x in range(width - 1):
      for y in range(height - 1):
        if (dot_img[y, x] == WALL_FLAG).all():
          neighbors = self.__get_square_neighbors(Point(x, y), dot_img)
          if (len(neighbors) == 4):
            dot_img = self.__kill_shorty(neighbors, self.__img, dot_img)

            with_removed = dot_img.copy()
            with_removed[y, x] = GREEN

            if show_preview:
              preview = cv2.addWeighted(self.__img, 0.2, with_removed, 0.9, 0)
              self.show_image(20, fullscreen = True, img = preview, text = '(%s, %s)' % (x, y))
    return dot_img
  
  def __line_length_horizontal(self, point, dot_img, show_preview = False):
    height, width, channels = self.__img.shape
    far_left_point = point.copy()
    while far_left_point.x - GRID_SIZE > 0 and (dot_img[far_left_point.y, far_left_point.x - GRID_SIZE] == WALL_FLAG).all():
      far_left_point.add(Point(-GRID_SIZE, 0))
    
    far_right_point = far_left_point.copy()
    count = 0
    while far_right_point.x + GRID_SIZE < width and (dot_img[far_right_point.y, far_right_point.x + GRID_SIZE] == WALL_FLAG).all():
      count += 1
      far_right_point.add(Point(GRID_SIZE, 0))
    
    if show_preview:
      preview = dot_img.copy()
      cv2.line(preview, far_left_point.asTuple(), far_right_point.asTuple(), GREEN)

      p1 = point.copy().add(Point(-1, -1)).asTuple()
      p2 = point.copy().add(Point(1, 1)).asTuple()
      cv2.rectangle(preview, p1, p2, RED)

      preview = cv2.addWeighted(self.__img, 0.2, preview, 0.9, 0)
      self.show_image(1, fullscreen = True, img = preview, text = 'Point %s' % point)

    return count
  
  def __line_length_vertical(self, point, dot_img, show_preview = False):
    height, width, channels = self.__img.shape
    far_top_point = point.copy()
    while far_top_point.y - GRID_SIZE > 0 and (dot_img[far_top_point.y - GRID_SIZE, far_top_point.x] == WALL_FLAG).all():
      far_top_point.add(Point(0, -GRID_SIZE))
    
    far_bottom_point = far_top_point.copy()
    count = 0
    while far_bottom_point.y + GRID_SIZE < height and (dot_img[far_bottom_point.y + GRID_SIZE, far_bottom_point.x] == WALL_FLAG).all():
      count += 1
      far_bottom_point.add(Point(0, GRID_SIZE))
    
    if show_preview:
      preview = dot_img.copy()
      cv2.line(preview, far_top_point.asTuple(), far_bottom_point.asTuple(), GREEN)
      cv2.rectangle(preview, point.copy().add(Point(-1, -1)).asTuple(), point.copy().add(Point(1, 1)).asTuple(), RED)
      preview = cv2.addWeighted(self.__img, 0.2, preview, 0.9, 0)
      self.show_image(1, fullscreen = True, img = preview, text = 'Point %s' % point)

    return count

  def __is_corner(self, point, dot_img):
    h = self.__line_length_horizontal(point, dot_img) >= RELEVANT_LINE_LENGTH
    v = self.__line_length_vertical(point, dot_img) >= RELEVANT_LINE_LENGTH
    return h and v

  def __erode(self, dot_img, show_preview = False):
    height, width, channels = self.__img.shape
    for x in range(width - 1):
      for y in range(height - 1):
        if (dot_img[y, x] == WALL_FLAG).all():
          point = Point(x, y)
          neighbors = self.__get_t_neighbors(point, dot_img)

          if len(neighbors) == 1:
            dot_img[y, x] = WHITE

            if show_preview:
              preview = dot_img.copy()
              p1 = point.copy().add(Point(-1, -1)).asTuple()
              p2 = point.copy().add(Point(1, 1)).asTuple()
              cv2.rectangle(preview, p1, p2, RED)
              self.show_image(1000, fullscreen = True, img = preview, text = 'Removing island point %s' % point)

          elif len(neighbors) == 2:
            is_on_wall = (self.__img[y, x] == BLACK).all()
            if not is_on_wall:
              neighbor = neighbors[1]
              if self.__is_corner(neighbor, dot_img):
                dot_img[y, x] = WHITE
              elif self.__line_lengths(point, dot_img) < RELEVANT_LINE_LENGTH:
                dot_img[y, x] = WHITE
            
            if show_preview:
              preview = dot_img.copy()
              p1 = point.copy().add(Point(-1, -1)).asTuple()
              p2 = point.copy().add(Point(1, 1)).asTuple()
              cv2.rectangle(preview, p1, p2, ORANGE)
              self.show_image(1000, fullscreen = True, img = preview, text = 'Outlier %s' % point)
          else:
            if show_preview:
              preview = dot_img.copy()
              p1 = point.copy().add(Point(-1, -1)).asTuple()
              p2 = point.copy().add(Point(1, 1)).asTuple()
              cv2.rectangle(preview, p1, p2, GREEN)
              self.show_image(1, fullscreen = True, img = preview, text = 'Point %s passed erosion with %s neighbors' % (point, len(neighbors)))
    return dot_img
  
  def __draw_t_lines(self, origin, points, img, color = BLUE, show_preview = False):
    for point in points:
      cv2.line(img, origin.asTuple(), point.asTuple(), color)
    if show_preview:
      self.show_image(10, fullscreen = True, img = img)
    return img


  def __wall_builder(self, dot_img, show_preview = False):
    height, width, channels = self.__img.shape
    wall_img = dot_img.copy()
    wall_img[:] = WHITE
    for x in range(width - 1):
      for y in range(height - 1):
        if (dot_img[y, x] == WALL_FLAG).all():
          point = Point(x, y)
          neighbors = self.__get_t_neighbors(point, dot_img)
          wall_img = self.__draw_t_lines(point, neighbors, wall_img)

          if show_preview:
            preview = cv2.addWeighted(self.__img, 0.5, dot_img, 0.5, 0)
            preview = cv2.addWeighted(preview, 0.5, wall_img, 0.5, 0)
            self.show_image(100, fullscreen = True, img = preview)
    return wall_img


  def snap_walls(self, show_preview = False):
    dot_img = self.__img.copy()
    dot_img[:] = WHITE
    height, width, channels = self.__img.shape
    # print('%s x %s' % (height, width))
    for x in range(width - 1):
      for y in range(height - 1):
        if self.__img[y, x, 0] == 0 and self.__img[y, x, 1] == 0 and self.__img[y, x, 2] == 0:
          snap_x = int(round(x / GRID_SIZE) * GRID_SIZE)
          snap_y = int(round(y / GRID_SIZE) * GRID_SIZE)
          dot_img[snap_y, snap_x] = WALL_FLAG

    # preview = cv2.addWeighted(self.__img, 0.2, dot_img, 0.9, 0)
    # self.show_image(0, fullscreen = True, img = preview)

    if show_preview:
      preview = cv2.addWeighted(self.__img, 0.2, dot_img, 0.9, 0)
      self.show_image(500, fullscreen = True, img = preview, text = 'After snap_walls()')

    dot_img = self.__four_corners(dot_img, show_preview = show_preview)

    if show_preview:
      preview = cv2.addWeighted(self.__img, 0.2, dot_img, 0.9, 0)
      self.show_image(500, fullscreen = True, img = preview, text = 'After four_corners()')

    dot_img = self.__erode(dot_img, show_preview = show_preview)

    if show_preview:
      preview = cv2.addWeighted(self.__img, 0.2, dot_img, 0.9, 0)
      self.show_image(500, fullscreen = True, img = preview, text = 'After erode()')

    wall_img = self.__wall_builder(dot_img, show_preview = False)

    if show_preview:
      preview = cv2.addWeighted(self.__img, 0.2, wall_img, 0.9, 0)
      self.show_image(500, fullscreen = True, img = preview, text = 'After wall_builder()')

    # optional preview:
    # preview = cv2.addWeighted(self.__img, 0.5, dot_img, 0.5, 0)
    # preview = cv2.addWeighted(preview, 0.5, wall_img, 0.5, 0)
    # self.__img = preview
    self.__img = wall_img
  def blur(self, amount):
    # self.__img = cv2.blur(self.__img, (amount, amount))
    self.__img = cv2.erode(self.__img, LG_KERNEL, iterations = 1)
    self.__img = cv2.dilate(self.__img, SOFT_KERNEL, iterations = 1)

    # self.__img = cv2.blur(self.__img, (amount, amount))
    # self.__img = cv2.erode(self.__img, SOFT_KERNEL, iterations = 1)
    # self.__img = cv2.dilate(self.__img, SOFT_KERNEL, iterations = 1)

    # self.__img = cv2.blur(self.__img, (amount, amount))
    # self.__img = cv2.erode(self.__img, SOFT_KERNEL, iterations = 1)
    # self.__img = cv2.dilate(self.__img, SOFT_KERNEL, iterations = 1)
  def grey_to_white(self):
    height, width, channels = self.__img.shape
    b_channel, g_channel, r_channel = cv2.split(self.__img)
    alpha_channel = b_channel.copy()
    self.__img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    for x in range(width - 1):
      for y in range(height - 1):
        alpha = 0
        if self.__img[y, x, 0] > 0:
          alpha = 255
        color = math.floor(math.atan(self.__img[y, x, 0]) * 150)
        # self.__img[y, x] = [color, color, color, alpha]
        self.__img[y, x, 0] = color
        self.__img[y, x, 1] = 20
        self.__img[y, x, 2] = 20
        self.__img[y, x, 3] = alpha


        # if self.__img[y, x, 0] > 0 and self.__img[y, x, 1] > 0 and self.__img[y, x, 2] > 0:
        #   self.__img[y, x] = WHITE
        # For colors:
          # self.__img[y, x] = [self.__img[y, x, 0] + 200, 0, self.__img[y, x, 0] + 200]
  def apply_distortion_matrix(self):
    height, width, channels = self.__img.shape
    matrix = self.__distortion_matrix.get_perspective_transform(width, height)
    # self.__img = cv2.perspectiveTransform(self.__img, matrix)
    self.__img = cv2.warpPerspective(self.__img, matrix, (width, height))

  def crop_to_distortion_matrix(self):
    top_left = self.__distortion_matrix.top_left
    bottom_right = self.__distortion_matrix.bottom_right
    self.__img = self.__img[top_left.x:bottom_right.x, top_left.y:bottom_right.y]
    
  def calculate_harris_corners(self):
    grey = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
    # harris_corners = cv2.cornerHarris(grey, blockSize = 2, ksize = 13, k = 0.1)
    harris_corners = cv2.cornerHarris(grey, blockSize = 2, ksize = 5, k = 0.1)
    # self.__distortion_matrix.add_harris_corners(harris_corners, self.__img)
    self.__distortion_matrix.add_harris_corners(harris_corners)
    self.harris_corners = harris_corners

  def draw_harris_corners(self, color = GREEN):
    corners = self.harris_corners
    # corners = cv2.dilate(corners, None)
    self.__img[corners > 0.01 * corners.max()] = color

  def smooth_harris_corners(self):
    self.harris_corners = cv2.dilate(self.harris_corners, SM_KERNEL, iterations = 2)
    self.harris_corners = cv2.erode(self.harris_corners, SM_KERNEL, iterations = 1)

  def draw_distorted_corners(self, color = BLUE):
    self.__img = self.__distortion_matrix.drawOn(self.__img, color)

  def color_shift(self):
    SWAP = [230, 230, 230]
    self.__img[np.where((self.__img == BLACK).all(axis = 2))] = WHITE
    self.__img[np.where((self.__img <= GREY).all(axis = 2))] = BLACK
  
  def fill(self, color):
    self.__img[:] = color

  def show_image(self, time, fullscreen = False, img = None, text = None):
    if img is None:
      img = self.__img
    if fullscreen:
      img = cv2.resize(img, SCREEN_SIZE, interpolation = cv2.INTER_AREA)
    if text is not None:
      cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.namedWindow(self.__in_path, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(self.__in_path, img)
    # cv2.moveWindow(self.__in_path, 1680, -400)
    cv2.moveWindow(self.__in_path, 0, 0)
    cv2.waitKey(time)
    cv2.destroyWindow(self.__in_path)

  def basic_cleanup(self):
    # Make black area bigger
    # for i in range(20):
    self.__img = cv2.erode(self.__img, SM_KERNEL, iterations = 2)
    self.__img = cv2.dilate(self.__img, SOFT_KERNEL, iterations = 2)

  def draw_grid(self):
    width = len(self.__img)
    height = len(self.__img[0])
    for i in range(0, height, self.__grid_size):
        cv2.line(self.__img, (i + self.__grid_offset, 0), (i + self.__grid_offset, width), RED)
    for j in range(0, width, self.__grid_size):
        cv2.line(self.__img, (0, j + self.__grid_offset), (height, j + self.__grid_offset), RED)

  def write_to_disk(self, path):
    cv2.imwrite(path, self.__img)
  def getImg(self):
    return self.__img.copy()
  def flip(self):
    self.__img = cv2.flip(self.__img, 0)

  def get_contours(self):
    ret, thresh = cv2.threshold(grey, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

  def resize(self, dim = SCREEN_SIZE):
    self.__img = cv2.resize(self.__img, dim, interpolation = cv2.INTER_AREA)
  def scale(self, factor):
    self.__img = cv2.resize(self.__img, None, fx=factor, fy=factor, interpolation = cv2.INTER_AREA)
