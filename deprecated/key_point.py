from math import sqrt, pow

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.magnitude = sqrt((x ** 2) + (y ** 2))

  def __repr__(self):
    return '(%s, %s)' % (self.x, self.y)

  def __str__(self):
    return '(%s, %s)' % (self.x, self.y)

  def normalize(self):
    return Point(self.x / self.magnitude, self.y / self.magnitude)

  def onGrid(self, grid_size, grid_offset = 0):
    dx = int(self.x - round(self.x + (grid_size / 2)) % grid_size) + grid_offset
    dy = int(self.y - round(self.y + (grid_size / 2)) % grid_size) + grid_offset
    return Point(dx, dy)

  def distance(self, point):
    return sqrt(((point.x - self.x) ** 2) + ((point.y - self.y) ** 2))

  def add(self, point):
    self.x += point.x
    self.y += point.y
    return self

  def divide(self, value):
    self.x /= value
    self.y /= value

  def copy(self):
    return Point(self.x, self.y)

  def asTuple(self):
    return (self.x, self.y)
  def inBounds(self, width, height):
    return self.x >= 0 and self.x < width and self.y >= 0 and self.y < height

class TranslationPoint:
  def __init__(self, x, y, grid_size, grid_offset, value = 0):
    self.src = Point(x, y)
    self.value = value
    self.dst = self.src.onGrid(grid_size, grid_offset)

  def __repr__(self):
    return '%s -> %s' % (self.src, self.dst)

  def __str__(self):
    return '%s -> %s' % (self.src, self.dst)
      