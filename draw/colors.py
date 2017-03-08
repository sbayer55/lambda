from collections import namedtuple
import numpy as np

class Rainbow:
  def __init__(self, 
    frequencyR, frequencyG, frequencyB,
    phaseR, phaseG, phaseB,
    widthR=127, widthG=127, widthB=127,
    minR=127, minG=127, minB=127, 
    step=np.radians(30)):
    """Class for generating a string of colors

    Args:
      frequencyR (int):
      frequencyG (int):
      frequencyB (int):
      phaseR (int):
      phaseG (int):
      phaseB (int):
      width (int): 1 - 127 effects luminosity
      center (int):
      step (int):
    """
    self.frequencyR = frequencyR
    self.frequencyG = frequencyG
    self.frequencyB = frequencyB
    self.phaseR = phaseR
    self.phaseG = phaseG
    self.phaseB = phaseB
    self.x = 0
    self.step = step
    self.widthR = widthR
    self.widthG = widthG
    self.widthB = widthB
    self.minR = minR
    self.minG = minG
    self.minB = minB

  def next(self, f=1.0):
    """Returns colors in a rainbow

    Returns:
      (b, g, r): color tuple
    """
    self.x += self.step

    r = (np.sin(self.frequencyR * self.x + 
      self.phaseR) * self.widthR + self.minR) * f
    g = (np.sin(self.frequencyG * self.x + 
      self.phaseG) * self.widthG + self.minG) * f
    b = (np.sin(self.frequencyB * self.x + 
      self.phaseB) * self.widthB + self.minB) * f

    return (b, g, r)

  def skip(self):
    self.x += np.radians(180)

COLORS = namedtuple('COLORS', ['red', 'yellow', 'green', 'blue', 'purple', 'cyan', 'gray', 'white', 'rainbow'])
COLORS.red = (0, 0, 255)
COLORS.yellow = (0, 255, 255)
COLORS.green = (0, 255, 0)
COLORS.brown = (45, 100, 175)
COLORS.blue = (255, 0, 0)
COLORS.purple = (255, 0, 255)
COLORS.cyan = (255, 255, 0)
COLORS.gray = (170, 170, 170)
COLORS.white = (255, 255, 255)
COLORS.rainbow = Rainbow(0.3, 0.3, 0.3,
  0, np.radians(90), np.radians(180))