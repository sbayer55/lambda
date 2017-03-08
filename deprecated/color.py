def hsla2bgra(h, s, l, a):
  h = float(h % 360)
  """
  hsla color -> bgra color
  h = range(0, 360)
  s = range(0, 1)
  l = range(0, 1)
  a = range(0, 255)
  """
  c = (1.0 - abs(2.0 * l - 1.0)) * s
  x = c * (1.0 - abs(h / 60.0 % 2.0 - 1.0))
  m = l - c / 2.0
  if 0 <= h < 60:
    return [int(round(0 * 255)), int(round(x * 255)), int(round(c * 255)), a]
  elif 60 <= h < 120:
    return [int(round(0 * 255)), int(round(c * 255)), int(round(x * 255)), a]
  elif 120 <= h < 180:
    return [int(round(x * 255)), int(round(c * 255)), int(round(0 * 255)), a]
  elif 180 <= h < 240:
    return [int(round(0 * 255)), int(round(x * 255)), int(round(c * 255)), a]
  elif 240 <= h < 300:
    return [int(round(c * 255)), int(round(0 * 255)), int(round(x * 255)), a]
  elif 300 <= h < 360:
    return [int(round(x * 255)), int(round(0 * 255)), int(round(c * 255)), a]
  else:
    raise ValueError
