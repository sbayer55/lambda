import csv
import cv2
import numpy as np
import sys, json
from bridge import *
from util import *
from color import *
import math

# 0.1524 = 1 sample per 6"
SAMPLE_PER_METER = 0.1524

CSV_TPATH = sys.argv[1]
CSV_NAME = sys.argv[2]
OCCUPANCY_GRID_TPATH = sys.argv[3]

TRANSPARENT = [0, 0, 0, 0]

with open(CSV_TPATH, 'rb') as csvfile:
  data = csv.reader(csvfile, delimiter=',', quotechar='|')
  meta_header = next(data)
  meta = next(data)
  room_number = meta[0]
  unit_name = meta[1]
  unit_uuid = meta[2]
  origin_x = float(meta[3])
  origin_y = float(meta[4])
  resolution = float(meta[5])
  header = next(data)

  occ_img = cv2.imread(OCCUPANCY_GRID_TPATH)

  if occ_img is None:
    log('Unable to open %s' % OCCUPANCY_GRID_TPATH)
  else:
    occ_img = vflip(occ_img)
    occ_img = add_alpha_channel(occ_img)
    temp_img = occ_img.copy()
    temp_img[:] = TRANSPARENT
    humidity_img = temp_img.copy()
    dust_img = temp_img.copy()

    min_vals = np.full(9, 999999, dtype=int)
    max_vals = np.full(9, -999999, dtype=int)
    
    data = list(data)
    for row in data:
      for n in range(0, 9):
        val = float(row[n])
        min_vals[n] = min(min_vals[n], val)
        max_vals[n] = max(max_vals[n], val)

    for row in data:
      time = int(row[0]) if len(row[0]) > 0 else 0
      x = int(round((float(row[1]) - origin_x) / resolution)) if len(row[1]) > 0 else 0
      y = int(round((float(row[2]) - origin_y) / resolution)) if len(row[2]) > 0 else 0
      back_temp = float(row[3]) if len(row[3]) > 0 else 0
      front_temp = float(row[4]) if len(row[4]) > 0 else 0
      back_pressure = float(row[5]) if len(row[5]) > 0 else 0
      front_pressure = float(row[6]) if len(row[6]) > 0 else 0
      humidity = float(row[7]) if len(row[7]) > 0 else 0
      dust_readings = int(row[8]) if len(row[8]) > 0 else 0
      VOC = float(row[9]) if len(row[9]) > 0 else 0

      temp_blue = map_val(front_temp, min_vals[4], max_vals[4], 50, 250)
      temp_red = 250 - map_val(front_temp, min_vals[4], max_vals[4], 50, 250)

      temp_img[y, x] = [temp_blue, 50, temp_red, 255]

      humidity_range = 90
      humidity_hue = map_val(humidity, 69.3, 70.3, 0, humidity_range) + 220
      humidity_color = hsla2bgra(humidity_hue, 1.0, 0.5, 200)
      print("%s => %s" % (humidity, humidity_color))

      humidity_img[y, x] = humidity_color

      dust_alpha = map_val(dust_readings, min_vals[8], max_vals[8], 50, 150)
      dust_rad = map_val(dust_readings, min_vals[8], max_vals[8], 1, 4)
      dust_color = [0, 0, 0, dust_alpha]
      cv2.circle(dust_img, (x, y), dust_rad, dust_color, -1)

    save(temp_img, CSV_TPATH + '-temp', CSV_NAME[0:-15] + 'temperature.png')
    save(humidity_img, CSV_TPATH  + '-humidity', CSV_NAME[0:-15] + 'humidity.png')
    save(dust_img, CSV_TPATH  + '-dust', CSV_NAME[0:-15] + 'dust.png')

    render = cv2.addWeighted(occ_img, 0.4, humidity_img, 0.6, 0.4)
    show_image(0, True, render)
    
