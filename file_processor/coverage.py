# FYI: exitCode=1 in javascript is probably a syntax error.
from map_maker import MapMaker
import sys, json, traceback
import time
import bridge
import util


# /var/folders/yl/31853pjx2lj80f8z5cqsbdww0000gn/T/upload_d041886e18e780a07ed751a46ee8183a
coverage_physical_path = sys.argv[1]

# 2016-10-08-01-01-01_pilot01_6166_occupancy.pgm
coverage_alias = sys.argv[2]

try:

  maker = MapMaker(coverage_physical_path)

  maker.grey_to_white()
  maker.flip()
  maker.scale(3)

  util.save(maker.getImg(), coverage_physical_path, coverage_alias, 'coverage')

except:
  e_type, value, tb = sys.exc_info()
  stack = ''.join(traceback.format_tb(tb))
  bridge.error('Unhandled exception" %s\nDescription: %s\nStack: %s' % (e_type, value, stack))