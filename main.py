import sys, json, traceback, time, os
from nodejs import bridge
from file_processor import occupancy

def main():
  list = bridge.args_to_virtual_files()

  if list.first.file_type == 'occupancy':
    occupancy.run(list.first, list)
  else:
    bridge.error('Unable to process %s.' % list.first.base)

if __name__ == '__main__':
  if 'PYTHON_ENV' in os.environ and os.environ['PYTHON_ENV'] == 'development':
    print('\n\n\033[35mRunning in \'development\'\033[0m')
    main()
  else:
    try:
      main()
    except:
      e_type, value, tb = sys.exc_info()
      stack = ''.join(traceback.format_tb(tb))
      bridge.error('Unhandled exception" %s\nDescription: %s\nStack: %s' % 
                   (e_type, value, stack))
else:
  bridge.error('Module occupancy __name__ != __main__')