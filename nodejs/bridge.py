import sys, json, time, os
from virtual_fs.virtual_file_list import VirtualFileList
from virtual_fs.virtual_file import VirtualFile

def log(message):
  if 'PYTHON_ENV' in os.environ and os.environ['PYTHON_ENV'] == 'development':
    print(message)
  else:
    with open("python.log", "wb") as log:
      log.write("Info: " + message)
    results = {
      'msg': message,
      'time': time.time()
    }
    print(json.dumps(results))

def error(message):
  if 'PYTHON_ENV' in os.environ and os.environ['PYTHON_ENV'] == 'development':
    print('\033[31mError: %s\033[0m' % message)
  else:
    with open("python.log", "wb") as log:
      log.write("Error: " + message)
    results = {
      'error': message,
      'time': time.time()
    }
    print(json.dumps(results))

def fileMade(path, alias):
  with open("python.log", "wb") as log:
    log.write("Output file: %s as %s" % (alias, path))
  results = {
    'file': {
      'alias': alias,
      'path': path
    },
    'time': time.time()
  }
  print(json.dumps(results))

def args_to_virtual_files():
  # Example:
  # sys.argv = [
  #   'arg_parser.py', 
  #   '/tmp/some_file', 
  #   '2017-01-09-12-34-04_pilot09_2014_coverage.pgm'
  # ]
  args = sys.argv

  # Remove script path:
  args.pop(0)

  # Check for error conditions:
  if len(args) == 0:
    raise ValueError('No args provided to arg_parser!')
  if len(args) % 2 != 0:
    raise ValueError('Args must be provided in sets of 3! (physical_path, s3_key)')
  
  # Store results here:
  list = VirtualFileList()

  # Loop through sys.argv and push to results:
  while len(args) > 0:
    physical_path = args.pop(0)
    s3_key = args.pop(0)
    v_file = VirtualFile(physical_path, s3_key)
    list.push(v_file)

  return list