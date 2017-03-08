class VirtualFileList:
  """VirtualFileList is a wrapper for a list of VirtualFile(s)

  VirtualFileList is available to make managing VirtualFile(s) convenient.

  Args:\n
    \t None
  """

  def __init__(self):
    self._virtual_files = []

  def push(self, virtual_file):
    if len(self._virtual_files) == 0:
      self.first = virtual_file
    self._virtual_files.append(virtual_file)

  def get(self, file_type):
    for file in self._virtual_files:
      if file.file_type is file_type:
        return file
    return None