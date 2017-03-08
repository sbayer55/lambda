class ClusterBin:
  def __init__(self, start, stop):
    """A range-like object to define a bin

    Args:
      start (int): Bin = [start, stop)
      stop (int): Bin = [start, stop)
    """
    self.start = start
    self.stop = stop
  def asRange(self):
    """Returns self as a range

    Returns:
      range: range(self.start, self.stop)
    """
    return range(self.start, self.stop)
  def getFrom(self, array):
    """Applies selection of bin to ndarray

    Args:
      array (array_like): ndarray (*)

    Return:
      array_like: ndarray[self.start:self.stop]
    """
    return array.take(self.asRange(), 
      axis = 0, 
      mode='wrap')

  def add(self, bin):
    """New bin that will encapsulate this, other bin and everything inbetween

    Args:
      bin (ClusterBin): Bin to add to this

    Returns:
      ClusterBin: Spanning this, bin and between
    """
    start = min(self.start, bin.start)
    stop = max(self.stop, bin.stop)
    return ClusterBin(start, stop)