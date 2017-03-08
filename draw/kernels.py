import numpy as np

class KERNEL:
  SM_KERNEL_SIZE = 5
  SM_KERNEL = np.ones((SM_KERNEL_SIZE, SM_KERNEL_SIZE), np.uint8)
  LG_KERNEL_SIZE = 20
  LG_KERNEL = np.ones((LG_KERNEL_SIZE, LG_KERNEL_SIZE), np.uint8)

  def size(n):
    return np.ones((n, n), np.uint8)