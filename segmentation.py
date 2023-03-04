import numpy as np

def weight(p, q):
  i1, j1 = p
  i2, j2 = q

  if i1 == i2 and j1 == j2:
    return 0
  
  if abs(i1 - i2) <= 20 and abs(j1 - j2) <= 20:
    pass
  
  return 0

def create_W(img):
  m, n, _ = img.shape
  result = np.zeros((m, n))



def graph_based_segmentation(img):
  """
  img: a hxwx3 numpy array with floating point values
  Returns the second-smallest absolute eigenvalue 
  """
  