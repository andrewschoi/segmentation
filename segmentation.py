import numpy as np

def weight(img, p, q):
  """
  weight function specified in hw
  """
  i1, j1 = p
  i2, j2 = q

  if i1 == i2 and j1 == j2:
    return 0
  
  if abs(i1 - i2) <= 20 and abs(j1 - j2) <= 20:
    c2 = img[i1][j1]
    c1 = img[i2][j2] #how do we get color difference of color image?
    return np.exp(-100 * abs(c1 - c2) ** 2)
  
  return 0

def create_W(img):
  m, n, _ = img.shape
  result = np.zeros((m, n))



def graph_based_segmentation(img):
  """
  img: a hxwx3 numpy array with floating point values
  Returns the second-smallest absolute eigenvalue 
  """
  pass