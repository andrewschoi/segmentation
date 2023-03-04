import numpy as np

def weight(img, i1, j1, i2, j2):
  """
  img: grayscale img? (how do to handle color image?)
  (i1, j1): first coordinate
  (i2, j2): second coordinate
  Returns output of weight function specified in hw
  """
  if i1 == i2 and j1 == j2:
    return 0
  
  if abs(i1 - i2) <= 20 and abs(j1 - j2) <= 20:
    c2 = img[i1][j1]
    c1 = img[i2][j2] #how do we get color difference of color image?
    return np.exp(-100 * abs(c1 - c2) ** 2)
  
  return 0

def create_W(img):
  """
  img: grayscale img? (how do to handle color image?)
  """
  m, n, _ = img.shape
  result = np.zeros((m, n))

  for i1 in range(m):
    for j1 in range(n):
      for i2 in range(m):
        for j2 in range(n):
          result[i1, j1] = weight(img, i1, j1, i2, j2)
  
  return result



def graph_based_segmentation(img):
  """
  img: a hxwx3 numpy array with floating point values
  Returns the second-smallest absolute eigenvalue 
  """
  pass