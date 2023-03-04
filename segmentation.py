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
  Returns W matrix as specified in hw
  """
  m, n, _ = img.shape
  result = np.zeros((m, n))

  for i1 in range(m):
    for j1 in range(n):
      for i2 in range(m):
        for j2 in range(n):
          result[i1, j1] = weight(img, i1, j1, i2, j2)
  
  return result

def create_D(img):
  """
  img: grayscale img? (how to handle color image?)
  Returns D matrix as specified in hw
  """
  m, n, _ = img.shape
  result = np.zeros((m, n))

  W = create_W(img) #create W matrix

  for i1 in range(m):
    for j1 in range(n):
      if i1 == j1:
        result[i1, j1] = np.sum(W[i1])
  
  return result

def create_A(img):
  """
  img: gray-scale img
  Returns matrix A as specified in hw
  """
  m, n, _ = img.shape

  I = np.identity(m)
  D = create_d(img)
  W = create_w(img)

  D_inverse = D.copy()

  coords = np.where(W != 0)
  for i, j in coords:
    D_inverse = 1 / [D_inverse[i, j]] #inverse of a diagonal matrix is its reciprocal of its diagonals
  
  return I - np.matmul(D_inverse, W) #find A as specified in hw
  




def graph_based_segmentation(img):
  """
  img: a hxwx3 numpy array with floating point values
  Returns the second-smallest absolute eigenvalue 
  """
  pass