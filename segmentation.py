import numpy as np

def dist(v1, v2):
  """
  v1, v2: vectors, numpy arrays
  Returns distance between v1 and v2
  """
  result = 0
  for e1, e2 in zip(v1, v2):
    result += (e1 - e2) ** 2
  return np.sqrt(result)

def weight(img, i1, j1, i2, j2):
  """
  img: color img
  (i1, j1): first coordinate
  (i2, j2): second coordinate
  Returns output of weight function specified in hw
  """
  if i1 == i2 and j1 == j2:
    return 0
  
  if abs(i1 - i2) <= 20 and abs(j1 - j2) <= 20:
    c1 = img[i1][j1]
    c2 = img[i2][j2]
    return np.exp(-100 * dist(c1, c2) ** 2)
  
  return 0

def create_W(img):
  """
  img: color img
  Returns W matrix as specified in hw
  """
  m, n, _ = img.shape
  result = np.zeros((m * n, m * n))

  for i1 in range(m * n):
    for j1 in range(m * n):
      for i2 in range(m * n):
        for j2 in range(m * n):
          result[i1, j1] = weight(img, i1, j1, i2, j2)
  
  return result

def create_D(img):
  """
  img: color img
  Returns D matrix as specified in hw
  """
  m, n, _ = img.shape
  result = np.zeros((m * n, m * n))

  W = create_W(img) #create W matrix

  for i1 in range(m * n):
    for j1 in range(m * n):
      if i1 == j1:
        result[i1, j1] = np.sum(W[i1])
  
  return result

def create_A(img):
  """
  img: color img
  Returns matrix A as specified in hw
  """
  m, n, _ = img.shape

  I = np.identity(m)
  D = create_D(img)
  W = create_W(img)

  D_inverse = D.copy()

  coords = np.where(W != 0)
  for i, j in coords:
    D_inverse = 1 / D_inverse[i, j] #inverse of a diagonal matrix is its reciprocal of its diagonals
  
  return I - np.matmul(D_inverse, W) #find A as specified in hw


def vectorize(img):
  """
  img: grayscale image
  Returns 1-d vector as specified in hw
  """
  m, n, _ = img.shape

  result = np.zeros(m * n)

  for i in range(m):
    for j in range(n):
      result[i * n + j] = img[i, j]
  return result



def graph_based_segmentation(img):
  """
  img: a hxwx3 numpy array with floating point values
  Returns the second-smallest absolute eigenvector reshaped into dim of original
  image 
  """
  m, n, _ = img.shape
  A = create_A(img)
  eigenvalues = np.linalg.eigvals(A)
  eigenvalues.sort()

  result = vectorize(img) * eigenvalues[1] #eigenvector associated with second smallest eigenvalue

  return result.reshape((m, n))
