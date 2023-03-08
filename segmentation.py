import numpy as np

def weight(img, coord1, coord2):
  """
  img: color img
  i, j: coordinates of m * n array
  Returns output of weight function specified in hw
  """
  i1, j1 = coord1
  i2, j2 = coord2

  if i1 == i2 and j1 == j2:
    return 0

  if abs(i1 - i2) <= 20 and abs(j1 - j2) <= 20:
    c1 = img[i1][j1]
    c2 = img[i2][j2]
    dist = np.linalg.norm(c1 - c2) ** 2
    return np.exp(-100 * dist)
  
  return 0

def create_W(img):
  """
  img: color img
  Returns W matrix as specified in hw
  """
  m, n, _ = img.shape
  result = np.zeros((m * n, m * n))

  for i in range(n):
    for j in range(m):
      p = i * n + j
      for k in range(n):
          for l in range(m):
            q = k * n + l
            w = weight(img, i, j, k, l)
            result[p, q] = w
  
  
  return result


def graph_based_segmentation(img):
  """
  img: a hxwx3 numpy array with floating point values
  Returns the second-smallest absolute eigenvector reshaped into dim of original
  image 
  """
  m, n, _ = img.shape
  W = create_W(img)
  D = np.diag(np.sum(W, axis=1))
  D_inverse = np.linalg.inv(D)
  I = np.identity(m * n)
  A = I - (D_inverse @ W)
  vals, vecs = np.linalg.eig(A)

  sec_index = np.argsort(vals, 1)
  sec_evec = vecs[:, sec_index]
  return np.reshape(sec_evec, (m, n))
