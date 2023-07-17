import numpy as np


image_1 = np.array([[ 1,  2,  3,  4],
                    [ 5,  6,  7,  8],
                    [ 9, 10, 11, 12],
                    [13, 14, 15, 16]]).astype(np.float32)

filtr_1 = np.array([[1, 2],
                    [3, 4]]).astype(np.float32)

image_2 = np.array([[ 1,  2,  3,  4,  1,  2],
                    [ 5,  6,  7,  8,  5,  6],
                    [ 9, 10, 11, 12,  9, 10],
                    [13, 14, 15, 16, 13, 14]]).astype(np.float32)

filtr_2 = np.array([[1, 2, 3],
                    [4, 5, 6]]).astype(np.float32)

image = image_2; filtr = filtr_2

print(image)

for i in range(image.shape[0]-filtr.shape[0]+1):
  for j in range(image.shape[1]-filtr.shape[1]+1):
    print(image[i:i+filtr.shape[0], j:j+filtr.shape[1]].flatten())
