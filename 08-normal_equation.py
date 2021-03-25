
"""
Normal Equation Noninvertibility, Andrew NG course, Week 2, Standford university
"""

import numpy as np
from numpy.linalg import pinv  # pseudo-inverse

np.set_printoptions(precision=2, suppress=True)

X = np.array([[1, 2104, 5, 1, 45],
              [1, 1416, 3, 2, 40],
              [1, 1534, 3, 2, 30],
              [1,  852, 2, 1, 36]])

print(X)
y = np.array([[460.],
              [232.],
              [315.],
              [178.]])

print("")
x_plus = pinv(X.T @ X) @ X.T @ y
x_plus = np.linalg.pinv(X)
print("The x+ matrix is: ") 
print(x_plus)
theta = pinv(X) @ y
theta = x_plus.dot(y)
print("")
print("The theta value is ")
print(theta)

#  theorem : If matrix 𝐴 is **non-singular**, then 𝐴−1= 𝐴

print(X.dot(theta) == y)             # Wrong way to campare to float matrices, the computer percision is playing role here
print(np.allclose(X.dot(theta), y))  # correct way for comparing numpy matrices
