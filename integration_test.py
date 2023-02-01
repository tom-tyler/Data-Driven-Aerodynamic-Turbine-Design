import numpy as np

rovx = np.array([[1,200,300],
                 [7,6,8],
                 [5,6,9]])

r = np.array([0.2,0.3,0.4])
rt = np.array([2,5,8])

integral = np.trapz(rovx,r,axis=0)

print(integral)