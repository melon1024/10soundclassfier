import numpy as np

for i in range(10):
    data = np.load(str(i)+".npz")
    print(data['X'].shape)
