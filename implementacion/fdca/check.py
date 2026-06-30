import numpy as np
# fm = np.load("data/20250926_1900/fire_mask.npy")
# vals, counts = np.unique(fm, return_counts=True)
# for v, c in zip(vals, counts):
#     print(v, c)

fm = np.load("data/20250926_1900/fire_mask.npy")
vals, counts = np.unique(fm, return_counts=True)
for v, c in zip(vals, counts): print(v, c)