import numpy as np
import torch

tab = np.array([0, 1, 2, 3, 1, 234, 535, 1, 3, 0])

print(np.where(tab == 0)[0])

# tab1 = np.zeros(shape=(3, 3))
# tab2 = np.ones(shape=(4, 2))

# print(Q := np.zeros(tab1.shape + (len(tab2),)), "\n", Q.shape)

# print(torch.cuda.is_available())



# x = torch.tensor(np.array([1, 2, 3, 4, 5, 1])).unsqueeze(dim=0)
# print(x[0, 3])
