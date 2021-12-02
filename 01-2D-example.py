import pandas as pd
from hosvd import hosvd
import torch
from sktensor.tucker import hooi
from sktensor import dtensor

from numpy.linalg import pinv, matrix_rank
from numpy import transpose, diag

print('Example in 2 dimensions:')

x = torch.tensor([[5,2], [3,3], [10,5]], dtype=torch.float64)

print('Original X is:')
print(x)

U = torch.tensor([[0.41, 0.1, 0.2, 0.35], [0.31,0.4, 0.5, 0.6],
	[0.86,0.7,0.8,0.9]], dtype=torch.float64).numpy()
##V = torch.tensor([[0.89, -0.46, 1, 0], [0.46,0.89, 0, 1]], dtype=torch.float64).numpy()
V = torch.tensor([[0.89, -0.46], [0.46,0.89]], dtype=torch.float64).numpy()

myU, myS, myV = torch.svd(x)

print(myU)

print(myS)

print(myV)

print('Reconstruction of original tensor based on hosvd:')
print(dtensor(diag(myS)).ttm(myU.numpy(), 0).ttm(transpose(myV.numpy(),(0,1)), 1))


U1 = torch.tensor([[0.41], [0.31],[0.86]], dtype=torch.float64).numpy()
#S1 = torch.tensor([[13.0, 0, 0.1, 0.2]], dtype=torch.float64).numpy()
S1 = torch.tensor([[13.0, 0]], dtype=torch.float64).numpy()

R = torch.tensor([[0.1, 0.2, 0.35], [0.4, 0.5, 0.6],
	[0.7,0.8,0.9]], dtype=torch.float64).numpy()

D = (dtensor(x).ttm(pinv(V),1) - 
	dtensor(S1).ttm(U1, 0)).ttm(pinv(R),0)

S = torch.tensor([[13.0, 0],
	[D[0,0], D[0,1]],
	[D[1,0], D[1,1]],
	[D[2,0], D[2,1]]], dtype=torch.float64).numpy()


print('Rank of R must be 3! Rank is: %d' % matrix_rank(R))

print('Reconstruction of original tensor:')
print(dtensor(S).ttm(U, 0).ttm(V, 1))

