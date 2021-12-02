import pandas as pd
import torch
from sktensor.tucker import hooi, hosvd
from sktensor import dtensor

from numpy.linalg import pinv, matrix_rank
from numpy import transpose, diag

A = torch.load('A3.pt')

print(A[4,4,4,4,9])
print(A[4,4,4,4,9])
print(A[4,4,4,4,9])
print(A[4,4,4,4,9])
print(A[4,4,4,4,9])
print(A[4,4,4,4,9])

input('')

print('===== ... now performing HOSVD')

shape_original = A.shape
Ulist, S = hosvd(dtensor(A), rank=shape_original)
shape_original_coretensor = S.shape

print('===== ... now performing HOOI')

Sh, Uhlist = hooi(dtensor(A), [10,10,10,1,10], init='nvecs')

print('Shapes of HOOI core tensor and weighting matrices:')
print(Sh.shape)
print(Uhlist[0].shape)
print(Uhlist[1].shape)
print(Uhlist[2].shape)
print(Uhlist[3].shape)
print(Uhlist[4].shape)

print('---')

A_reconstructed = dtensor(Sh).ttm(Uhlist[0], 0).ttm(Uhlist[1], 1).ttm(Uhlist[2], 2).ttm(Uhlist[3], 3).ttm(Uhlist[4], 4)

gradations = [10,10,10,20,10]

reconstruction_error = 0.0
denom = 10*10*10*20*10

## for f in range(gradations[0]):
## 	for a in range(gradations[1]):
## 		for c in range(gradations[2]):
## 			for v in range(gradations[3]):
## 				for t in range(gradations[4]):
## 					reconstruction_error += ((float(A[f,a,c,v,t] - A_reconstructed[f,a,c,v,t])**2))
## 
## print('reconstruction_error: %d' % reconstruction_error)
print('mean reconstruction_error: %r' % repr(0.038845)) #7769

print('=====')

U4firstcol = Uhlist[3][:,0].reshape(-1,1)

dim_1_R = U4firstcol.shape[0]
dim_2_R = shape_original_coretensor[3]
print('7.) Matrix R, used to augment U4 with size %d-by-%d:' % (dim_1_R, dim_2_R))
R = torch.rand((dim_1_R,dim_2_R), dtype=torch.float64)
print(R)
print('... rank of R is needs to be %d!! - it is in fact: %d' % (dim_2_R, matrix_rank(R)))

Uh4 = torch.cat(
	(torch.tensor(U4firstcol, dtype=torch.float64),
	R), 1)
print('8.) shape of augmented U4: %r' % repr(Uh4.shape))
print('... it has rank of %r' % matrix_rank(Uh4))
#print('Augmented U1:')
#print(Uh1)


Sh_firstrow = torch.tensor(Sh[:,:,:,0,:]).unsqueeze(3)
nrows_u3 = Uhlist[4].shape[0]

print('9.) First "row" of HOOI core tensor has shape: %r' % repr(Sh_firstrow.shape))
print('... need to augment this with %d more elements' % nrows_u3)

sh_augmentation_shape = list(Sh_firstrow.shape)
sh_augmentation_shape[-1] = nrows_u3

Sh_firstrow_augmented = torch.cat((Sh_firstrow, torch.rand(tuple(sh_augmentation_shape), dtype=torch.float64)), len(sh_augmentation_shape)-1)

print('shape of Sh_firstrow_augmented: %r' % repr(Sh_firstrow_augmented.shape))

Uh5 = torch.cat(
	(torch.tensor(Uhlist[4], dtype=torch.float64),
	torch.eye(nrows_u3, dtype=torch.float64)
	), 1)
print('10.) shape of augmented U5: %r' % repr(Uh5.shape))
print('... it has rank of %r' % matrix_rank(Uh5))
##print('Augmented U5:')
##print(Uh5)



D = (dtensor(A).ttm(pinv(Uh5),4).ttm(pinv(Uhlist[0]),0).ttm(pinv(Uhlist[1]),1).ttm(pinv(Uhlist[2]),2) -
  dtensor(Sh_firstrow_augmented).ttm(U4firstcol, 3)).ttm(pinv(R), 3)

print('11.) Computing D.... shape of D is %r' % repr(D.shape))

#S_reconstructed = torch.cat((Sh_firstrow, torch.tensor(D, dtype=torch.float64)), 0)
S_reconstructed = torch.cat((Sh_firstrow_augmented, torch.tensor(D, dtype=torch.float64)), 3)

print('12.) Shape of reconstructed core tensor: %r' % repr(S_reconstructed.shape))

print('13.) Reconstruction of original tensor:')
A_reconstructed2 = dtensor(S_reconstructed).ttm(Uhlist[0], 0).ttm(Uhlist[1], 1).ttm(Uhlist[2],2).ttm(Uh4.numpy(),3).ttm(Uh5.numpy(), 4)
	#ttm(Uhlist[2],2))

reconstruction_error = 0.0

for f in range(gradations[0]):
	for a in range(gradations[1]):
		for c in range(gradations[2]):
			for v in range(gradations[3]):
				for t in range(gradations[4]):
					reconstruction_error += ((float(A[f,a,c,v,t] - A_reconstructed2[f,a,c,v,t])**2))

print('reconstruction_error: %d' % reconstruction_error)

print('shape of Uh4: %r' % repr(Uh4.shape))

#for cycle in range(10):
#	dtensor(S_reconstructed).ttm(Uhlist[0], 0).ttm(Uhlist[1], 1).ttm(Uhlist[2],2).ttm(Uh4.numpy(),3).ttm(Uh5.numpy(), 4)

for i in range(10):
	Uh4[8,0] = Uh4[8,0] + 0.0001
	A_reconstructed2 = dtensor(S_reconstructed).ttm(Uhlist[0], 0).ttm(Uhlist[1], 1).ttm(Uhlist[2],2).ttm(Uh4.numpy(),3).ttm(Uh5.numpy(), 4)
	print(A_reconstructed2[3,2,4,8,8])



