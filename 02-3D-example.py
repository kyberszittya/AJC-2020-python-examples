import pandas as pd
import torch
from sktensor.tucker import hooi, hosvd
from sktensor import dtensor

from numpy.linalg import pinv, matrix_rank
from numpy import transpose, diag

print('===============')

print('Example in 3 dimensions:')

x = torch.tensor([
	[[5,2], [3,3], [1,5]],
	[[8,9], [2,102], [10,5]],
	[[3,92], [3,4], [8,-1]],
	[[1,2], [5,3], [2,97]],], dtype=torch.float64)


shape_original = x.shape
print('1.) Original tensor:')
print(x)
print('Original X has shape %r, and is:' % repr(shape_original))

print('===== ... now performing HOSVD')

Ulist, S = hosvd(dtensor(x), rank=shape_original)

print('2.) reconstruction of original tensor based on hosvd (check that is equal to 1.):')
reconstructed_x_from_hosvd = dtensor(S).ttm(Ulist[0], 0).ttm(Ulist[1], 1).ttm(Ulist[2], 2)
print(reconstructed_x_from_hosvd)
#
print('3.) Core tensor from HOSVD:')
print(S)
shape_original_coretensor = S.shape

print('4.) Reconstrcuted core tensor using pseudo-inverses of weighting matrices (must be equal to 3.)! ):')
reconstructed_core_tensor = reconstructed_x_from_hosvd.ttm(pinv(Ulist[0]), 0).ttm(pinv(Ulist[1]), 1).ttm(pinv(Ulist[2]), 2)
print(reconstructed_core_tensor)


print('===== ... now performing HOOI')

Sh, Uhlist = hooi(dtensor(x), [1,3,2], init='nvecs')

print('reconstruction of original tensor based on hooi (obviously different from 1.) and 2.):')
print(dtensor(Sh).
	ttm(Uhlist[0], 0).
	ttm(Uhlist[1], 1).
	ttm(Uhlist[2], 2))

print('shape of HOOI core tensor:')
print(Sh.shape)

print('shape of HOOI U1: %r' % repr(Uhlist[0].shape))

print('shape of HOOI U2: %r' % repr(Uhlist[1].shape))

print('shape of HOOI U3: %r' % repr(Uhlist[2].shape))

nrows_u3 = Uhlist[2].shape[0]
print('number of rows in U3 are %d' % nrows_u3)

print('ranks of each are: %r, %r, %r' % 
	(matrix_rank(Uhlist[0]), matrix_rank(Uhlist[1]), matrix_rank(Uhlist[2])))

U1firstcol = Uhlist[0][:,0].reshape(-1,1)
print('5.) HOOI U1:')
print(Uhlist[0])
print('6.) first column of HOOI U1 (compare with 5.))')
print(U1firstcol)
print('... First column of U1 has shape %r:' % repr(U1firstcol.shape))

print('===== ... now doing augmentations')

dim_1_R = U1firstcol.shape[0]
dim_2_R = shape_original_coretensor[0]
print('7.) Matrix R, used to augment U1 with size %d-by-%d:' % (dim_1_R, dim_2_R))
R = torch.rand((dim_1_R,dim_2_R), dtype=torch.float64)
print(R)
print('... rank of R is needs to be %d!! - it is in fact: %d' % (dim_2_R, matrix_rank(R)))

Uh1 = torch.cat(
	(torch.tensor(U1firstcol, dtype=torch.float64),
	R), 1)
print('8.) shape of augmented U1: %r' % repr(Uh1.shape))
print('... it has rank of %r' % matrix_rank(Uh1))
print('Augmented U1:')
print(Uh1)


Sh_firstrow = torch.tensor(Sh[0,:,:]).unsqueeze(0)

print('9.) First "row" of HOOI core tensor has shape: %r' % repr(Sh_firstrow.shape))
print('... need to augment this with %d more elements' % nrows_u3)

sh_augmentation_shape = list(Sh_firstrow.shape)
sh_augmentation_shape[-1] = 2

Sh_firstrow_augmented = torch.cat((Sh_firstrow, torch.rand(tuple(sh_augmentation_shape), dtype=torch.float64)), len(sh_augmentation_shape)-1)

print('shape of Sh_firstrow_augmented: %r' % repr(Sh_firstrow_augmented.shape))

Uh3 = torch.cat(
	(torch.tensor(Uhlist[2], dtype=torch.float64),
	torch.eye(2, dtype=torch.float64)
	), 1)
print('10.) shape of augmented U3: %r' % repr(Uh3.shape))
print('... it has rank of %r' % matrix_rank(Uh3))
print('Augmented U3:')
print(Uh3)



#D = (dtensor(x).ttm(pinv(Uhlist[2]),2).ttm(pinv(Uhlist[1]),1) -
#  dtensor(Sh_firstrow).ttm(U1firstcol, 0)).ttm(pinv(R), 0)

D = (dtensor(x).ttm(pinv(Uh3),2).ttm(pinv(Uhlist[1]),1) -
  dtensor(Sh_firstrow_augmented).ttm(U1firstcol, 0)).ttm(pinv(R), 0)

print('11.) Computing D.... shape of D is %r' % repr(D.shape))

#S_reconstructed = torch.cat((Sh_firstrow, torch.tensor(D, dtype=torch.float64)), 0)
S_reconstructed = torch.cat((Sh_firstrow_augmented, torch.tensor(D, dtype=torch.float64)), 0)

print('12.) Shape of reconstructed core tensor: %r' % repr(S_reconstructed.shape))

#print(dtensor(Sh).ttm(Uhlist[0], 0).ttm(Uhlist[1], 1).ttm(Uhlist[2], 2))


print('13.) Reconstruction of original tensor:')
print(dtensor(S_reconstructed).
	ttm(Uh1.numpy(), 0).
	ttm(Uhlist[1], 1).
	ttm(Uh3.numpy(),2))
	#ttm(Uhlist[2],2))

print('14.) Core tensor')
print(S_reconstructed)

print(Uh1)

print(Uhlist[1])

print(Uh3)

