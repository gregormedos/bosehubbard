import numpy as np
import bosehubbard as bh

hs = bh.HilbertSpace(5, 2)
print(hs.basis)
print(hs.dim)
hs = bh.HilbertSpace(5, 2, space='K', crystal_momentum=0)
print(hs.basis)
print(hs.dim)
print(hs.representative_basis)
print(hs.representative_dim)
