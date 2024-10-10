"""
BoseHubbard
===========

Content
-------
- Transformations on Fock states
- Fock basis generators
- HilbertSpace class with methods for constructing operators in its Fock basis
- DecomposedHilbertSpace class

Usage
-----
Uses NumPy for linear algebra.  `numpy` is imported as `np`.
Uses SciPy for combinatorics.  `special` is imported from `scipy`.
Uses h5py for HDF5 file format.
A Fock state is represented with an `np.ndarray` with the shape `(num_sites,)`,
where `num_sites` is the number of sites.
A state in the Fock basis is represented with an `np.ndarray` with the
shape `(dim,)`, where `dim` is the Hilbert space dimension.
A Fock basis is represented with an `np.ndarray` with the shape
`(dim, num_sites)`, where `num_sites` is the number of sites.
An operator in the Fock basis is represented with an `np.ndarray` with
the shape `(dim, dim)`.

"""
from bosehubbard.dim import *
from bosehubbard.fock import *
from bosehubbard.basis import *
from bosehubbard.model import *
from bosehubbard.spectrum import *
