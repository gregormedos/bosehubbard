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
import numpy as np
from scipy import special
import h5py


def fock_lower(s_state: np.ndarray, i: int):
    """
    Return a copy of a Fock state with one lowered site.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    i : int
        Site index

    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.copy(s_state)
    t_state[i] -= 1
    return t_state


def fock_raise(s_state: np.ndarray, i: int):
    """
    Return a copy of a Fock state with one raised site.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    i : int
        Site index

    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.copy(s_state)
    t_state[i] += 1
    return t_state


def fock_translation(s_state: np.ndarray, r: int):
    """
    Return a copy of a Fock state, translated in the right direction by `r` number of sites.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    r : int
        Number of sites by which to move in the right direction
        
    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.roll(s_state, r)
    return t_state


def fock_checkstate(s_state: np.ndarray, num_sites: int, crystal_momentum: int):
    """
    Check if the given Fock state is the representative state under translations.
    If so, calculate the translation period of the representative state.
    If not, return the translation period as -1 to dismiss the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
    crystal_momentum : int
        Crystal momentum

    Returns
    -------
    translation_period : int
        Translation period of the representative state

    """
    translation_period = -1  # not representative
    t_state = np.copy(s_state)
    for i in range(1, num_sites + 1):
        t_state = np.roll(t_state, 1)
        if tuple(t_state) > tuple(s_state):
            return translation_period
        elif tuple(t_state) == tuple(s_state):
            if (crystal_momentum % (num_sites / i)) == 0:
                translation_period = i
                return translation_period
            else:
                return translation_period


def fock_representative(s_state: np.ndarray, num_sites: int):
    """
    Find the representative state under translations for the given Fock state and
    calculate the number of translations from the representative state to the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites

    Returns
    -------
    representative_state : np.ndarray
        Representative state
    num_translations : int
        Number of translations

    """
    representative_state = np.copy(s_state)
    num_translations = 0
    t_state = np.copy(s_state)
    for i in range(1, num_sites):
        t_state = np.roll(t_state, 1)
        if tuple(t_state) > tuple(representative_state):
            representative_state = np.copy(t_state)
            num_translations = i

    return representative_state, num_translations


def fock_reflection(s_state: np.ndarray):
    """
    Return a copy of a Fock state, reflected through its center.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
        
    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.flipud(s_state)
    return t_state


def fock_checkstate_reflection(s_state: np.ndarray, num_sites: int, crystal_momentum: int):
    """
    Check if the given Fock state is the representative state under translations and reflection.
    If so, calculate number of translations need to get from the reflection of the representative state
    back to the representative state.
    If not, return the translation period as -1 to dismiss the given Fock state.
    If the reflections are not connected under translations,
    return -1 to indicate that two copies of this representative state will be stored.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations and reflection.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
    crystal_momentum : int
        Crystal momentum

    Returns
    -------
    translation_period : int
        Translation period of the representative state
    num_translations_reflection : int
        Number of translations after reflection

    """
    translation_period = fock_checkstate(s_state, num_sites, crystal_momentum)
    num_translations_reflection = -1  # the reflections are not connected under translations
    t_state = np.flipud(s_state)
    for i in range(translation_period):
        if tuple(t_state) > tuple(s_state):
            translation_period = -1  # not representative
            return translation_period, num_translations_reflection
        elif tuple(t_state) == tuple(s_state):
            num_translations_reflection = i
            return translation_period, num_translations_reflection
        t_state = np.roll(t_state, 1)

    return translation_period, num_translations_reflection


def fock_representative_reflection(s_state: np.ndarray, num_sites: int):
    """
    Find the representative state for the reflection of the given Fock
    state and calculate the number of translations and reflections from
    the representative state to the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations and reflection.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
        
    Returns
    -------
    representative_state : np.ndarray
        Representative state
    num_translations : int
        Number of translations
    num_reflections : int
        Number of reflections

    """
    representative_state, num_translations = fock_representative(s_state, num_sites)
    num_reflections = 0
    t_state = np.flipud(s_state)
    for i in range(num_sites):
        if tuple(t_state) > tuple(representative_state):
            representative_state = np.copy(t_state)
            num_translations = i
            num_reflections = 1
        t_state = np.roll(t_state, 1)

    return representative_state, num_translations, num_reflections


def dim_full_hardcore(num_sites: int):
    """
    Calculate the full hardcore bosonic Hilbert space dimension,
    given the number of sites `num_sites`.

    Parameters
    ----------
    num_sites : int
        Number of sites

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    return 2 ** num_sites


def dim_full_softcore(num_sites: int, n_max: int):
    """
    Calculate the full softcore bosonic Hilbert space dimension,
    given the number of sites `num_sites` and the maximum site occupation number `n_max`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum site occupation number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    return (n_max + 1) ** num_sites


def gen_basis_full_hardcore(num_sites: int, dim: int):
    """
    Generate the full Hilbert space Fock basis, given the number of
    sites `num_sites` and the full Hilbert space dimension.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum number of bosons on site
    dim : int
        Hilbert space dimension
    
    Returns
    -------
    basis : np.ndarray
        Hilbert space Fock basis

    """
    if num_sites > 1:
        basis = np.empty((dim, num_sites), dtype=int)
        a = 0
        for n in range(2):
            sub_num_sites = num_sites - 1
            sub_dim = dim // 2
            basis[a: a + sub_dim, 0] = 1 - n
            basis[a: a + sub_dim, 1:] = gen_basis_full_hardcore(sub_num_sites, sub_dim)
            a += sub_dim
    elif num_sites == 1:
        basis = np.empty((dim, 1), dtype=int)
        for n in range(dim):
            basis[n] = 1 - n
    else:
        basis = None

    return basis


def gen_basis_full_softcore(num_sites: int, dim: int, n_max: int):
    """
    Generate the full Hilbert space Fock basis, given the number of
    sites `num_sites`, the restriction on the maximum number of bosons on site
    `n_max` and the full Hilbert space dimension.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum number of bosons on site
    dim : int
        Hilbert space dimension
    
    Returns
    -------
    basis : np.ndarray
        Hilbert space Fock basis

    """
    if num_sites > 1:
        basis = np.empty((dim, num_sites), dtype=int)
        a = 0
        for n in range(n_max + 1):
            sub_num_sites = num_sites - 1
            sub_dim = dim // (n_max + 1)
            basis[a: a + sub_dim, 0] = n_max - n
            basis[a: a + sub_dim, 1:] = gen_basis_full_softcore(sub_num_sites, sub_dim, n_max)
            a += sub_dim
    elif num_sites == 1:
        basis = np.empty((dim, 1), dtype=int)
        for n in range(dim):
            basis[n] = n_max - n
    else:
        basis = None

    return basis


def dim_nblock_hardcore(num_sites: int, n_tot: int):
    """
    Calculate the N-block hardcore bosonic Hilbert space dimension,
    given the number of sites `num_sites` and the total particle number `n_tot`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total particle number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    if num_sites < n_tot:
        return 0
    else:
        return special.comb(num_sites, n_tot, exact=True)


def dim_nblock_bosonic(num_sites: int, n_tot: int):
    """
    Calculate the N-block bosonic Hilbert space dimension,
    given the number of sites `num_sites` and the total particle number `n_tot`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total particle number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    if num_sites == 0:
        if n_tot == 0:
            return 1
        else:
            return 0
    else:
        return special.comb(num_sites + n_tot - 1, n_tot, exact=True)


def dim_nblock_softcore(num_sites: int, n_tot: int, n_max: int):
    """
    Calculate the N-block softcore bosonic Hilbert space dimension,
    given the number of sites `num_sites`, the total particle number `n_tot`
    and the maximum site occupation number `n_max`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total particle number
    n_max : int
        Maximum site occupation number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    if num_sites * n_max < n_tot:
        return 0
    elif num_sites == 0 and n_tot == 0:
        return 1
    else:
        dimension = 0
        for k in range(num_sites + 1):
            dimension += (
                (-1) ** k
                * special.comb(num_sites, k, exact=True)
                * special.comb(num_sites + n_tot - k * (n_max + 1) - 1, num_sites - 1, exact=True)
            )
    
    return dimension


def gen_basis_nblock_hardcore(num_sites: int, n_tot: int, dim: int):
    """
    Generate the N-block Hilbert space Fock basis, given the number of
    sites `num_sites`, the total number of hardcore bosons `n_tot`
    and the N-block Hilbert space dimension.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total number of bosons
    dim : int
        Hilbert space dimension
    
    Returns
    -------
    basis : np.ndarray
        Hilbert space Fock basis

    """
    if num_sites < n_tot:
        basis = None
    elif num_sites > 1:
        basis = np.empty((dim, num_sites), dtype=int)
        a = 0
        for n in range(n_tot - 1, n_tot + 1):
            sub_num_sites = num_sites - 1
            sub_dim = dim_nblock_hardcore(sub_num_sites, n)
            if sub_dim > 0:
                basis[a: a + sub_dim, 0] = n_tot - n
                basis[a: a + sub_dim, 1:] = gen_basis_nblock_hardcore(sub_num_sites, n, sub_dim)
                a += sub_dim
    elif num_sites == 1:
        basis = np.array([n_tot], dtype=int)
    else:
        basis = None

    return basis


def gen_basis_nblock_bosonic(num_sites: int, n_tot: int, dim: int):
    """
    Generate the N-block Hilbert space Fock basis, given the number of
    sites `num_sites`, the total number of bosons `n_tot` and the N-block Hilbert
    space dimension.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total number of bosons
    dim : int
        Hilbert space dimension
    
    Returns
    -------
    basis : np.ndarray
        Hilbert space Fock basis

    """
    if num_sites > 1:
        basis = np.empty((dim, num_sites), dtype=int)
        a = 0
        for n in range(n_tot + 1):
            sub_num_sites = num_sites - 1
            sub_dim = dim_nblock_bosonic(sub_num_sites, n)
            basis[a: a + sub_dim, 0] = n_tot - n
            basis[a: a + sub_dim, 1:] = gen_basis_nblock_bosonic(sub_num_sites, n, sub_dim)
            a += sub_dim
    elif num_sites == 1:
        basis = np.array([n_tot], dtype=int)
    else:
        basis = None

    return basis


def gen_basis_nblock_softcore(num_sites: int, n_tot: int, dim: int, n_max: int):
    """
    Generate the N-block Hilbert space Fock basis, given the number of
    sites `num_sites`, the total number of softcore bosons `n_tot`,
    the N-block Hilbert space dimension,
    and the restriction on the maximum number of bosons on site `n_max`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total number of bosons
    n_max : int
        Maximum number of bosons on site
    
    Returns
    -------
    basis : np.ndarray
        Hilbert space Fock basis
    dim : int
        Hilbert space dimension

    """
    if num_sites * n_max < n_tot:
        basis = None
    elif num_sites > 1:
        basis = np.empty((dim, num_sites), dtype=int)
        a = 0
        for n in range(n_tot - n_max, n_tot + 1):
            sub_num_sites = num_sites - 1
            sub_dim = dim_nblock_softcore(sub_num_sites, n, n_max)
            if sub_dim > 0:
                basis[a: a + sub_dim, 0] = n_tot - n
                basis[a: a + sub_dim, 1:] = gen_basis_nblock_softcore(sub_num_sites, n, sub_dim, n_max)
                a += sub_dim
    elif num_sites == 1:
        basis = np.array([n_tot], dtype=int)
    else:
        basis = None

    return basis


def dim_full(num_sites: int, n_max: int):
    if n_max == 1:
        return dim_full_hardcore(num_sites)
    elif n_max > 1:
        return dim_full_softcore(num_sites, n_max)
    else:
        raise ValueError("Argument n_max of dim_full must be greater than 0 but was given " + f"{n_max}")
    

def gen_basis_full(num_sites: int, dim: int, n_max: int):
    if n_max == 1:
        return gen_basis_full_hardcore(num_sites, dim)
    elif n_max > 1:
        return gen_basis_full_softcore(num_sites, dim, n_max)
    else:
        raise ValueError("Argument n_max of gen_basis_full must be greater than 0 but was given " + f"{n_max}")
    

def dim_nblock(num_sites: int, n_tot: int, n_max: int):
    if n_max == 1:
        return dim_nblock_hardcore(num_sites, n_tot)
    elif n_max >= n_tot:
        return dim_nblock_bosonic(num_sites, n_tot)
    elif n_max > 1:
        return dim_nblock_softcore(num_sites, n_tot, n_max)
    else:
        raise ValueError("Argument n_max of dim_nblock must be greater than 0 but was given " + f"{n_max}")
    

def gen_basis_nblock(num_sites: int, n_tot: int, dim: int, n_max: int):
    if n_max == 1:
        return gen_basis_nblock_hardcore(num_sites, n_tot, dim)
    elif n_max >= n_tot:
        return gen_basis_nblock_bosonic(num_sites, n_tot, dim)
    elif n_max > 1:
        return gen_basis_nblock_softcore(num_sites, n_tot, dim, n_max)
    else:
        raise ValueError("Argument n_max of gen_basis_nblock must be greater than 0 but was given " + f"{n_max}")


def gen_basis_nblock_from_full(super_basis: np.ndarray, n_tot: int):
    """
    Generate the N-block Hilbert space Fock basis, given the total
    number of bosons `n_tot` and the full Hilbert space Fock basis
    `super_basis`.

    Parameters
    ----------
    super_basis : np.ndarray
        Hilbert space Fock basis
    n_tot : int
        Total number of bosons
    
    Returns
    -------
    basis : np.ndarray
        Hilbert space Fock basis
    dim : int
        Hilbert space dimension

    """
    # we only want the pointers to the Fock states that belong to a
    # subspace with a good quantum number n_tot
    state_list = []
    for state_a in super_basis:
        if np.sum(state_a) == n_tot:
            state_list.append(state_a)  # intentionally avoiding copying
    basis = np.array(state_list, dtype=int)
    dim = basis.shape[0]

    return basis, dim


def gen_representative_basis_kblock(super_basis: np.ndarray, num_sites: int, crystal_momentum: int):
    """
    Generate the K-block Hilbert space representative Fock basis, given the number of sites `num_sites`,
    the crystal momentum `crystal_momentum` and the Hilbert space Fock basis `basis`.

    Parameters
    ----------
    super_basis : np.ndarray
        Hilbert space Fock basis
    num_sites : int
        Number of sites
    crystal_momentum : int
        Crystal momentum
    
    Returns
    -------
    representative_basis : np.ndarray
        Hilbert space representative Fock basis
    translation_periods : np.ndarray
        Translation periods of the representative states
    representative_dim : int
        Hilbert space representative dimension

    """
    # we only want the pointers to the Fock states that belong to a
    # subspace with a good quantum number crystal_momentum
    representative_state_list = []
    translation_period_list = []
    for state_a in super_basis:
        translation_period_a = fock_checkstate(state_a, num_sites, crystal_momentum)
        if translation_period_a > 0:
            representative_state_list.append(state_a)  # intentionally avoiding copying
            translation_period_list.append(translation_period_a)
    representative_basis = np.array(representative_state_list, dtype=int)
    translation_periods = np.array(translation_period_list, dtype=int)
    representative_dim = representative_basis.shape[0]

    return representative_basis, translation_periods, representative_dim


def gen_representative_basis_pkblock(
        super_basis: np.ndarray,
        num_sites: int,
        crystal_momentum: int,
        reflection_parity: int
):
    """
    Generate the PK-block Hilbert space representative Fock basis,
    given the number of sites `num_sites`,
    the crystal momentum `crystal_momentum`
    and the reflection parity `reflection_parity`.

    Parameters
    ----------
    super_basis : np.ndarray
        Hilbert space Fock basis
    num_sites : int
        Number of sites
    crystal_momentum : int
        Crystal momentum
    reflection_parity : int
        Reflection parity
    
    Returns
    -------
    representative_basis : np.ndarray
        Hilbert space representative Fock basis
    translation_periods : np.ndarray
        Translation periods of the representative states
    nums_translations_reflection : np.ndarray
        Number of translations after reflection
    representative_dim : int
        Hilbert space representative dimension

    """
    # we only want the pointers to the Fock states that belong to a
    # subspace with a good quantum number reflection_parity
    representative_state_list = []
    translation_period_list = []
    num_translations_reflection_list = []
    for state_a in super_basis:
        translation_period_a, num_translations_reflection_a = fock_checkstate_reflection(
            state_a,
            num_sites,
            crystal_momentum
        )
        if num_translations_reflection_a >= 0:
            if 1.0 + reflection_parity * np.cos(2.0 * np.pi / num_sites * crystal_momentum * num_translations_reflection_a) == 0.0:
                translation_period_a = -1
        if translation_period_a > 0:
            representative_state_list.append(state_a)  # intentionally avoiding copying
            translation_period_list.append(translation_period_a)
            num_translations_reflection_list.append(num_translations_reflection_a)
    representative_basis = np.array(representative_state_list, dtype=int)
    translation_periods = np.array(translation_period_list, dtype=int)
    nums_translations_reflection = np.array(num_translations_reflection_list, dtype=int)
    representative_dim = representative_basis.shape[0]

    return representative_basis, translation_periods, nums_translations_reflection, representative_dim


class HilbertSpace:
    """
    A HilbertSpace object represents a Hilbert space.

    At initialization a Fock basis is constructed for constructing
    operators in the Fock basis.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum number of bosons on site
    space : str, default='full'
        {'full', 'N', 'K', 'KN', 'PK', 'PKN'}
    n_tot : int, optional
        Total number of bosons
    crystal_momentum : int, optional
        Crystal momentum
    reflection_parity : int, optional
        Reflection parity
    
    """

    def __init__(
            self,
            num_sites: int,
            n_max: int,
            space: str = 'full',
            n_tot: int = None,
            crystal_momentum: int = None,
            reflection_parity: int = None,
    ):
        self.num_sites = num_sites
        self.n_max = n_max
        self.space = space
        self.basis = None
        self.findstate = None
        self.dim = None
        self.n_tot = n_tot
        self.crystal_momentum = crystal_momentum
        self.representative_basis = None
        self.representative_findstate = None
        self.translation_periods = None
        self.representative_dim = None
        self.reflection_parity = reflection_parity
        self.nums_translations_reflection = None

        if space == 'full':
            self.dim = dim_full(num_sites, n_max)
            self.basis = gen_basis_full(num_sites, self.dim, n_max)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
        
        elif space == 'N':
            self.dim = dim_nblock(num_sites, n_tot, n_max)
            self.basis = gen_basis_nblock(num_sites, n_tot, self.dim, n_max)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
            
        elif space == 'K':
            self.dim = dim_full(num_sites, n_max)
            self.basis = gen_basis_full(num_sites, self.dim, n_max)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
            (
                self.representative_basis,
                self.translation_periods,
                self.representative_dim
            ) = gen_representative_basis_kblock(self.basis, num_sites, crystal_momentum)
            self.representative_findstate = {}
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

        elif space == 'KN':
            self.dim = dim_nblock(num_sites, n_tot, n_max)
            self.basis = gen_basis_nblock(num_sites, n_tot, self.dim, n_max)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
            (
                self.representative_basis,
                self.translation_periods,
                self.representative_dim
            ) = gen_representative_basis_kblock(self.basis, num_sites, crystal_momentum)
            self.representative_findstate = {}
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

        elif space == 'PK' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
            self.dim = dim_full(num_sites, n_max)
            self.basis = gen_basis_full(num_sites, self.dim, n_max)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
            (
                self.representative_basis,
                self.translation_periods,
                self.nums_translations_reflection,
                self.representative_dim
            ) = gen_representative_basis_pkblock(
                self.basis,
                num_sites,
                crystal_momentum,
                reflection_parity
            )
            self.representative_findstate = {}
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

        elif space == 'PKN' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
            self.dim = dim_nblock(num_sites, n_tot, n_max)
            self.basis = gen_basis_nblock(num_sites, n_tot, self.dim, n_max)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
            (
                self.representative_basis,
                self.translation_periods,
                self.nums_translations_reflection,
                self.representative_dim
            ) = gen_representative_basis_pkblock(
                self.basis,
                num_sites,
                crystal_momentum,
                reflection_parity
            )
            self.representative_findstate = {}
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

    # Basis transformation
    def basis_transformation_n(self, mat: np.ndarray):
        change_of_basis_mat = np.zeros_like(mat)
        beginning_of_block = 0
        for n in range(self.num_sites * self.n_max + 1):
            basis_n, dim_n = gen_basis_nblock_from_full(self.basis, n)
            for a in range(dim_n):
                state_a = basis_n[a]
                change_of_basis_mat[self.findstate[tuple(state_a)], beginning_of_block + a] += 1.0
            beginning_of_block += dim_n

        return change_of_basis_mat

    def basis_transformation_k(self, mat: np.ndarray):
        if self.crystal_momentum == 0 or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
            dtype = float
        else:
            dtype = complex
        change_of_basis_mat = np.zeros(mat.shape, dtype=dtype)
        beginning_of_block = 0
        for k in range(self.num_sites):
            (
                representative_basis_k,
                translation_periods_k,
                representative_dim_k
            ) = gen_representative_basis_kblock(self.basis, self.num_sites, k)
            for a in range(representative_dim_k):
                representative_state_a = representative_basis_k[a]
                translation_period_a = translation_periods_k[a]
                normalization_a = np.sqrt(translation_period_a) / self.num_sites
                for r in range(self.num_sites):
                    phase_arg = -2.0 * np.pi / self.num_sites * k * r 
                    if (self.crystal_momentum == 0) or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
                        bloch_wave = np.cos(phase_arg)
                    else:
                        bloch_wave = np.exp(1.0j * phase_arg)
                    t_state_a = np.roll(representative_state_a, r)
                    change_of_basis_mat[
                        self.findstate[tuple(t_state_a)],
                        beginning_of_block + a
                    ] += normalization_a * bloch_wave
            beginning_of_block += representative_dim_k

        return change_of_basis_mat
    
    def basis_transformation_kn(self, mat: np.ndarray):
        if self.crystal_momentum == 0 or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
            dtype = float
        else:
            dtype = complex
        change_of_basis_mat = np.zeros(mat.shape, dtype=dtype)
        beginning_of_block = 0
        for n in range(self.num_sites * self.n_max + 1):
            basis_n, dim_n = gen_basis_nblock_from_full(self.basis, n)
            for k in range(self.num_sites):
                (
                    representative_basis_kn,
                    translation_periods_kn,
                    representative_dim_kn
                ) = gen_representative_basis_kblock(basis_n, self.num_sites, k)
                for a in range(representative_dim_kn):
                    representative_state_a = representative_basis_kn[a]
                    translation_period_a = translation_periods_kn[a]
                    normalization_a = np.sqrt(translation_period_a) / self.num_sites
                    for r in range(self.num_sites):
                        phase_arg = -2.0 * np.pi / self.num_sites * k * r 
                        if (self.crystal_momentum == 0) or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
                            bloch_wave = np.cos(phase_arg)
                        else:
                            bloch_wave = np.exp(1.0j * phase_arg)
                        t_state_a = np.roll(representative_state_a, r)
                        change_of_basis_mat[
                            self.findstate[tuple(t_state_a)],
                            beginning_of_block + a
                        ] += normalization_a * bloch_wave
                beginning_of_block += representative_dim_kn

        return change_of_basis_mat

    def basis_transformation_pk(self, mat: np.ndarray):
        change_of_basis_mat = np.zeros(mat.shape, dtype=float)
        beginning_of_block = 0
        for p in (1, -1):
            (
                representative_basis_pk,
                translation_periods_pk,
                nums_translations_reflection_pk,
                representative_dim_pk
            ) = gen_representative_basis_pkblock(
                self.basis,
                self.num_sites,
                self.crystal_momentum,
                p
            )
            for a in range(representative_dim_pk):
                representative_state_a = representative_basis_pk[a]
                num_translations_reflection_a = nums_translations_reflection_pk[a]
                if num_translations_reflection_a >= 0:
                    normalization_a = 1.0
                else:
                    normalization_a = np.sqrt(2.0) / 2.0
                change_of_basis_mat[
                    self.representative_findstate[tuple(representative_state_a)],
                    beginning_of_block + a
                ] += normalization_a
                if num_translations_reflection_a == -1:
                    r_state_a, num_translations_a = fock_representative(np.flipud(representative_state_a), self.num_sites)
                    change_of_basis_mat[
                        self.representative_findstate[tuple(r_state_a)],
                        beginning_of_block + a
                    ] += normalization_a * p * np.cos(2.0 * np.pi / self.num_sites * self.crystal_momentum * num_translations_a)
            beginning_of_block += representative_dim_pk

        return change_of_basis_mat

    # Coulomb interaction Hamiltonian
    def op_hamiltonian_interaction(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            mat[a, a] = 0.5 * np.sum(state_a * (state_a - 1))

        return mat

    # K-block Coulomb interaction Hamiltonian
    def op_hamiltonian_interaction_k(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=float)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            mat[a, a] = 0.5 * np.sum(representative_state_a * (representative_state_a - 1))

        return mat

    # quadratic operator
    def _op_quadratic(self, mat: np.ndarray, state_a: np.ndarray, a: int, i: int, d: tuple, r: tuple):
        n_i = state_a[i]
        t_state = np.copy(state_a)
        t_state[i] += r[0]
        if 0 <= t_state[i] <= self.n_max:
            for d_j in d:
                j = i + d_j
                j = j % self.num_sites  # PBC IF NEEDED
                n_j = t_state[j]
                state_b = np.copy(t_state)
                state_b[j] += r[1]
                if 0 <= state_b[j] <= self.n_max:
                    if tuple(state_b) in self.findstate:
                        b = self.findstate[tuple(state_b)]
                        mat[a, b] += np.sqrt((n_i + (r[0]+1)/2.0) * (n_j + (r[1]+1)/2.0))

    # tunneling Hamiltonian with OBC
    def op_hamiltonian_tunnel_obc(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            self._op_quadratic(mat, state_a, a, 0, (1,), (-1, 1))
            self._op_quadratic(mat, state_a, a,  self.num_sites - 1, (-1,), (-1, 1))
            for i in range(1, self.num_sites - 1):
                self._op_quadratic(mat, state_a, a, i, (1, -1), (-1, 1))

        return -mat

    # tunneling Hamiltonian with PBC
    def op_hamiltonian_tunnel_pbc(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            for i in range(self.num_sites):
                self._op_quadratic(mat, state_a, a, i, (1, -1), (-1, 1))

        return -mat

    # K-block quadratic operator
    def _op_quadratic_k(
            self,
            mat: np.ndarray,
            representative_state_a: np.ndarray,
            translation_period_a: int,
            a: int,
            i: int,
            d: tuple,
            r: tuple
    ):
        n_i = representative_state_a[i]
        t_state = np.copy(representative_state_a)
        t_state[i] += r[0]
        if 0 <= t_state[i] <= self.n_max:
            for d_j in d:
                j = i + d_j
                j = j % self.num_sites  # PBC IF NEEDED
                n_j = t_state[j]
                state_b = np.copy(t_state)
                state_b[j] += r[1]
                if 0 <= state_b[j] <= self.n_max:
                    representative_state_b, num_translations_b = fock_representative(state_b, self.num_sites)
                    if tuple(representative_state_b) in self.representative_findstate:
                        b = self.representative_findstate[tuple(representative_state_b)]
                        translation_period_b = self.translation_periods[b]
                        phase_arg = 2.0 * np.pi / self.num_sites * self.crystal_momentum * num_translations_b
                        if (self.crystal_momentum == 0) or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
                            bloch_wave = np.cos(phase_arg)
                        else:
                            bloch_wave = np.exp(1.0j * phase_arg)  # complex conjugated
                        mat[a, b] += np.sqrt(
                            (n_i + (r[0]+1)/2.0) * (n_j + (r[1]+1)/2.0) * translation_period_a / translation_period_b
                        ) * bloch_wave  # complex conjugated

    # K-block tunneling Hamiltonian
    def op_hamiltonian_tunnel_k(self):
        if self.crystal_momentum == 0 or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
            dtype = float
        else:
            dtype = complex
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=dtype)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            translation_period_a = self.translation_periods[a]
            for i in range(self.num_sites):
                self._op_quadratic_k(mat, representative_state_a, translation_period_a, a, i, (1, -1), (-1, 1))

        return -mat

    # PK-block quadratic operator
    def _op_quadratic_pk(
            self,
            mat: np.ndarray,
            representative_state_a: np.ndarray,
            translation_period_a: int,
            factor_a: float,
            a: int,
            i: int,
            d: tuple,
            r: tuple
    ):
        n_i = representative_state_a[i]
        t_state = np.copy(representative_state_a)
        t_state[i] += r[0]
        if 0 <= t_state[i] <= self.n_max:
            for d_j in d:
                j = i + d_j
                j = j % self.num_sites  # PBC IF NEEDED
                n_j = t_state[j]
                state_b = np.copy(t_state)
                state_b[j] += r[1]
                if 0 <= state_b[j] <= self.n_max:
                    (
                        representative_state_b,
                        num_translations_b,
                        num_reflections_b
                    ) = fock_representative_reflection(state_b, self.num_sites)
                    if tuple(representative_state_b) in self.representative_findstate:
                        b = self.representative_findstate[tuple(representative_state_b)]
                        translation_period_b = self.translation_periods[b]
                        num_translations_reflection_b = self.nums_translations_reflection[b]
                        if num_translations_reflection_b >= 0:
                            factor_b = 2.0
                        else:
                            factor_b = 1.0
                        phase_arg = 2.0 * np.pi / self.num_sites * self.crystal_momentum * num_translations_b
                        mat[a, b] += np.sqrt(
                            (n_i + (r[0]+1)/2.0) * (n_j + (r[1]+1)/2.0)
                            * translation_period_a * factor_b
                            / (translation_period_b * factor_a)
                        ) * self.reflection_parity ** num_reflections_b * np.cos(phase_arg)

    # PK-block tunneling Hamiltonian
    def op_hamiltonian_tunnel_pk(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=float)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            translation_period_a = self.translation_periods[a]
            num_translations_reflection_a = self.nums_translations_reflection[a]
            if num_translations_reflection_a >= 0:
                factor_a = 2.0
            else:
                factor_a = 1.0
            for i in range(self.num_sites):
                self._op_quadratic_pk(mat, representative_state_a, translation_period_a, factor_a, a, i, (1, -1), (-1, 1))

        return -mat

    # linear operator
    def _op_linear(self, mat: np.ndarray, state_a: np.ndarray, a: int, i: int, r: int):
        n_i = state_a[i]
        state_b = np.copy(state_a)
        state_b[i] += r
        if 0 <= state_b[i] <= self.n_max:
            if tuple(state_b) in self.findstate:
                b = self.findstate[tuple(state_b)]
                mat[a, b] += np.sqrt(n_i + (r+1)/2.0)

    # annihilation and creation Hamiltonian
    def op_hamiltonian_annihilate_create(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            for i in range(self.num_sites):
                self._op_linear(mat, state_a, a, i, -1)
                self._op_linear(mat, state_a, a, i, 1)

        return mat
    
    # pair annihilation and creation Hamiltonian with OBC
    def op_hamiltonian_annihilate_create_pair_obc(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            for i in range(self.num_sites - 1):
                self._op_quadratic(mat, state_a, a, i, (1,), (-1, -1))
                self._op_quadratic(mat, state_a, a, i, (1,), (1, 1))

        return mat
    
    # pair annihilation and creation Hamiltonian with PBC
    def op_hamiltonian_annihilate_create_pair_pbc(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            for i in range(self.num_sites):
                self._op_quadratic(mat, state_a, a, i, (1,), (-1, -1))
                self._op_quadratic(mat, state_a, a, i, (1,), (1, 1))

        return mat

    # K-block linear operator
    def _op_linear_k(
            self,
            mat: np.ndarray,
            representative_state_a: np.ndarray,
            translation_period_a: int,
            a: int,
            i: int,
            r: int
    ):
        n_i = representative_state_a[i]
        b_state = np.copy(representative_state_a)
        b_state[i] += r
        if 0 <= b_state[i] <= self.n_max:
            representative_state_b, num_translations_b = fock_representative(b_state, self.num_sites)
            if tuple(representative_state_b) in self.representative_findstate:
                b = self.representative_findstate[tuple(representative_state_b)]
                translation_period_b = self.translation_periods[b]
                phase_arg = 2.0 * np.pi / self.num_sites * self.crystal_momentum * num_translations_b
                if (self.crystal_momentum == 0) or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)  # complex conjugated
                mat[a, b] += np.sqrt(
                    (n_i + (r+1)/2.0) * translation_period_a / translation_period_b
                ) * bloch_wave  # complex conjugated

    # K-block annihilation and creation Hamiltonian
    def op_hamiltonian_annihilate_create_k(self):
        if self.crystal_momentum == 0 or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
            dtype = float
        else:
            dtype = complex
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=dtype)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            translation_period_a = self.translation_periods[a]
            for i in range(self.num_sites):
                self._op_linear_k(mat, representative_state_a, translation_period_a, a, i, -1)
                self._op_linear_k(mat, representative_state_a, translation_period_a, a, i, 1)

        return mat
    
    # K-block pair annihilation and creation Hamiltonian
    def op_hamiltonian_annihilate_create_pair_k(self):
        if self.crystal_momentum == 0 or (self.num_sites % 2 == 0 and self.crystal_momentum == self.num_sites // 2):
            dtype = float
        else:
            dtype = complex
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=dtype)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            translation_period_a = self.translation_periods[a]
            for i in range(self.num_sites):
                self._op_quadratic_k(mat, representative_state_a, translation_period_a, a, i, (1,), (-1, -1))
                self._op_quadratic_k(mat, representative_state_a, translation_period_a, a, i, (1,), (1, 1))

        return mat

    # PK-block linear operator
    def _op_linear_pk(
            self,
            mat: np.ndarray,
            representative_state_a: np.ndarray,
            translation_period_a: int,
            factor_a: float,
            a: int,
            i: int,
            r: int
    ):
        n_i = representative_state_a[i]
        state_b = np.copy(representative_state_a)
        state_b[i] += r
        if 0 <= state_b[i] <= self.n_max:
            (
                representative_state_b,
                num_translations_b,
                num_reflections_b
            ) = fock_representative_reflection(state_b, self.num_sites)
            if tuple(representative_state_b) in self.representative_findstate:
                b = self.representative_findstate[tuple(representative_state_b)]
                translation_period_b = self.translation_periods[b]
                num_translations_reflection_b = self.nums_translations_reflection[b]
                if num_translations_reflection_b >= 0:
                    factor_b = 2.0
                else:
                    factor_b = 1.0
                phase_arg = 2.0 * np.pi / self.num_sites * self.crystal_momentum * num_translations_b
                mat[a, b] += np.sqrt(
                    (n_i + (r+1)/2.0)
                    * translation_period_a * factor_b
                    / (translation_period_b * factor_a)
                ) * self.reflection_parity ** num_reflections_b * np.cos(phase_arg)
                        
    # PK-block annihilation and creation Hamiltonian
    def op_hamiltonian_annihilate_create_pk(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=float)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            translation_period_a = self.translation_periods[a]
            num_translations_reflection_a = self.nums_translations_reflection[a]
            if num_translations_reflection_a >= 0:
                factor_a = 2.0
            else:
                factor_a = 1.0
            for i in range(self.num_sites):
                self._op_linear_pk(mat, representative_state_a, translation_period_a, factor_a, a, i, -1)
                self._op_linear_pk(mat, representative_state_a, translation_period_a, factor_a, a, i, 1)

        return mat
    
    # PK-block pair annihilation and creation Hamiltonian
    def op_hamiltonian_annihilate_create_pair_pk(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=float)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            translation_period_a = self.translation_periods[a]
            num_translations_reflection_a = self.nums_translations_reflection[a]
            if num_translations_reflection_a >= 0:
                factor_a = 2.0
            else:
                factor_a = 1.0
            for i in range(self.num_sites):
                self._op_quadratic_pk(mat, representative_state_a, translation_period_a, factor_a, a, i, (1,), (-1, -1))
                self._op_quadratic_pk(mat, representative_state_a, translation_period_a, factor_a, a, i, (1,), (1, 1))

        return mat


class DecomposedHilbertSpace(HilbertSpace):
    """
    A DecomposedHilbertSpace object represents a decomposition of a Hilbert space possibly into smaller Hilbert spaces.

    At initialization a Fock basis is constructed for constructing operators in the Fock basis.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum number of bosons on site
    space : str, default='full'
        {'full', 'N', 'K', 'KN', 'PK', 'PKN'}
    sym : str, optional
        {'N', 'K', 'KN', 'PK', 'PKN'}
    n_tot : int, optional
        Total number of bosons
    crystal_momentum : int, optional
        Crystal momentum
    reflection_parity : int, optional
        Reflection parity
    super_basis : np.ndarray, optional
        Hilbert space Fock basis
    super_findstate : dict, optional
        Map from Fock basis state to quantum number
    super_dim : int, optional
        Hilbert space dimension
    
    """

    def __init__(
            self,
            num_sites: int,
            n_max: int,
            space: str = 'full',
            sym: str = None,
            n_tot: int = None,
            crystal_momentum: int = None,
            reflection_parity: int = None,
            super_basis: np.ndarray = None,
            super_findstate: dict = None,
            super_dim: int = None
    ):
        self.num_sites = num_sites
        self.n_max = n_max
        self.space = space
        self.sym = sym
        self.basis = None
        self.findstate = None
        self.dim = None
        self.subspaces = None
        self.n_tot = n_tot
        self.crystal_momentum = crystal_momentum
        self.representative_basis = None
        self.representative_findstate = None
        self.translation_periods = None
        self.representative_dim = None
        self.reflection_parity = reflection_parity
        self.nums_translations_reflection = None

        if super_basis is None:
            super().__init__(
                num_sites,
                n_max,
                space,
                n_tot,
                crystal_momentum,
                reflection_parity
            )
        
        elif space == 'N':
            self.basis, self.dim = gen_basis_nblock_from_full(super_basis, n_tot)
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
        
        elif space in {'K', 'KN'}:
            self.basis = super_basis
            self.findstate = super_findstate
            self.dim = super_dim
            (
                self.representative_basis,
                self.translation_periods,
                self.representative_dim
            ) = gen_representative_basis_kblock(self.basis, num_sites, crystal_momentum)
            self.representative_findstate = {}
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

        elif space in {'PK', 'PKN'} and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
            self.basis = super_basis
            self.findstate = super_findstate
            self.dim = super_dim
            (
                self.representative_basis,
                self.translation_periods,
                self.nums_translations_reflection,
                self.representative_dim
            ) = gen_representative_basis_pkblock(
                self.basis,
                num_sites,
                crystal_momentum,
                reflection_parity
            )
            self.representative_findstate = {}
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

        if space == 'full':
            if sym in {'N', 'KN', 'PKN'}:
                self.subspaces = []
                for n in range(num_sites * n_max + 1):
                    self.subspaces.append(
                        DecomposedHilbertSpace(
                            num_sites,
                            n_max,
                            'N',
                            sym,
                            n_tot=n,
                            super_basis=self.basis,  # intentionally avoiding copying
                            super_findstate=self.findstate,
                            super_dim=self.dim
                        )
                    )
            elif sym in {'K', 'PK'}:
                self.subspaces = []
                for k in range(num_sites):
                    self.subspaces.append(
                        DecomposedHilbertSpace(
                            num_sites,
                            n_max,
                            'K',
                            sym,
                            crystal_momentum=k,
                            super_basis=self.basis,  # intentionally avoiding copying
                            super_findstate=self.findstate,
                            super_dim=self.dim
                        )
                    )

        elif space == 'N':
            if sym in {'KN', 'PKN'}:
                self.subspaces = []
                for k in range(num_sites):
                    self.subspaces.append(
                        DecomposedHilbertSpace(
                            num_sites,
                            n_max,
                            'KN',
                            sym,
                            n_tot=n_tot,
                            crystal_momentum=k,
                            super_basis=self.basis,  # intentionally avoiding copying
                            super_findstate=self.findstate,
                            super_dim=self.dim
                        )
                    )

        elif space == 'K':
            if sym == 'PK' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
                self.subspaces = []
                for p in (1, -1):
                    self.subspaces.append(
                        DecomposedHilbertSpace(
                            num_sites,
                            n_max,
                            'PK',
                            sym,
                            crystal_momentum=crystal_momentum,
                            reflection_parity=p,
                            super_basis=self.basis,  # intentionally avoiding copying
                            super_findstate=self.findstate,
                            super_dim=self.dim
                        )
                    )

        elif space == 'KN':
            if sym == 'PKN' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
                self.subspaces = []
                for p in (1, -1):
                    self.subspaces.append(
                        DecomposedHilbertSpace(
                            num_sites,
                            n_max,
                            space='PKN',
                            sym=sym,
                            n_tot=n_tot,
                            crystal_momentum=crystal_momentum,
                            reflection_parity=p,
                            super_basis=self.basis,  # intentionally avoiding copying
                            super_findstate=self.findstate,
                            super_dim=self.dim
                        )
                    )


HAMILTONIAN_DICT = {
    'PK': {
        't': HilbertSpace.op_hamiltonian_tunnel_pk,
        'U': HilbertSpace.op_hamiltonian_interaction_k,
        'V1': HilbertSpace.op_hamiltonian_annihilate_create_pk,
        'V2': HilbertSpace.op_hamiltonian_annihilate_create_pair_pk
    },
    'K': {
        't': HilbertSpace.op_hamiltonian_tunnel_k,
        'U': HilbertSpace.op_hamiltonian_interaction_k,
        'V1': HilbertSpace.op_hamiltonian_annihilate_create_k,
        'V2': HilbertSpace.op_hamiltonian_annihilate_create_pair_k
    },
    'None': {
        't': HilbertSpace.op_hamiltonian_tunnel_pbc,
        'U': HilbertSpace.op_hamiltonian_interaction,
        'V1': HilbertSpace.op_hamiltonian_annihilate_create,
        'V2': HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc
    }
}


def run_decomposed(
        dir_name: str,
        file_name: str,
        L: int,
        M: int,
        terms: tuple,
        space: str = 'full',
        sym: str = None,
        N: int = None,
        K: int = None,
        P: int = None
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = DecomposedHilbertSpace(num_sites=L, n_max=M, space=space, sym=sym, n_tot=N, crystal_momentum=K, reflection_parity=P)
        print(hs.dim)
        exact_diagonalization_decomposed(file, group, hs, terms)


def run(
        dir_name: str,
        file_name: str,
        L: int,
        M: int,
        terms: tuple,
        space: str = 'full',
        N: int = None,
        K: int = None,
        P: int = None
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = HilbertSpace(num_sites=L, n_max=M, space=space, n_tot=N, crystal_momentum=K, reflection_parity=P)
        print(hs.dim)
        exact_diagonalization(file, group, hs, terms)


def read_eigen_energies_decomposed(
        dir_name: str,
        file_name: str
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        if 'representative_dim' in group['param']:
            representative_dim = group['param/representative_dim']
            eigen_energies = np.empty(representative_dim, dtype=float)
        else:
            eigen_energies = np.empty(dim, dtype=float)
        counter = [0]
        get_eigen_energies_decomposed(group, eigen_energies, counter)
        return eigen_energies
    

def read_eigen_states_decomposed(
        dir_name: str,
        file_name: str
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        basis = group['basis'][()]
        findstate = dict()
        if 'representative_dim' in group['param']:
            representative_dim = group['param/representative_dim']
            for a in range(representative_dim):
                findstate[tuple(basis[a])] = a
            eigen_states = np.empty((dim, representative_dim), dtype=complex)
        else:
            for a in range(dim):
                findstate[tuple(basis[a])] = a
            eigen_states = np.empty((dim, dim), dtype=complex)
        counter = [0]
        get_eigen_states_decomposed(group, eigen_states, findstate, counter, dim)
        return eigen_states


def read_mid_spectrum_eigen_energies_decomposed(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_energies = []
        dim_sectors = []
        get_mid_spectrum_eigen_energies_decomposed(group, eigen_energies, dim_sectors, num)
        return eigen_energies, dim_sectors


def read_mid_spectrum_eigen_states_decomposed(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        dim = group['dim'][()]
        basis = group['basis'][()]
        findstate = dict()
        for a in range(dim):
            findstate[tuple(basis[a])] = a
        eigen_states = []
        dim_sectors = []
        get_mid_spectrum_eigen_states_decomposed(group, eigen_states, dim_sectors, findstate, dim, num)
        return eigen_states, dim_sectors


def read_eigen_energies(
        dir_name: str,
        file_name: str      
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_energies = get_eigen_energies(group)
        return eigen_energies


def read_eigen_states(
        dir_name: str,
        file_name: str
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_states = get_eigen_states(group)
        return eigen_states


def read_mid_spectrum_eigen_energies(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_energies = get_mid_spectrum_eigen_energies(group, num)
        return eigen_energies


def read_mid_spectrum_eigen_states(
        dir_name: str,
        file_name: str,
        num: int
):
    with h5py.File(f'{dir_name}{file_name}.h5', 'r') as file:
        group = file['data']
        eigen_states = get_mid_spectrum_eigen_states(group, num)
        return eigen_states


def exact_diagonalization_decomposed(
        file: h5py.File,
        group: h5py.Group,
        hs: DecomposedHilbertSpace,
        terms: tuple
):
    param = group.create_group('param')
    param.create_dataset('num_sites', data=hs.num_sites)
    param.create_dataset('n_max', data=hs.n_max)
    param.create_dataset('space', data=hs.space)
    group.create_dataset('dim', data=hs.dim)
    group.create_dataset('basis', data=hs.basis)
    if hs.sym is not None:
        param.create_dataset('sym', data=hs.sym)
    if hs.n_tot is not None:
        param.create_dataset('n_tot', data=hs.n_tot)
    if hs.crystal_momentum is not None:
        param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        group.create_dataset('representative_basis', data=hs.representative_basis)
        group.create_dataset('translation_periods', data=hs.translation_periods)
        group.create_dataset('representative_dim', data=hs.representative_dim)
    if hs.reflection_parity is not None:
        param.create_dataset('reflection_parity', data=hs.reflection_parity)
        group.create_dataset('num_translations_reflection', data=hs.nums_translations_reflection)
    file.flush()
    if hs.subspaces is not None:
        group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            if subspace.dim > 0:
                exact_diagonalization_decomposed(
                    file,
                    group.create_group(f'subspaces/{i:04d}'),
                    subspace,
                    terms
                )
    else:
        spectrum = group.create_group('spectrum')
        if hs.space in {'PK', 'PKN'}:
            hamiltonians = HAMILTONIAN_DICT['PK']
        elif hs.space in ('K', 'KN'):
            hamiltonians = HAMILTONIAN_DICT['K']
        else:
            hamiltonians = HAMILTONIAN_DICT['None']
        hamiltonian = np.sum([rate * hamiltonians[term](hs) for term, rate in terms], axis=0)
        print(hamiltonian.shape)
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def exact_diagonalization(
        file: h5py.File,
        group: h5py.Group,
        hs: HilbertSpace,
        terms: tuple
):
    param = group.create_group('param')
    param.create_dataset('num_sites', data=hs.num_sites)
    param.create_dataset('n_max', data=hs.n_max)
    param.create_dataset('space', data=hs.space)
    group.create_dataset('dim', data=hs.dim)
    group.create_dataset('basis', data=hs.basis)
    if hs.n_tot is not None:
        param.create_dataset('n_tot', data=hs.n_tot)
    if hs.crystal_momentum is not None:
        param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        group.create_dataset('representative_basis', data=hs.representative_basis)
        group.create_dataset('translation_periods', data=hs.translation_periods)
        group.create_dataset('representative_dim', data=hs.representative_dim)
    if hs.reflection_parity is not None:
        param.create_dataset('reflection_parity', data=hs.reflection_parity)
        group.create_dataset('num_translations_reflection', data=hs.nums_translations_reflection)
    file.flush()
    spectrum = group.create_group('spectrum')
    if hs.space in {'PK', 'PKN'}:
        hamiltonians = HAMILTONIAN_DICT['PK']
    elif hs.space in ('K', 'KN'):
        hamiltonians = HAMILTONIAN_DICT['K']
    else:
        hamiltonians = HAMILTONIAN_DICT['None']
    hamiltonian = np.sum([rate * hamiltonians[term](hs) for term, rate in terms], axis=0)
    print(hamiltonian.shape)
    eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
    spectrum.create_dataset('eigen_energies', data=eigen_energies)
    spectrum.create_dataset('eigen_states', data=eigen_states)
    file.flush()


def get_eigen_energies_decomposed(group: h5py.Group, eigen_energies: np.ndarray, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_energies_decomposed(group[f'subspaces/{subspace_name}'], eigen_energies, counter)
    else:
        j = counter[0]
        subspace = group['param/space'][()].decode()
        if 'K' in subspace:
            representative_dim = group['representative_dim'][()]
            counter[0] += representative_dim
            eigen_energies_sector = group['spectrum/eigen_energies'][()]
            eigen_energies[j: j + representative_dim] = eigen_energies_sector
        else:
            dim_sector = group['dim'][()]
            counter[0] += dim_sector
            eigen_energies_sector = group['spectrum/eigen_energies'][()]
            eigen_energies[j: j + dim_sector] = eigen_energies_sector


def get_eigen_states_decomposed(group: h5py.Group, eigen_states: np.ndarray, findstate: dict, counter: list, dim: int):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_states_decomposed(group[f'subspaces/{subspace_name}'], eigen_states, findstate, counter, dim)
    else:
        j = counter[0]
        subspace = group['param/space'][()].decode()
        if subspace in {'KN', 'K'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            representative_basis = group['representative_basis'][()]
            translation_periods = group['translation_periods'][()]
            representative_dim = group['representative_dim'][()]
            counter[0] += representative_dim
            representative_eigen_states = group['spectrum/eigen_states'][()]
            eigen_states_sector = np.zeros((dim, representative_dim), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                amplitudes_a = representative_eigen_states[a]
                normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
            eigen_states[:, j: j + representative_dim] = eigen_states_sector
        elif subspace in {'PKN', 'PK'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            reflection_parity = group['param/reflection_parity'][()]
            representative_basis = group['representative_basis'][()]
            translation_periods = group['translation_periods'][()]
            num_translations_reflection = group['num_translations_reflection'][()]
            representative_dim = group['representative_dim'][()]
            counter[0] += representative_dim
            representative_eigen_states = group['spectrum/eigen_states'][()]
            eigen_states_sector = np.zeros((dim, representative_dim), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                num_translations_reflection_a = num_translations_reflection[a]
                amplitudes_a = representative_eigen_states[a]
                if num_translations_reflection_a == -1:
                    normalization_a = np.sqrt(2.0 * translation_period_a) / (2.0 * num_sites)
                else:
                    normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                    if num_translations_reflection_a == -1:
                        t_representative_state_a = np.flipud(t_representative_state_a)
                        eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a
            eigen_states[:, j: j + representative_dim] = eigen_states_sector
        else:
            dim_sector = group['dim'][()]
            counter[0] += dim_sector
            eigen_states_sector = group['spectrum/eigen_states'][()]
            eigen_states[:, j: j + dim_sector] = eigen_states_sector


def get_mid_spectrum_eigen_energies_decomposed(group: h5py.Group, eigen_energies: list, dim_sectors: list, num: int):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_mid_spectrum_eigen_energies_decomposed(group[f'subspaces/{subspace_name}'], eigen_energies, dim_sectors, num)
    else:
        subspace = group['param/space'][()].decode()
        if 'K' in subspace:
            representative_dim = group['representative_dim'][()]
            eigen_energies_sector = group['spectrum/eigen_energies'][(representative_dim - num) // 2: (representative_dim + num) // 2]
            eigen_energies.append(eigen_energies_sector)
            dim_sectors.append(representative_dim)
        else:
            dim_sector = group['dim'][()]
            eigen_energies_sector = group['spectrum/eigen_energies'][(dim_sector - num) // 2: (dim_sector + num) // 2]
            eigen_energies.append(eigen_energies_sector)
            dim_sectors.append(dim_sector)


def get_mid_spectrum_eigen_states_decomposed(group: h5py.Group, eigen_states: list, dim_sectors: list, findstate: dict, dim: int, num: int):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_mid_spectrum_eigen_states_decomposed(group[f'subspaces/{subspace_name}'], eigen_states, dim_sectors, findstate, dim, num)
    else:
        subspace = group['param/space'][()].decode()
        if subspace in {'KN', 'K'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            representative_basis = group['representative_basis'][()]
            translation_periods = group['translation_periods'][()]
            representative_dim = group['representative_dim'][()]
            representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
            eigen_states_sector = np.zeros((dim, num), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                amplitudes_a = representative_eigen_states[a]
                normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
            eigen_states.append(eigen_states_sector)
            dim_sectors.append(representative_dim)
        elif subspace in {'PKN', 'PK'}:
            num_sites = group['param/num_sites'][()]
            crystal_momentum = group['param/crystal_momentum'][()]
            reflection_parity = group['param/reflection_parity'][()]
            representative_basis = group['representative_basis'][()]
            translation_periods = group['translation_periods'][()]
            num_translations_reflection = group['num_translations_reflection'][()]
            representative_dim = group['representative_dim'][()]
            representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
            eigen_states_sector = np.zeros((dim, num), dtype=complex)
            for a in range(representative_dim):
                representative_state_a = representative_basis[a]
                translation_period_a = translation_periods[a]
                num_translations_reflection_a = num_translations_reflection[a]
                amplitudes_a = representative_eigen_states[a]
                if num_translations_reflection_a == -1:
                    normalization_a = np.sqrt(2.0 * translation_period_a) / (2.0 * num_sites)
                else:
                    normalization_a = np.sqrt(translation_period_a) / num_sites
                for r in range(num_sites):
                    phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                    bloch_wave = np.exp(1.0j * phase_arg)
                    t_representative_state_a = np.roll(representative_state_a, r)
                    eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                    if num_translations_reflection_a == -1:
                        t_representative_state_a = np.flipud(t_representative_state_a)
                        eigen_states_sector[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a
            eigen_states.append(eigen_states_sector)
            dim_sectors.append(representative_dim)
        else:
            dim_sector = group['dim'][()]
            eigen_states_sector = group['spectrum/eigen_states'][:, (dim_sector - num) // 2: (dim_sector + num) // 2]
            eigen_states.append(eigen_states_sector)
            dim_sectors.append(dim_sector)


def get_eigen_energies(group: h5py.Group):
    eigen_energies = group['spectrum/eigen_energies'][()]
    return eigen_energies


def get_eigen_states(group: h5py.Group):
    space = group['param/space'][()].decode()
    dim = group['dim'][()]
    basis = group['basis'][()]
    findstate = dict()
    for a in range(dim):
        findstate[tuple(basis[a])] = a
    if space in {'KN', 'K'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        representative_basis = group['representative_basis'][()]
        translation_periods = group['translation_periods'][()]
        representative_dim = group['representative_dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][()]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, representative_dim), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
    elif space in {'PKN', 'PK'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        reflection_parity = group['param/reflection_parity'][()]
        representative_basis = group['representative_basis'][()]
        translation_periods = group['translation_periods'][()]
        num_translations_reflection = group['num_translations_reflection'][()]
        representative_dim = group['representative_dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][()]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, representative_dim), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            num_translations_reflection_a = num_translations_reflection[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            if num_translations_reflection_a == -1:
                normalization_a /= np.sqrt(2.0)
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                if num_translations_reflection_a == -1:
                    t_representative_state_a = np.flipud(t_representative_state_a)
                    eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a
    else:
        eigen_states = group['spectrum/eigen_states'][()]
    return eigen_states


def get_mid_spectrum_eigen_energies(group: h5py.Group, num: int):
    space = group['param/space'][()].decode()
    if 'K' in space:
        representative_dim = group['representative_dim'][()]
        eigen_energies = group['spectrum/eigen_energies'][(representative_dim - num) // 2: (representative_dim + num) // 2]
    else:
        dim = group['dim'][()]
        eigen_energies = group['spectrum/eigen_energies'][(dim - num) // 2: (dim + num) // 2]
    return eigen_energies


def get_mid_spectrum_eigen_states(group: h5py.Group, num: int):
    space = group['param/space'][()].decode()
    dim = group['dim'][()]
    basis = group['basis'][()]
    findstate = dict()
    for a in range(dim):
        findstate[tuple(basis[a])] = a
    if space in {'KN', 'K'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        representative_basis = group['representative_basis'][()]
        translation_periods = group['translation_periods'][()]
        representative_dim = group['representative_dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, num), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
    elif space in {'PKN', 'PK'}:
        num_sites = group['param/num_sites'][()]
        crystal_momentum = group['param/crystal_momentum'][()]
        reflection_parity = group['param/reflection_parity'][()]
        representative_basis = group['representative_basis'][()]
        translation_periods = group['translation_periods'][()]
        num_translations_reflection = group['num_translations_reflection'][()]
        representative_dim = group['representative_dim'][()]
        representative_eigen_states = group['spectrum/eigen_states'][:, (representative_dim - num) // 2: (representative_dim + num) // 2]
        if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
            dtype = float
        else:
            dtype = complex
        eigen_states = np.zeros((dim, num), dtype=dtype)
        for a in range(representative_dim):
            representative_state_a = representative_basis[a]
            translation_period_a = translation_periods[a]
            num_translations_reflection_a = num_translations_reflection[a]
            amplitudes_a = representative_eigen_states[a]
            normalization_a = np.sqrt(translation_period_a) / num_sites
            if num_translations_reflection_a == -1:
                normalization_a /= np.sqrt(2.0)
            for r in range(num_sites):
                phase_arg = -2.0 * np.pi / num_sites * crystal_momentum * r
                if (crystal_momentum == 0) or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2):
                    bloch_wave = np.cos(phase_arg)
                else:
                    bloch_wave = np.exp(1.0j * phase_arg)
                t_representative_state_a = np.roll(representative_state_a, r)
                eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * bloch_wave * amplitudes_a
                if num_translations_reflection_a == -1:
                    t_representative_state_a = np.flipud(t_representative_state_a)
                    eigen_states[findstate[tuple(t_representative_state_a)], :] += normalization_a * reflection_parity * bloch_wave * amplitudes_a

    else:
        eigen_states = group['spectrum/eigen_states'][:, (dim - num) // 2: (dim + num) // 2]
    return eigen_states
