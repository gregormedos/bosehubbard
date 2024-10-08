import numpy as np
from .dim import *
from .fock import *


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


def gen_basis_full(num_sites: int, dim: int, n_max: int):
    if n_max == 1:
        return gen_basis_full_hardcore(num_sites, dim)
    elif n_max > 1:
        return gen_basis_full_softcore(num_sites, dim, n_max)
    else:
        raise ValueError("Argument n_max of gen_basis_full must be greater than 0 but was given " + f"{n_max}")


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
