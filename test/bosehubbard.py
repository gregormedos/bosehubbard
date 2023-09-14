"""
BoseHubbard
===========

Content
-------
- Operators on Fock states
- Fock basis generators
- HilbertSpace class with methods for constructing operators in its
  Fock basis

Usage
-----
Uses NumPy for linear algebra.  `numpy` is imported as `np`.
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
    Check if the given Fock state is the representative state and
    calculate the period of the representative state.

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
    representative_period : int
        Period of the representative state

    """
    representative_period = -1  # not representative
    t_state = s_state
    for i in range(1, num_sites + 1):
        t_state = fock_translation(t_state, 1)
        if tuple(t_state) > tuple(s_state):
            return representative_period
        elif tuple(t_state) == tuple(s_state):
            if (crystal_momentum % (num_sites / i)) == 0:
                representative_period = i
                return representative_period
            else:
                return representative_period


def fock_representative(s_state: np.ndarray, num_sites: int):
    """
    Find the representative state for the given Fock state and
    calculate the number of translations from the representative
    state to the given Fock state.

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
    r_state : np.ndarray
        Representative state
    phase : int
        Number of translations

    """
    r_state = np.copy(s_state)
    t_state = s_state
    phase = 0
    for i in range(1, num_sites):
        t_state = fock_translation(t_state, 1)
        if tuple(t_state) > tuple(r_state):
            r_state = np.copy(t_state)
            phase = i

    return r_state, phase


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


def fock_checkstate_reflection(s_state: np.ndarray, representative_period: int):
    """
    Check if the reflection of the given Fock state is the
    representative state and calculate the reflection period of the
    representative state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations and reflections.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    representative_period : int
        Period of the representative state
    Returns
    -------
    representative_period : int
        Period of the representative state
    representative_reflection_period : int
        Reflection period of the representative state

    """
    representative_reflection_period = -1  # not representative
    t_state = fock_reflection(s_state)
    for i in range(representative_period):
        if tuple(t_state) > tuple(s_state):
            representative_period = -1  # not representative
            return representative_period, representative_reflection_period
        elif tuple(t_state) == tuple(s_state):
            representative_reflection_period = i
            return representative_period, representative_reflection_period
        t_state = fock_translation(t_state, 1)

    return representative_period, representative_reflection_period


def fock_representative_reflection(s_state: np.ndarray, num_sites: int, phase: int):
    """
    Find the representative state for the reflection of the given Fock
    state and calculate the number of translations and reflections from
    the representative state to the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations and reflections.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
    phase : int
        Number of translations
        
    Returns
    -------
    r_state : np.ndarray
        Representative state
    phase : int
        Number of translations
    reflection_phase : int
        Number of reflections

    """
    r_state = np.copy(s_state)
    t_state = fock_reflection(s_state)
    reflection_phase = 0
    for i in range(1, num_sites):
        t_state = fock_translation(t_state, 1)
        if tuple(t_state) > tuple(r_state):
            r_state = np.copy(t_state)
            phase = i
            reflection_phase = 1

    return r_state, phase, reflection_phase


def dim_full(num_sites: int, n_max: int):
    """
    Calculate the full Hilbert space dimension, given the number of
    sites `num_sites` and the restriction on the maximum number of bosons on
    site `n_max`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum number of bosons on site

    Returns
    -------
    dim : int
        Hilbert space dimension
    
    """
    return (n_max + 1) ** num_sites


def gen_basis_full(num_sites: int, n_max: int, dim: int):
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
            basis[a: a + sub_dim, 1:] = gen_basis_full(sub_num_sites, n_max, sub_dim)
            a += sub_dim
    else:
        basis = np.empty((dim, num_sites), dtype=int)
        for n in range(dim):
            basis[n] = n_max - n

    return basis


def dim_nblock(num_sites: int, n_tot: int):
    """
    Calculate the N-block Hilbert space dimension, given the number of
    sites `num_sites` and the total number of bosons `n_tot`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total number of bosons

    Returns
    -------
    dim : int
        Hilbert space dimension
    
    """
    return (np.math.factorial(num_sites + n_tot - 1) // np.math.factorial(num_sites - 1)
            // np.math.factorial(n_tot))


def gen_basis_nblock(num_sites: int, n_tot: int, dim: int):
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
            sub_dim = dim_nblock(sub_num_sites, n)
            basis[a: a + sub_dim, 0] = n_tot - n
            basis[a: a + sub_dim, 1:] = gen_basis_nblock(sub_num_sites, n, sub_dim)
            a += sub_dim
    else:
        basis = np.array([n_tot], dtype=int)

    return basis


def gen_basis_nblock_nmax(num_sites: int, n_tot: int, n_max: int):
    """
    Generate the N-block Hilbert space Fock basis, given the number of
    sites `num_sites`, the total number of bosons `n_tot` and the restriction on
    the maximum number of bosons on site `n_max`.

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
    dim : int
        Hilbert space dimension
    basis : np.ndarray
        Hilbert space Fock basis

    """
    if n_tot <= n_max:
        dim = dim_nblock(num_sites, n_tot)
        basis = gen_basis_nblock(num_sites, n_tot, dim)
    elif num_sites > 1:
        n_min = n_tot - n_max
        sub_dim_list = list()
        basis_block_list = list()
        basis_block_list_len = n_max + 1
        for n in range(n_min, n_tot + 1):
            sub_dim, sub_basis = gen_basis_nblock_nmax(num_sites - 1, n, n_max)
            if sub_basis is None:
                basis_block_list_len -= 1
            else:
                basis_block = np.empty((sub_dim, num_sites), dtype=int)
                basis_block[:, 0] = n_tot - n
                basis_block[:, 1:] = sub_basis
                sub_dim_list.append(sub_dim)
                basis_block_list.append(basis_block)
        dim = np.sum(sub_dim_list, dtype=int)
        basis = np.empty((dim, num_sites), dtype=int)
        a = 0
        for i in range(basis_block_list_len):
            basis[a: a + sub_dim_list[i], :] = basis_block_list[i]
            a += sub_dim_list[i]
    else:
        dim = None
        basis = None

    return dim, basis


def gen_basis_nblock_from_full(n_tot: int, super_basis: np.ndarray):
    """
    Generate the N-block Hilbert space Fock basis, given the total
    number of bosons `n_tot` and the full Hilbert space Fock basis
    `super_basis`.

    Parameters
    ----------
    n_tot : int
        Total number of bosons
    super_basis : np.ndarray
        Hilbert space Fock basis
    
    Returns
    -------
    dim : int
        Hilbert space dimension
    basis : np.ndarray
        Hilbert space Fock basis

    """
    # we only want the pointers to the Fock states that belong to a
    # subspace with a good quantum number n_tot
    state_list = list()
    for state_a in super_basis:
        if np.sum(state_a) == n_tot:
            state_list.append(state_a)  # intentionally avoiding copying
    basis = np.array(state_list, dtype=int)
    dim = basis.shape[0]

    return dim, basis


def gen_basis_knblock(num_sites: int,
                      n_tot: int,
                      n_max: int,
                      crystal_momentum: int,
                      super_basis: np.ndarray = None):
    """
    Generate the KN-block Hilbert space Fock basis, given the number of
    sites `num_sites`, the total number of bosons `n_tot`, the restriction on
    the maximum number of bosons on site `n_max`
    and the crystal momentum `crystal_momentum`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total number of bosons
    n_max : int
        Maximum number of bosons on site
    crystal_momentum : int
        Crystal momentum
    super_basis : np.ndarray, optional
        Hilbert space Fock basis
    
    Returns
    -------
    representative_dim : int
        Hilbert space dimension
    representative_basis : np.ndarray
        Hilbert space Fock basis
    representative_periods : np.ndarray
        Periods of the representative states

    """
    if super_basis is None:
        super_basis = gen_basis_nblock_nmax(num_sites, n_tot, n_max)[1]

    representative_state_list = list()
    representative_period_list = list()
    for state_a in super_basis:
        period = fock_checkstate(state_a, num_sites, crystal_momentum)
        if period >= 0:
            representative_state_list.append(state_a)  # intentionally avoiding copying
            representative_period_list.append(period)
    representative_basis = np.array(representative_state_list, dtype=int)
    representative_periods = np.array(representative_period_list, dtype=int)
    representative_dim = representative_basis.shape[0]

    return representative_dim, representative_basis, representative_periods


def gen_basis_pknblock(num_sites: int,
                       n_tot: int,
                       n_max: int,
                       crystal_momentum: int,
                       reflection_parity: int,
                       super_representative_dim: int = None,
                       super_representative_basis: np.ndarray = None,
                       super_representative_periods: int = None):
    """
    Generate the PKN-block Hilbert space Fock basis, given the number of
    sites `num_sites`, the total number of bosons `n_tot`, the restriction on
    the maximum number of bosons on site `n_max`,
    the crystal momentum `crystal_momentum`
    and the reflection parity `reflection_parity`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total number of bosons
    n_max : int, optional
        Maximum number of bosons on site
    crystal_momentum : int
        Crystal momentum
    reflection_parity : int
        Reflection parity
    super_representative_dim : int, optional
        Hilbert space dimension
    super_representative_basis : np.ndarray, optional
        Hilbert space Fock basis
    super_representative_periods : np.ndarray, optional
        Periods of the representative states
    
    Returns
    -------
    representative_dim : int
        Hilbert space dimension
    representative_basis : np.ndarray
        Hilbert space Fock basis
    representative_periods : np.ndarray
        Periods of the representative states
    representative_reflection_periods : np.ndarray
        Reflection periods of the representative states

    """
    if super_representative_basis is None:
        (super_representative_dim,
         super_representative_basis,
         super_representative_periods
         ) = gen_basis_knblock(num_sites, n_tot, n_max, crystal_momentum)

    representative_state_list = list()
    representative_period_list = list()
    representative_reflection_period_list = list()
    for a in range(super_representative_dim):
        representative_state_a = super_representative_basis[a]
        representative_period = super_representative_periods[a]
        representative_period, representative_reflection_period = fock_checkstate_reflection(representative_state_a,
                                                                                             representative_period)
        if representative_reflection_period != -1:
            if 1.0 + reflection_parity * np.cos(2.0 * np.pi / num_sites
                                                * crystal_momentum * representative_reflection_period) == 0.0:
                representative_period = -1
        if representative_period > 0:
            representative_state_list.append(representative_state_a)  # intentionally avoiding copying
            representative_period_list.append(representative_period)
            representative_reflection_period_list.append(representative_reflection_period)
    representative_basis = np.array(representative_state_list, dtype=int)
    representative_periods = np.array(representative_period_list, dtype=int)
    representative_reflection_periods = np.array(representative_reflection_period_list, dtype=int)
    representative_dim = representative_basis.shape[0]

    return representative_dim, representative_basis, representative_periods, representative_reflection_periods


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
        {'full', 'N', 'KN', 'PKN'}
    sym : str, optional
        {'N', 'KN', 'PKN'}
    n_tot : int, optional
        Total number of bosons
    crystal_momentum : int, optional
        Crystal momentum
    reflection_parity : int, optional
        Reflection parity
    
    """

    def __init__(self,
                 num_sites: int,
                 n_max: int,
                 space: str = 'full',
                 sym: str = None,
                 n_tot: int = None,
                 crystal_momentum: int = None,
                 reflection_parity: int = None):
        self.num_sites = num_sites
        self.n_max = n_max
        self.space = space
        self.dim = None
        self.basis = None
        self.sym = sym
        self.subspaces = None
        self.n_tot = n_tot
        self.crystal_momentum = crystal_momentum
        self.representative_dim = None
        self.representative_basis = None
        self.representative_periods = None
        self.representative_findstate = None
        self.reflection_parity = reflection_parity
        self.representative_reflection_periods = None

        if space == 'full':
            self.dim = dim_full(num_sites, n_max)
            self.basis = gen_basis_full(num_sites, n_max, self.dim)
            if sym in ('N', 'KN', 'PKN'):
                self.subspaces = list()
                for n in range(num_sites * n_max + 1):
                    self.subspaces.append(HilbertSubspace(self.basis,  # intentionally avoiding copying
                                                          num_sites,
                                                          n_max,
                                                          space='N',
                                                          sym=sym,
                                                          n_tot=n))

        elif space == 'N':
            self.dim, self.basis = gen_basis_nblock_nmax(num_sites, n_tot, n_max)
            if sym in ('KN', 'PKN'):
                self.subspaces = list()
                for k in range(num_sites):
                    self.subspaces.append(HilbertSubspace(self.basis,  # intentionally avoiding copying
                                                          num_sites,
                                                          n_max,
                                                          space='KN',
                                                          sym=sym,
                                                          n_tot=n_tot,
                                                          crystal_momentum=k))

        elif space == 'KN':
            self.dim, self.basis = gen_basis_nblock_nmax(num_sites, n_tot, n_max)
            (self.representative_dim,
             self.representative_basis,
             self.representative_periods) = gen_basis_knblock(num_sites, n_tot, n_max, crystal_momentum)
            self.representative_findstate = dict()
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a
            if sym == 'PKN' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
                self.subspaces = list()
                for p in (1, -1):
                    self.subspaces.append(HilbertSubspace(self.representative_basis,  # intentionally avoiding copying
                                                          num_sites,
                                                          n_max,
                                                          space='PKN',
                                                          sym=sym,
                                                          n_tot=n_tot,
                                                          crystal_momentum=crystal_momentum,
                                                          super_dim=self.representative_dim,
                                                          super_periods=self.representative_periods,
                                                          reflection_parity=p))

        elif space == 'PKN' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
            self.dim, self.basis = gen_basis_nblock_nmax(num_sites, n_tot, n_max)
            (self.representative_dim,
             self.representative_basis,
             self.representative_periods,
             self.representative_reflection_periods) = gen_basis_pknblock(num_sites,
                                                                          n_tot,
                                                                          n_max,
                                                                          crystal_momentum,
                                                                          reflection_parity)
            self.representative_findstate = dict()
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a

        self.findstate = dict()
        for a in range(self.dim):
            self.findstate[tuple(self.basis[a])] = a

    # Coulomb interaction Hamiltonian
    def op_hamiltonian_interaction(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            mat[a, a] = 0.5 * np.sum(state_a * (state_a - 1))

        return mat

    # KN-block Coulomb interaction Hamiltonian
    def op_hamiltonian_interaction_k(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=float)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            mat[a, a] = 0.5 * np.sum(representative_state_a * (representative_state_a - 1))

        return mat

    # hopping operator
    def __op_hop(self, i: int, d: tuple, a: int, state_a: np.ndarray, mat: np.ndarray):
        if state_a[i] > 0:
            n_i = state_a[i]
            t_state = fock_lower(state_a, i)
            for d_j in d:
                j = i + d_j
                j = j - round(j / self.num_sites) * self.num_sites  # PBC IF NEEDED
                if t_state[j] < self.n_max:
                    n_j = t_state[j]
                    state_b = fock_raise(t_state, j)
                    b = self.findstate[tuple(state_b)]
                    mat[a, b] += np.sqrt(n_i * (n_j + 1))

    # tunneling Hamiltonian with OBC
    def op_hamiltonian_tunnel_obc(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            self.__op_hop(0, (1,), a, state_a, mat)
            self.__op_hop(self.num_sites - 1, (-1,), a, state_a, mat)
            for i in range(1, self.num_sites - 1):
                self.__op_hop(i, (1, -1), a, state_a, mat)

        return mat

    # tunneling Hamiltonian with PBC
    def op_hamiltonian_tunnel_pbc(self):
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            state_a = self.basis[a]
            for i in range(self.num_sites):
                self.__op_hop(i, (1, -1), a, state_a, mat)

        return mat

    # KN-block hopping operator
    def __op_hop_k(self,
                   i: int,
                   d: tuple,
                   a: int,
                   representative_state_a: np.ndarray,
                   representative_period_a: int,
                   mat: np.ndarray):
        if representative_state_a[i] > 0:
            n_i = representative_state_a[i]
            t_state = fock_lower(representative_state_a, i)
            for d_j in d:
                j = i + d_j
                j = j - round(j / self.num_sites) * self.num_sites  # PBC IF NEEDED
                if t_state[j] < self.n_max:
                    n_j = t_state[j]
                    representative_state_b, phase = fock_representative(fock_raise(t_state, j), self.num_sites)
                    if tuple(representative_state_b) in self.representative_findstate:
                        b = self.representative_findstate[tuple(representative_state_b)]
                        representative_period_b = self.representative_periods[b]
                        phase_arg = 2.0 * np.pi / self.num_sites * self.crystal_momentum * phase
                        mat[a, b] += np.sqrt(n_i * (n_j + 1) * representative_period_a
                                             / representative_period_b) * np.exp(1.0j * phase_arg)  # complex conjugated

    # KN-block tunneling Hamiltonian
    def op_hamiltonian_tunnel_k(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=complex)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            representative_period_a = self.representative_periods[a]
            for i in range(self.num_sites):
                self.__op_hop_k(i, (1, -1), a, representative_state_a, representative_period_a, mat)

        return mat

    # PKN-block hopping operator
    def __op_hop_pk(self,
                    i: int,
                    d: tuple,
                    a: int,
                    representative_state_a: np.ndarray,
                    representative_period_a: int,
                    factor_a: float,
                    mat: np.ndarray):
        if representative_state_a[i] > 0:
            n_i = representative_state_a[i]
            t_state = fock_lower(representative_state_a, i)
            for d_j in d:
                j = i + d_j
                j = j - round(j / self.num_sites) * self.num_sites  # PBC IF NEEDED
                if t_state[j] < self.n_max:
                    n_j = t_state[j]
                    representative_state_b, phase = fock_representative(fock_raise(t_state, j), self.num_sites)
                    (representative_state_b,
                     phase,
                     reflection_phase) = fock_representative_reflection(representative_state_b, self.num_sites, phase)
                    if tuple(representative_state_b) in self.representative_findstate:
                        b = self.representative_findstate[tuple(representative_state_b)]
                        representative_period_b = self.representative_periods[b]
                        representative_reflection_period_b = self.representative_reflection_periods[b]
                        phase_arg = 2.0 * np.pi / self.num_sites * self.crystal_momentum * phase
                        if representative_reflection_period_b != -1:
                            representative_reflection_period_arg_b = (2.0 * np.pi / self.num_sites
                                                                      * self.crystal_momentum
                                                                      * representative_reflection_period_b)
                            factor_b = 1.0 + self.reflection_parity * np.cos(representative_reflection_period_arg_b)
                            factor = ((np.cos(phase_arg) + self.reflection_parity
                                      * np.cos(phase_arg - representative_reflection_period_arg_b))
                                      / (1.0 + self.reflection_parity
                                      * np.cos(representative_reflection_period_arg_b)))
                        else:
                            factor_b = 1.0
                            factor = np.cos(phase_arg)
                        mat[b, a] += (np.sqrt(n_i * (n_j + 1) * representative_period_a * factor_b
                                              / (representative_period_b * factor_a))
                                      * factor * self.reflection_parity ** reflection_phase)  # NOT complex conjugated

    # PKN-block tunneling Hamiltonian
    def op_hamiltonian_tunnel_pk(self):
        mat = np.zeros((self.representative_dim, self.representative_dim), dtype=float)
        for a in range(self.representative_dim):
            representative_state_a = self.representative_basis[a]
            representative_period_a = self.representative_periods[a]
            representative_reflection_period_a = self.representative_reflection_periods[a]
            if representative_reflection_period_a != -1:
                representative_reflection_period_arg_a = (2.0 * np.pi / self.num_sites
                                                          * self.crystal_momentum
                                                          * representative_reflection_period_a)
                factor_a = 1.0 + self.reflection_parity * np.cos(representative_reflection_period_arg_a)
            else:
                factor_a = 1.0
            for i in range(self.num_sites):
                self.__op_hop_pk(i, (1, -1), a, representative_state_a, representative_period_a, factor_a, mat)

        return mat


class HilbertSubspace(HilbertSpace):
    """
    A HilbertSubspace object represents a Hilbert subspace.

    At initialization a Fock basis is constructed for constructing
    operators in the Fock basis.

    Parameters
    ----------
    super_basis : np.ndarray
        Hilbert space Fock basis
    num_sites : int
        Number of sites
    n_max : int
        Maximum number of bosons on site
    n_tot : int
        Total number of bosons
    space : str, default='N'
        {'N', 'KN', 'PKN'}
    sym : str, default='N'
        {'N', 'KN', 'PKN'}
    crystal_momentum : int, optional
        Crystal momentum
    super_dim : int, optional
        Hilbert space dimension
    super_periods : np.ndarray, optional
        Periods of the representative states
    reflection_parity : int, optional
        Reflection parity
    
    """

    def __init__(self,
                 super_basis: np.ndarray,
                 num_sites: int,
                 n_max: int,
                 n_tot: int,
                 space: str = 'N',
                 sym: str = 'N',
                 crystal_momentum: int = None,
                 super_dim: int = None,
                 super_periods: np.ndarray = None,
                 reflection_parity: int = None):
        self.super_basis = super_basis
        self.num_sites = num_sites
        self.n_max = n_max
        self.n_tot = n_tot
        self.space = space
        self.dim = None
        self.basis = None
        self.sym = sym
        self.subspaces = None
        self.crystal_momentum = crystal_momentum
        self.representative_dim = None
        self.representative_basis = None
        self.representative_periods = None
        self.super_dim = super_dim
        self.super_periods = super_periods
        self.reflection_parity = reflection_parity
        self.reflection_periods = None

        if space == 'N':
            self.dim, self.basis = gen_basis_nblock_from_full(n_tot, super_basis)
            self.findstate = dict()
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a
            if sym in ('KN', 'PKN'):
                self.subspaces = list()
                for k in range(num_sites):
                    self.subspaces.append(HilbertSubspace(self.basis,  # intentionally avoiding copying
                                                          num_sites,
                                                          n_max,
                                                          space='KN',
                                                          sym=sym,
                                                          n_tot=n_tot,
                                                          crystal_momentum=k))

        elif space == 'KN':
            self.dim, self.basis = gen_basis_nblock_nmax(num_sites, n_tot, n_max)
            (self.representative_dim,
             self.representative_basis,
             self.representative_periods) = gen_basis_knblock(num_sites,
                                                              n_tot,
                                                              n_max,
                                                              crystal_momentum,
                                                              super_basis=super_basis)
            self.representative_findstate = dict()
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a
            if sym == 'PKN' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
                self.subspaces = list()
                for p in (1, -1):
                    self.subspaces.append(HilbertSubspace(self.representative_basis,
                                                          num_sites,
                                                          n_max,
                                                          space='PKN',
                                                          sym=sym,
                                                          n_tot=n_tot,
                                                          crystal_momentum=crystal_momentum,
                                                          super_dim=self.representative_dim,
                                                          super_periods=self.representative_periods,
                                                          reflection_parity=p))

        elif space == 'PKN':
            self.dim, self.basis = gen_basis_nblock_nmax(num_sites, n_tot, n_max)
            (self.representative_dim,
             self.representative_basis,
             self.representative_periods,
             self.representative_reflection_periods) = gen_basis_pknblock(num_sites,
                                                                          n_tot,
                                                                          n_max,
                                                                          crystal_momentum,
                                                                          reflection_parity,
                                                                          super_representative_dim=super_dim,
                                                                          super_representative_basis=super_basis,
                                                                          super_representative_periods=super_periods)
            self.representative_findstate = dict()
            for a in range(self.representative_dim):
                self.representative_findstate[tuple(self.representative_basis[a])] = a
