import numpy as np
from .dim import *
from .basis import *


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
        self.dim = None
        self.basis = None
        self.findstate = None
        self.super_dim = None
        self.super_basis = None
        self.n_tot = n_tot
        self.crystal_momentum = crystal_momentum
        self.translation_periods = None
        self.reflection_parity = reflection_parity
        self.nums_translations_reflection = None

        if space == 'full':
            self.dim = dim_full(num_sites, n_max)
            self.basis = gen_basis_full(num_sites, self.dim, n_max)
            self.super_dim = self.dim
            self.super_basis = self.basis  # intentionally avoid copying

        elif space == 'N':
            self.dim = dim_nblock(num_sites, n_tot, n_max)
            self.basis = gen_basis_nblock(num_sites, n_tot, self.dim, n_max)
            self.super_dim = self.dim
            self.super_basis = self.basis  # intentionally avoid copying

        elif space == 'K':
            self.super_dim = dim_full(num_sites, n_max)
            self.super_basis = gen_basis_full(num_sites, self.super_dim, n_max)
            (
                self.basis,
                self.translation_periods,
                self.dim
            ) = gen_representative_basis_kblock(self.super_basis, num_sites, crystal_momentum)

        elif space == 'KN':
            self.super_dim = dim_nblock(num_sites, n_tot, n_max)
            self.super_basis = gen_basis_nblock(num_sites, n_tot, self.super_dim, n_max)
            (
                self.basis,
                self.translation_periods,
                self.dim
            ) = gen_representative_basis_kblock(self.super_basis, num_sites, crystal_momentum)

        elif space == 'PK' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
            self.super_dim = dim_full(num_sites, n_max)
            self.super_basis = gen_basis_full(num_sites, self.super_dim, n_max)
            (
                self.basis,
                self.translation_periods,
                self.nums_translations_reflection,
                self.dim
            ) = gen_representative_basis_pkblock(
                self.super_basis,
                num_sites,
                crystal_momentum,
                reflection_parity
            )

        elif space == 'PKN' and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
            self.super_dim = dim_nblock(num_sites, n_tot, n_max)
            self.super_basis = gen_basis_nblock(num_sites, n_tot, self.super_dim, n_max)
            (
                self.basis,
                self.translation_periods,
                self.nums_translations_reflection,
                self.dim
            ) = gen_representative_basis_pkblock(
                self.super_basis,
                num_sites,
                crystal_momentum,
                reflection_parity
            )

        else:
            raise ValueError("Value of `space` must be in `{'full', 'N', 'K', 'KN', 'PK', 'PKN'}`")
        
        self.findstate = {}
        for a in range(self.dim):
            self.findstate[tuple(self.basis[a])] = a

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
                self.super_basis,
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
                    self.findstate[tuple(representative_state_a)],
                    beginning_of_block + a
                ] += normalization_a
                if num_translations_reflection_a == -1:
                    r_state_a, num_translations_a = fock_representative(np.flipud(representative_state_a), self.num_sites)
                    change_of_basis_mat[
                        self.findstate[tuple(r_state_a)],
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
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
                    if tuple(representative_state_b) in self.findstate:
                        b = self.findstate[tuple(representative_state_b)]
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
        mat = np.zeros((self.dim, self.dim), dtype=dtype)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
                    if tuple(representative_state_b) in self.findstate:
                        b = self.findstate[tuple(representative_state_b)]
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
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
            if tuple(representative_state_b) in self.findstate:
                b = self.findstate[tuple(representative_state_b)]
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
        mat = np.zeros((self.dim, self.dim), dtype=dtype)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
        mat = np.zeros((self.dim, self.dim), dtype=dtype)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
            if tuple(representative_state_b) in self.findstate:
                b = self.findstate[tuple(representative_state_b)]
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
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
        mat = np.zeros((self.dim, self.dim), dtype=float)
        for a in range(self.dim):
            representative_state_a = self.basis[a]
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
            super_dim: int = None,
            super_basis: np.ndarray = None,
    ):
        self.num_sites = num_sites
        self.n_max = n_max
        self.space = space
        self.sym = sym
        self.dim = None
        self.basis = None
        self.findstate = None
        self.super_dim = None
        self.super_basis = None
        self.n_tot = n_tot
        self.crystal_momentum = crystal_momentum
        self.translation_periods = None
        self.reflection_parity = reflection_parity
        self.nums_translations_reflection = None
        self.subspaces = None

        if super_dim is None or super_basis is None:
            super().__init__(
                num_sites,
                n_max,
                space,
                n_tot,
                crystal_momentum,
                reflection_parity
            )
        
        else:
            self.super_dim = super_dim
            self.super_basis = super_basis  # intentionally avoiding copying

            if space == 'full':        
                self.dim = super_dim
                self.basis = super_basis  # intentionally avoiding copying

            elif space == 'N':
                self.basis, self.dim = gen_basis_nblock_from_full(super_basis, n_tot)
            
            elif space in {'K', 'KN'}:
                (
                    self.basis,
                    self.translation_periods,
                    self.dim
                ) = gen_representative_basis_kblock(super_basis, num_sites, crystal_momentum)

            elif space in {'PK', 'PKN'} and (crystal_momentum == 0 or (num_sites % 2 == 0 and crystal_momentum == num_sites // 2)):
                (
                    self.basis,
                    self.translation_periods,
                    self.nums_translations_reflection,
                    self.dim
                ) = gen_representative_basis_pkblock(
                    super_basis,
                    num_sites,
                    crystal_momentum,
                    reflection_parity
                )

            else:
                raise ValueError("Value of `space` must be in `{'full', 'N', 'K', 'KN', 'PK', 'PKN'}`")
            
            self.findstate = {}
            for a in range(self.dim):
                self.findstate[tuple(self.basis[a])] = a

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
                            super_dim=self.dim,
                            super_basis=self.basis  # intentionally avoiding copying
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
                            super_dim=self.dim,
                            super_basis=self.basis  # intentionally avoiding copying
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
                            super_dim=self.dim,
                            super_basis=self.basis  # intentionally avoiding copying
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
                            super_dim=self.super_dim,
                            super_basis=self.super_basis  # intentionally avoiding copying
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
                            super_dim=super_dim,
                            super_basis=super_basis  # intentionally avoiding copying
                        )
                    )
