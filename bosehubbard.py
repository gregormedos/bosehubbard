import numpy as np



## generate basis with maximum occupancy for every site
# L = number of sites
# Nmax = maximum occupancy for every site
def gen_basis_nmax(L, Nmax):
    if L > 1:
        D = (Nmax + 1)**L
        basis = np.zeros((D, L), dtype=int)
        j = 0
        for n in range(Nmax + 1):
            d = D // (Nmax + 1)
            basis[j:j + d, 0] = n
            basis[j:j + d, 1:] = gen_basis_nmax(L - 1, Nmax)
            j += d
    else:
        D = Nmax + 1
        basis = np.zeros((D, 1), dtype=int)
        for n in range(Nmax + 1):
            basis[n] = n

    return basis


## generate basis with const. N and maximum occupancy for every site
## from basis with maximum occupancy for every site (not scalable)
# N = total number of quanta
# L = number of sites
# Nmax = maximum occupancy for every site
def gen_basis_n_nmax_from_nmax(N, L, Nmax):
    D = (Nmax + 1)**L
    basis = gen_basis_nmax(L, Nmax)
    new_basis = list()
    for j in range(D):
        if np.sum(basis[j]) == N:
            new_basis.append(basis[j])
    
    return np.array(new_basis)


## calculate the dimension of a N-block
# N = total number of quanta
# L = number of sites
# Nmax = maximum occupancy for every site
def dim_nblock(N, L):
    return np.math.factorial(N + L - 1) // np.math.factorial(N) // np.math.factorial(L - 1)


## generate N-block basis 
# N = total number of quanta
# L = number of sites
# Nmax = maximum occupancy for every site
def gen_basis_nblock(N, L, D):
    if L > 1:
        basis = np.zeros((D, L), dtype=int)
        j = 0
        for n in range(N + 1):
            d = dim_nblock(n, L - 1)
            basis[j:j + d, 0] = N - n
            basis[j:j + d, 1:] = gen_basis_nblock(n, L - 1, d)
            j += d
    else:
        basis = np.array([N], dtype=int)
    
    return basis


## generate basis with const. N and maximum occupancy for every site
## directly (scalable)
# N = total number of quanta
# L = number of sites
# Nmax = maximum occupancy for every site
def gen_basis_n_nmax(N, L, Nmax):
    if (N - Nmax < 0):
        Nmax = N
    
    if L > 1:
        basis_l = list()
        for n in range(Nmax + 1):
            block_subbasis = gen_basis_n_nmax(N - n, L - 1, Nmax)
            d = len(block_subbasis)
            block_basis = np.zeros((d, L), dtype=int)
            for j in range(d):
                block_basis[j, 0] = n
                block_basis[:, 1:] = block_subbasis
            
            basis_l.append(block_basis)
        
        basis_raw = basis_l[0]
        for i in range(1, Nmax + 1):
            basis_raw = np.append(basis_raw, basis_l[i], axis=0)
    else:
        basis_raw = np.array([Nmax], dtype=int)

    basis_l = list()
    d = 0
    for state in basis_raw:
        if np.sum(state) == N:
            basis_l.append(state)
            d += 1
    
    basis = np.zeros((d, L), dtype=int)
    for j in range(d):
        basis[j] = basis_l[j]
    
    return basis


# create a Hilbert space
# N = total number of quanta
# L = number of sites
# Nmax = maximum occupancy for every site
class HilbertSpace:
    def __init__(self, N, L, Nmax):
        self.N = N                                              # total number of quanta
        self.L = L                                              # number of sites
        self.Nmax = Nmax                                        # maximum occupancy for any site
        self.basis = gen_basis_n_nmax(N, L, Nmax)               # basis with const. N and maximum occupancy for every site
        self.D = len(self.basis)                                # dimension of basis
        self.map = dict()
        for j in range(self.D):
            self.map[tuple(self.basis[j])] = j                  # mapping fock states to indices


    # given index return fock state
    def get_fock_state(self, j):
        fock = np.copy(self.basis[j])
        return fock


    # lowering operator
    def op_lower(self, i, fock):
        copy = np.copy(fock)
        copy[i] -= 1
        return copy

    def op_lower_n(self, i, n, fock):
        copy = np.copy(fock)
        if n == 0:
            return copy
        else:
            return self.op_lower(i, self.op_lower_n(i, n - 1, copy))


    # raising operator
    def op_raise(self, i, fock):
        copy = np.copy(fock)
        copy[i] += 1
        return copy

    def op_raise_n(self, i, n, fock):
        copy = np.copy(fock)
        if n == 0:
            return copy
        else:
            return self.op_raise(i, self.op_raise_n(i, n - 1, copy))


    # Coulomb interaction
    def op_interaction(self, U):
        new_states = np.zeros((self.D, self.D), dtype=float)
        for j in range(self.D):
            fock = self.get_fock_state(j)
            amplitude = 0.0
            for i in range(self.L):
                amplitude += fock[i] * (fock[i] - 1)
            
            new_states[j, j] = 0.5 * U * amplitude

        return new_states


    # Hamiltonian with PBC
    def op_kinetic_pbc(self, t):
        new_states = np.zeros((self.D, self.D), dtype=float)
        for j in range(self.D):
            fock = self.get_fock_state(j)
            for i in range(self.L):
                if fock[i] > 0:
                    amplitude1 = fock[i]
                    lower_fock = self.op_lower(i, fock)

                    indeks = (i + 1) % self.L # PBC
                    if lower_fock[indeks] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                        new_fock = self.op_raise(indeks, lower_fock)
                        new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

                    indeks = (i + self.L - 1) % self.L # PBC
                    if lower_fock[indeks] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                        new_fock = self.op_raise(indeks, lower_fock)
                        new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

        return new_states

    def op_hamiltonian_pbc(self, t, inter, U):
        if inter:
            return self.op_kinetic_pbc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_pbc(t)


    # Hamiltonian with OBC
    def op_kinetic_obc(self, t):
        new_states = np.zeros((self.D, self.D), dtype=float)
        for j in range(self.D):
            fock = self.get_fock_state(j)

            if fock[0] > 0:
                amplitude1 = fock[0]
                lower_fock = self.op_lower(0, fock)
                indeks = 1 # OBC
                if fock[indeks] < self.Nmax:
                    amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                    new_fock = self.op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

            if fock[self.L - 1] > 0:
                amplitude1 = fock[self.L - 1]
                lower_fock = self.op_lower(self.L - 1, fock)
                indeks = self.L - 2 # OBC
                if fock[indeks] < self.Nmax:
                    amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                    new_fock = self.op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude
                
            for i in range(1, self.L - 1):
                if fock[i] > 0:
                    amplitude1 = fock[i]
                    lower_fock = self.op_lower(i, fock)

                    indeks = i + 1 # OBC
                    if fock[indeks] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                        new_fock = self.op_raise(indeks, lower_fock)
                        new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

                    indeks = i - 1 # OBC
                    if fock[indeks] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                        new_fock = self.op_raise(indeks, lower_fock)
                        new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

        return new_states


    def op_hamiltonian_obc(self, t, inter, U):
        if inter:
            return self.op_kinetic_obc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_obc(t)


# create the density of states
class DensityOfStates:
    def __init__(self, eigen_h, epsilon):
        self.eigen_h = eigen_h              # eigen energies
        self.epsilon = epsilon              # broadening of delta function


    def delta(self, eigen, E):
        return self.epsilon / ((E - eigen)**2 + self.epsilon**2) / np.pi


    def dos(self, E):
        res = 0.0
        for eigen in self.eigen_h:
            res += self.delta(eigen, E)
        
        res /= len(self.eigen_h)
        return res
