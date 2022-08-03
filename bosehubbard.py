import numpy as np



#-----------------------------------------------------------------------------------------------
## calculate the dimension of a N-block
# Lsites = number of sites
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def dim_nblock(Lsites, Nquanta):
    return np.math.factorial(Lsites + Nquanta - 1) // np.math.factorial(Lsites - 1) // np.math.factorial(Nquanta)


#-----------------------------------------------------------------------------------------------
## generate N-block basis 
# Lsites = number of sites
# Nquanta = total number of quanta
# Dim = dimension of basis
#-----------------------------------------------------------------------------------------------
def gen_basis_nblock(Lsites, Nquanta, Dim):
    if Lsites > 1:
        basis = np.zeros((Dim, Lsites), dtype=int)
        j = 0
        for n in range(Nquanta + 1):
            l = Lsites - 1
            d = dim_nblock(n, l)
            basis[j:j + d, 0] = Nquanta - n
            basis[j:j + d, 1:] = gen_basis_nblock(l, n, d)
            j += d
    else:
        basis = np.array([Nquanta], dtype=int)
    
    return basis


#-----------------------------------------------------------------------------------------------
## generate basis with maximum occupancy for every site
# Lsites = number of sites
# Nmax = maximum occupancy for every site
#-----------------------------------------------------------------------------------------------
def gen_basis_nmax(Lsites, Nmax):
    if Lsites > 1:
        Dim = (Nmax + 1)**Lsites
        basis = np.zeros((Dim, Lsites), dtype=int)
        j = 0
        for n in range(Nmax + 1):
            l = Lsites - 1
            d = Dim // (Nmax + 1)
            basis[j:j + d, 0] = n
            basis[j:j + d, 1:] = gen_basis_nmax(l, Nmax)
            j += d
    else:
        Dim = Nmax + 1
        basis = np.zeros((Dim, 1), dtype=int)
        for n in range(Nmax + 1):
            basis[n] = n

    return basis


#-----------------------------------------------------------------------------------------------
## generate basis with const. N and maximum occupancy for every site
## from basis with maximum occupancy for every site (not scalable)
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_n_nmax_from_nmax(Lsites, Nmax, Nquanta):
    Dim = (Nmax + 1)**Lsites
    basis = gen_basis_nmax(Lsites, Nmax)
    new_basis = list()
    for j in range(Dim):
        if np.sum(basis[j]) == Nquanta:
            new_basis.append(basis[j])
    
    return np.array(new_basis)


#-----------------------------------------------------------------------------------------------
## generate basis with const. N and maximum occupancy for every site
## directly (scalable)
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_n_nmax(Lsites, Nmax, Nquanta):
    if (Nquanta < Nmax):
        Nmax = Nquanta
    
    if Lsites > 1:
        basis_list = list()
        for n in range(Nmax + 1):
            block_subbasis = gen_basis_n_nmax(Lsites - 1, Nmax, Nquanta - n)
            d = len(block_subbasis)
            block_basis = np.zeros((d, Lsites), dtype=int)
            for j in range(d):
                block_basis[j, 0] = n
                block_basis[:, 1:] = block_subbasis
            
            basis_list.append(block_basis)
        
        basis_raw = basis_list[0]
        for i in range(1, Nmax + 1):
            basis_raw = np.append(basis_raw, basis_list[i], axis=0)
    else:
        basis_raw = np.array([Nmax], dtype=int)

    basis_list = list()
    d = 0
    for state in basis_raw:
        if np.sum(state) == Nquanta:
            basis_list.append(state)
            d += 1
    
    basis = np.zeros((d, Lsites), dtype=int)
    for j in range(d):
        basis[j] = basis_list[j]
    
    return basis


#-----------------------------------------------------------------------------------------------
## create a Hilbert space
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
class HilbertSpace:
    def __init__(self, Lsites, Nmax, Sym, Nquanta):
        self.Lsites = Lsites                                               # number of sites
        self.Nmax = Nmax                                                   # maximum occupancy for any site
        if Sym == 'N':
            self.Nquanta = Nquanta                                         # total number of quanta
            self.basis = gen_basis_n_nmax(Lsites, Nmax, Nquanta)           # basis with const. N and maximum occupancy for every site
        else:
            self.basis = gen_basis_nmax(Lsites, Nmax)
        
        self.Dim = len(self.basis)                                         # dimension of basis
        self.map = dict()
        for j in range(self.Dim):
            self.map[tuple(self.basis[j])] = j                             # mapping fock states to indices


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
        new_states = np.zeros((self.Dim, self.Dim), dtype=float)
        for j in range(self.Dim):
            fock = self.get_fock_state(j)
            amplitude = 0.0
            for i in range(self.Lsites):
                amplitude += fock[i] * (fock[i] - 1)
            
            new_states[j, j] = 0.5 * U * amplitude

        return new_states


    # Hamiltonian with PBC
    def op_kinetic_pbc(self, t):
        new_states = np.zeros((self.Dim, self.Dim), dtype=float)
        for j in range(self.Dim):
            fock = self.get_fock_state(j)
            for i in range(self.Lsites):
                if fock[i] > 0:
                    amplitude1 = fock[i]
                    lower_fock = self.op_lower(i, fock)

                    indeks = (i + 1) % self.Lsites # PBC
                    if lower_fock[indeks] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                        new_fock = self.op_raise(indeks, lower_fock)
                        new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

                    indeks = (i + self.Lsites - 1) % self.Lsites # PBC
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
        new_states = np.zeros((self.Dim, self.Dim), dtype=float)
        for j in range(self.Dim):
            fock = self.get_fock_state(j)

            if fock[0] > 0:
                amplitude1 = fock[0]
                lower_fock = self.op_lower(0, fock)
                indeks = 1 # OBC
                if fock[indeks] < self.Nmax:
                    amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                    new_fock = self.op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

            if fock[self.Lsites - 1] > 0:
                amplitude1 = fock[self.Lsites - 1]
                lower_fock = self.op_lower(self.Lsites - 1, fock)
                indeks = self.Lsites - 2 # OBC
                if fock[indeks] < self.Nmax:
                    amplitude = np.sqrt(amplitude1 * (lower_fock[indeks] + 1))
                    new_fock = self.op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude
                
            for i in range(1, self.Lsites - 1):
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


#-----------------------------------------------------------------------------------------------
## create the density of states
# eigen_h = eqigen energies
# epsilon = broadening of delta function
#-----------------------------------------------------------------------------------------------
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
