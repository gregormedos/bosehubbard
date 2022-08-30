import numpy as np



#-----------------------------------------------------------------------------------------------
## lowering operator
#-----------------------------------------------------------------------------------------------
def op_lower(i, fock):
    new_fock = np.copy(fock)
    new_fock[i] -= 1
    return new_fock


#-----------------------------------------------------------------------------------------------
## raising operator
#-----------------------------------------------------------------------------------------------
def op_raise(i, fock):
    new_fock = np.copy(fock)
    new_fock[i] += 1
    return new_fock


#-----------------------------------------------------------------------------------------------
## translation operator
#-----------------------------------------------------------------------------------------------
def op_translation(fock, Lsites):
    new_fock = np.copy(fock)
    for i in range(Lsites):
        new_fock[i] = fock[(i - 1) % Lsites] # PBC
    return new_fock


#-----------------------------------------------------------------------------------------------
## Lorentzian approximation of Delta function
#-----------------------------------------------------------------------------------------------
def Lorentzian(x, epsilon):
        return epsilon / (x**2 + epsilon**2) / np.pi


#-----------------------------------------------------------------------------------------------
## generate full basis with maximum occupancy for every site
# Lsites = number of sites
# Nmax = maximum occupancy for every site
#-----------------------------------------------------------------------------------------------
def gen_basis_nmax(Lsites, Nmax):
    if Lsites > 1:
        Dim = (Nmax + 1)**Lsites
        basis = np.zeros((Dim, Lsites), dtype=int)
        a = 0
        for n in range(Nmax + 1):
            l = Lsites - 1
            d = Dim // (Nmax + 1)
            basis[a:a + d, 0] = n
            basis[a:a + d, 1:] = gen_basis_nmax(l, Nmax)
            a += d
    else:
        Dim = Nmax + 1
        basis = np.zeros((Dim, 1), dtype=int)
        for n in range(Nmax + 1):
            basis[n] = n

    return basis


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
        a = 0
        for n in range(Nquanta + 1):
            l = Lsites - 1
            d = dim_nblock(n, l)
            basis[a:a + d, 0] = Nquanta - n
            basis[a:a + d, 1:] = gen_basis_nblock(l, n, d)
            a += d
    else:
        basis = np.array([Nquanta], dtype=int)
    
    return basis


#-----------------------------------------------------------------------------------------------
## generate basis with const. N and maximum occupancy for every site
## directly (scalable)
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def __gen_basis_n_nmax(Lsites, Nmax, Nquanta):
    if Nquanta < Nmax:
        Nmax = Nquanta
    
    if Lsites > 1:
        basis_list = list()
        for n in range(Nmax + 1):
            block_subbasis = __gen_basis_n_nmax(Lsites - 1, Nmax, Nquanta - n)
            d = len(block_subbasis)
            block_basis = np.zeros((d, Lsites), dtype=int)
            for a in range(d):
                block_basis[a, 0] = n
            block_basis[:, 1:] = block_subbasis
            basis_list.append(block_basis)

        basis_raw = basis_list[0]
        for i in range(1, Nmax + 1):
            basis_raw = np.append(basis_raw, basis_list[i], axis=0)
    else:
        basis_raw = np.array([Nmax], dtype=int)

    return basis_raw

def gen_basis_n_nmax(Lsites, Nmax, Nquanta):
    basis_raw = __gen_basis_n_nmax(Lsites, Nmax, Nquanta)
    basis_list = list()
    for state in basis_raw:
        if np.sum(state) == Nquanta:
            basis_list.append(state)
    
    return np.array(basis_list, dtype=int)


#-----------------------------------------------------------------------------------------------
## generate basis with const. N and maximum occupancy for every site
## from full basis with maximum occupancy for every site (not scalable)
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_n_nmax_from_nmax(superbasis, Nquanta):
    basis_list = list()
    for state in superbasis:
        if np.sum(state) == Nquanta:
            basis_list.append(np.copy(state))
    
    return np.array(basis_list, dtype=int)


#-----------------------------------------------------------------------------------------------
## generate basis with translational symmetry (k)
## with const. N nd maximum occupancy for every site
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_k_n_nmax(Lsites, Nmax, Nquanta, Kmoment):
    basis_raw = gen_basis_n_nmax(Lsites, Nmax, Nquanta)
    d = len(basis_raw)
    basis_list = list()
    periodicity_list = list()
    for a in range(d):
        Period = -1
        state = np.copy(basis_raw[a])
        for i in range(1, Lsites + 1):
            state = op_translation(state, Lsites)
            if tuple(state) < tuple(basis_raw[a]):
                break
            elif tuple(state) == tuple(basis_raw[a]):
                if (Kmoment % (Lsites / i)) == 0:
                    Period = i
                    break
                else:
                    break
    
        if Period >= 0:
            basis_list.append(state)
            periodicity_list.append(Period)

    return (np.array(basis_list, dtype=int), np.array(periodicity_list, dtype=int))


#-----------------------------------------------------------------------------------------------
## generate basis with translational symmetry (k)
## with const. N nd maximum occupancy for every site
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_k_n_nmax_from_n_nmax(superbasis, Lsites, Kmoment):
    d = len(superbasis)
    basis_list = list()
    periodicity_list = list()
    for a in range(d):
        Period = -1
        state = np.copy(superbasis[a])
        for i in range(1, Lsites + 1):
            state = op_translation(state, Lsites)
            if tuple(state) < tuple(superbasis[a]):
                break
            elif tuple(state) == tuple(superbasis[a]):
                if (Kmoment % (Lsites / i)) == 0:
                    Period = i
                    break
                else:
                    break
    
        if Period >= 0:
            basis_list.append(state)
            periodicity_list.append(Period)

    return (np.array(basis_list, dtype=int), np.array(periodicity_list, dtype=int))


#-----------------------------------------------------------------------------------------------
## create a Hilbert space
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Sym = symmetry type for basis generation
# Nquanta = total number of quanta
# Kmoment = crystal momentum k
#-----------------------------------------------------------------------------------------------
class HilbertSpace:
    ## construct a Hilbert space with given parameters
    def __init__(self, Lsites, Nmax, Diag='full', Sym=None, Nquanta=None, Kmoment=None):
        self.Lsites = Lsites                                               # number of sites
        self.Nmax = Nmax                                                   # maximum occupancy for any site
        
        if Diag == 'full':
            self.basis = gen_basis_nmax(Lsites, Nmax)     # full Namx basis
            if Sym in ('N', 'kN'):
                self.subspaces = list()
                for n in range(Lsites * Nmax + 1):
                    self.subspaces.append(HilbertSubspace(self.basis, Lsites, Nmax, Diag='N', Sym=Sym, Nquanta=n)) # N-block Hilbert subspaces

        elif Diag == 'N':
            self.Nquanta = Nquanta                                         # total number of quanta
            if Nquanta < Nmax:
                self.basis = gen_basis_nblock(Lsites, Nquanta, dim_nblock(Lsites, Nquanta))  # N-block basis
            else:
                self.basis = gen_basis_n_nmax(Lsites, Nmax, Nquanta)                         # N Nmax basis
            if Sym == 'kN':
                self.subspaces = list()
                for k in range(Lsites):
                    self.subspaces.append(HilbertSubspace(self.basis, Lsites, Nmax, Diag='kN', Sym=Sym, Nquanta=Nquanta, Kmoment=k))  # kN Nmax basis
                
        elif Diag == 'kN':
            self.Nquanta = Nquanta                                                                                # total number of quanta
            self.Kmoment = Kmoment                                                                                # crystal momentum k
            self.basis, self.periodicities = gen_basis_k_n_nmax(Lsites, Nmax, Nquanta, Kmoment)                   # kN Nmax basis

        self.Dim = len(self.basis)                                         # dimension of Hilbert space
        self.map = dict()
        for a in range(self.Dim):
            self.map[tuple(self.basis[a])] = a                             # mapping fock states to indices


    ## given index return fock state
    def get_fock_state(self, j):
        fock = np.copy(self.basis[j])
        return fock


    ## Coulomb interaction
    def op_interaction(self, U):
        matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            fock = self.get_fock_state(a)
            amplitude = 0.0
            for i in range(self.Lsites):
                amplitude += fock[i] * (fock[i] - 1)
            
            matrika[a, a] = 0.5 * U * amplitude

        return matrika


    ## Hamiltonian with OBC
    def op_kinetic_obc(self, t):
        matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            fock_a = self.get_fock_state(a)

            if fock_a[0] > 0:
                amplitude1 = fock_a[0]
                lower_fock = op_lower(0, fock_a)
                j = 1 # OBC
                if lower_fock[j] < self.Nmax:
                    amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1))
                    fock_b = op_raise(j, lower_fock)
                    b = self.map[tuple(fock_b)]
                    matrika[a, b] += - t * amplitude

            if fock_a[self.Lsites - 1] > 0:
                amplitude1 = fock_a[self.Lsites - 1]
                lower_fock = op_lower(self.Lsites - 1, fock_a)
                j = self.Lsites - 2 # OBC
                if lower_fock[j] < self.Nmax:
                    amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1))
                    fock_b = op_raise(j, lower_fock)
                    b = self.map[tuple(fock_b)]
                    matrika[a, b] += - t * amplitude
                
            for i in range(1, self.Lsites - 1):
                if fock_a[i] > 0:
                    amplitude1 = fock_a[i]
                    lower_fock = op_lower(i, fock_a)

                    j = i + 1 # OBC
                    if lower_fock[j] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1))
                        fock_b = op_raise(j, lower_fock)
                        b = self.map[tuple(fock_b)]
                        matrika[a, b] += - t * amplitude

                    j = i - 1 # OBC
                    if lower_fock[j] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1))
                        fock_b = op_raise(j, lower_fock)
                        b = self.map[tuple(fock_b)]
                        matrika[a, b] += - t * amplitude

        return matrika

    def op_hamiltonian_obc(self, t, inter, U):
        if inter:
            return self.op_kinetic_obc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_obc(t)


    ## Hamiltonian with PBC
    def op_kinetic_pbc(self, t):
        matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            fock_a = self.get_fock_state(a)
            for i in range(self.Lsites):
                if fock_a[i] > 0:
                    amplitude1 = fock_a[i]
                    lower_fock = op_lower(i, fock_a)

                    j = (i + 1) % self.Lsites # PBC
                    if lower_fock[j] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1))
                        fock_b = op_raise(j, lower_fock)
                        b = self.map[tuple(fock_b)]
                        matrika[a, b] += - t * amplitude

                    j = (i + self.Lsites - 1) % self.Lsites # PBC
                    if lower_fock[j] < self.Nmax:
                        amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1))
                        fock_b = op_raise(j, lower_fock)
                        b = self.map[tuple(fock_b)]
                        matrika[a, b] += - t * amplitude

        return matrika

    def op_hamiltonian_pbc(self, t, inter, U):
        if inter:
            return self.op_kinetic_pbc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_pbc(t)


    ## Coulomb interaction for k-block Hamiltonian
    def op_interaction_k(self, U):
        matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            fock = self.get_fock_state(a)
            amplitude = 0.0
            for i in range(self.Lsites):
                amplitude += fock[i] * (fock[i] - 1)
            
            matrika[a, a] = 0.5 * U * amplitude

        return matrika


    ## find representative state for states linked by translations
    def representative(self, fock):
        rep_fock = np.copy(fock)
        new_fock = np.copy(fock)
        Phase = 0
        for i in range(1, self.Lsites + 1):
            new_fock = op_translation(new_fock, self.Lsites)
            if tuple(new_fock) < tuple(rep_fock):
                rep_fock = np.copy(new_fock)
                Phase = i
        
        return (rep_fock, Phase)


    ## kN-block Hamiltonian
    def op_kinetic_k(self, t):
        matrika = np.zeros((self.Dim, self.Dim), dtype=complex)
        for a in range(self.Dim):
            fock_a = self.get_fock_state(a)
            Period_a = self.periodicities[a]
            for i in range(self.Lsites):
                if fock_a[i] > 0:
                    amplitude1 = fock_a[i]
                    lower_fock = op_lower(i, fock_a)

                    j = (i + 1) % self.Lsites # PBC
                    if lower_fock[j] < self.Nmax:
                        new_fock = op_raise(j, lower_fock)
                        fock_b, Phase = self.representative(new_fock)
                        if tuple(fock_b) in self.map:
                            b = self.map[tuple(fock_b)]
                            Period_b = self.periodicities[b]
                            amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1) * Period_a / Period_b)
                            Arg = 2.0 * np.pi * self.Kmoment * Phase / self.Lsites
                            matrika[a, b] += - t * amplitude * complex(np.cos(Arg), np.sin(Arg))

                    j = (i + self.Lsites - 1) % self.Lsites # PBC
                    if lower_fock[j] < self.Nmax:
                        new_fock = op_raise(j, lower_fock)
                        fock_b, Phase = self.representative(new_fock)
                        if tuple(fock_b) in self.map:
                            b = self.map[tuple(fock_b)]
                            Period_b = self.periodicities[b]
                            amplitude = np.sqrt(amplitude1 * (lower_fock[j] + 1) * Period_a / Period_b)
                            Arg = 2.0 * np.pi * self.Kmoment * Phase / self.Lsites
                            matrika[a, b] += - t * amplitude * complex(np.cos(Arg), np.sin(Arg))

        return matrika

    def op_hamiltonian_k(self, t, inter, U):
        if inter:
            return self.op_kinetic_k(t) + self.op_interaction_k(U)
        else:
            return self.op_kinetic_k(t)


#-----------------------------------------------------------------------------------------------
## create a Hilbert subspace
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Sym = symmetry type for basis generation
# Nquanta = total number of quanta
# Kmoment = crystal momentum k
#-----------------------------------------------------------------------------------------------
class HilbertSubspace(HilbertSpace):
    ## construct a Hilbert subspace with given parameters
    def __init__(self, superbasis, Lsites, Nmax, Nquanta, Diag='N', Sym='N', Kmoment=None):
        self.Lsites = Lsites                                               # number of sites
        self.Nmax = Nmax                                                   # maximum occupancy for any site
        self.Nquanta = Nquanta                                             # total number of quanta
        self.superbasis = superbasis

        if Diag == 'N':
            self.basis = gen_basis_n_nmax_from_nmax(superbasis, Nquanta)                      # N Nmax basis
            if Sym == 'kN':
                self.subspaces = list()
                for k in range(Lsites):
                    self.subspaces.append(HilbertSubspace(self.basis, Lsites, Nmax, Diag='kN', Sym=Sym, Nquanta=Nquanta, Kmoment=k))  # kN Nmax basis
                
        elif Diag == 'kN':
            self.Kmoment = Kmoment                                                                          # crystal momentum k
            self.basis, self.periodicities = gen_basis_k_n_nmax_from_n_nmax(superbasis, Lsites, Kmoment)            # kN Nmax basis

        self.Dim = len(self.basis)                                         # dimension of Hilbert space
        self.map = dict()
        for a in range(self.Dim):
            self.map[tuple(self.basis[a])] = a                             # mapping fock states to indices


#-----------------------------------------------------------------------------------------------
## create the density of states
# eigen_h = eqigen energies
# epsilon = broadening of delta function
#-----------------------------------------------------------------------------------------------
class DensityOfStates:
    def __init__(self, spectrum, epsilon):
        self.sprectrum = spectrum              # eigen energies
        self.NumE = len(spectrum)              # number of eigen energies
        self.epsilon = epsilon                 # broadening of Delta function
        self.energies = np.linspace(spectrum[0] - 2 * epsilon, spectrum[-1] + 2 * epsilon, 1001)


    ## density of state
    def __dos(self, E):
        return sum([Lorentzian(E - eigenE, self.epsilon) for eigenE in self.sprectrum]) / self.NumE

    def dos(self):
        dist = self.__dos(self.energies)
        return dist
