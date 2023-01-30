import numpy as np



#-----------------------------------------------------------------------------------------------
## lowering operator
# i = site
# s_state = given state
#-----------------------------------------------------------------------------------------------
def op_lower(i, s_state):
    t_state = np.copy(s_state)
    t_state[i] -= 1
    return t_state


#-----------------------------------------------------------------------------------------------
## raising operator
# i = site
# s_state = given state
#-----------------------------------------------------------------------------------------------
def op_raise(i, s_state):
    t_state = np.copy(s_state)
    t_state[i] += 1
    return t_state


#-----------------------------------------------------------------------------------------------
## translation operator
# s_state = given state
# Lsites = number of sites
#-----------------------------------------------------------------------------------------------
def op_translation(s_state, Lsites):
    t_state = np.copy(s_state)
    for i in range(Lsites):
        t_state[i] = s_state[(i - 1) % Lsites] # PBC
    return t_state


#-----------------------------------------------------------------------------------------------
## check if the given state is the representative state
## calculate the period of the representative state
## representative state is the highest integer tuple among all states linked by translations
# s_state = given state
# Lsites = number of sites
# Kmoment = crystal momentum
#-----------------------------------------------------------------------------------------------
def checkstate(s_state, Lsites, Kmoment):
    Period = -1
    t_state = s_state
    for i in range(1, Lsites + 1):
        t_state = op_translation(t_state, Lsites)
        if tuple(t_state) > tuple(s_state):
            return Period
        elif tuple(t_state) == tuple(s_state):
            if (Kmoment % (Lsites / i)) == 0:
                Period = i
                return Period
            else:
                return Period


#-----------------------------------------------------------------------------------------------
## find the representative state for the given state
## calculate the Phase of translation from the representative state to the given state
## representative state is the highest integer tuple among all states linked by translations
# s_state = given state
# Lsites = number of sites
#-----------------------------------------------------------------------------------------------
def representative(s_state, Lsites):
    r_state = np.copy(s_state)
    t_state = s_state
    Phase = 0
    for i in range(1, Lsites + 1):
        t_state = op_translation(t_state, Lsites)
        if tuple(t_state) > tuple(r_state):
            r_state = t_state
            Phase = i
    
    return (r_state, Phase)


#-----------------------------------------------------------------------------------------------
## reflection operator
# s_state = given state
# Lsites = number of sites
#-----------------------------------------------------------------------------------------------
def op_reflection(s_state, Lsites):
    t_state = np.copy(s_state)
    for i in range(Lsites):
        t_state[i] = s_state[Lsites - 1 - i]
    return t_state


#-----------------------------------------------------------------------------------------------
## check if the given state is the representative state
## calculate the period of the representative state
## representative state is the highest integer tuple among all states linked by translations
# s_state = given state
# Lsites = number of sites
# Kmoment = crystal momentum
#-----------------------------------------------------------------------------------------------
def checkstate_reflection(s_state, Lsites, Kmoment, Period):
    ReflectionPeriod = -1
    t_state = op_reflection(s_state, Lsites)
    for i in range(Period):
        if tuple(t_state) > tuple(s_state):
            Period = -1
            return (Period, ReflectionPeriod)
        elif tuple(t_state) == tuple(s_state):
            ReflectionPeriod = i
            return (Period, ReflectionPeriod)
        t_state = op_translation(t_state, Lsites)

    return (Period, ReflectionPeriod)


#-----------------------------------------------------------------------------------------------
## calculate the dimension of Nmax full Hilbert space
# Lsites = number of sites
# Nmax = maximum occupancy for every site
#-----------------------------------------------------------------------------------------------
def dim_full_nmax(Lsites, Nmax):
    return (Nmax + 1)**Lsites


#-----------------------------------------------------------------------------------------------
## generate Nmax full basis
## with maximum occupancy for every site (Nmax)
## directly (NOT SCALABLE)
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Dim = Hilbert space dimension
#-----------------------------------------------------------------------------------------------
def gen_basis_full_nmax(Lsites, Nmax, Dim):
    if Lsites > 1:
        Basis = np.zeros((Dim, Lsites), dtype=int)
        a = 0
        for n in range(Nmax + 1):
            l = Lsites - 1
            d = Dim // (Nmax + 1)
            Basis[a:a + d, 0] = Nmax - n
            Basis[a:a + d, 1:] = gen_basis_full_nmax(l, Nmax, d)
            a += d
    else:
        Dim = Nmax + 1
        Basis = np.zeros((Dim, 1), dtype=int)
        for n in range(Nmax + 1):
            Basis[n] = Nmax - n

    return Basis


#-----------------------------------------------------------------------------------------------
## calculate the dimension of N-block Hilbert space
# Lsites = number of sites
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def dim_nblock(Lsites, Nquanta):
    return np.math.factorial(Lsites + Nquanta - 1) // np.math.factorial(Lsites - 1) // np.math.factorial(Nquanta)


#-----------------------------------------------------------------------------------------------
## generate N-block basis
## with conservation of the total number of quanta (N)
## directly (SCALABLE)
# Lsites = number of sites
# Nquanta = total number of quanta
# Dim = Hilbert space dimension
#-----------------------------------------------------------------------------------------------
def gen_basis_nblock(Lsites, Nquanta, Dim):
    if Lsites > 1:
        Basis = np.zeros((Dim, Lsites), dtype=int)
        a = 0
        for n in range(Nquanta + 1):
            l = Lsites - 1
            d = dim_nblock(l, n)
            Basis[a:a + d, 0] = Nquanta - n
            Basis[a:a + d, 1:] = gen_basis_nblock(l, n, d)
            a += d
    else:
        Basis = np.array([Nquanta], dtype=int)
    
    return Basis


#-----------------------------------------------------------------------------------------------
## generate Nmax N-block basis
## with conservation of the total number of quanta (N)
## with maximum occupancy for every site (Nmax)
## directly (SCALABLE)
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_nblock_nmax(Lsites, Nmax, Nquanta):
    if Nquanta <= Nmax:
        Dim = dim_nblock(Lsites, Nquanta)
        return (Dim, gen_basis_nblock(Lsites, Nquanta, Dim))
    
    if Lsites > 1:
        Nmin = Nquanta - Nmax
        d_list = list()
        basis_list = list()
        basis_list_len = Nmax + 1
        for n in range(Nmin, Nquanta + 1):
            d, subbasis = gen_basis_nblock_nmax(Lsites - 1, Nmax, n)
            if d < 1:
                basis_list_len -= 1
                continue

            basis_block = np.zeros((d, Lsites), dtype=int)
            for a in range(d):
                basis_block[a, 0] = Nquanta - n
            basis_block[:, 1:] = subbasis
            d_list.append(d)
            basis_list.append(basis_block)

        Dim = np.sum(d_list)
        Basis = np.zeros((Dim, Lsites), dtype=int)
        a = 0
        for i in range(basis_list_len):
            Basis[a:a + d_list[i], :] = basis_list[i]
            a += d_list[i]
        
        return (Dim, Basis)

    else:
        if Nquanta > Nmax:
            return (0, None)
        
        return (1, np.array([Nquanta], dtype=int))


#-----------------------------------------------------------------------------------------------
## generate Nmax N-block basis
## with conservation of the total number of quanta (N)
## with maximum occupancy for every site (Nmax)
## from given full basis (NOT SCALABLE)
# superbasis = basis from which the subbasis is generated
# Nquanta = total number of quanta
#-----------------------------------------------------------------------------------------------
def gen_basis_nblock_from_full(Superbasis, Nquanta):
    Dim = 0
    basis_list = list()
    for s_state in Superbasis:
        if np.sum(s_state) == Nquanta:
            Dim += 1
            basis_list.append(np.copy(s_state))
    
    return (Dim, np.array(basis_list, dtype=int))


#-----------------------------------------------------------------------------------------------
## generate kN-block basis
## with conservation of the crystal momentum (k)
## with conservation of the total number of quanta (N)
## by creating N-block basis (SCALABLE)
## from given N-block basis (SCALABLE)
# Lsites = number of sites
# Nquanta = total number of quanta
# Kmoment = crystal momentum
# superbasis = basis from which the subbasis is generated
# Nmax = maximum occupancy for every site
#-----------------------------------------------------------------------------------------------
def gen_basis_knblock(Lsites, Nquanta, Kmoment, Superbasis=None, Nmax=None):
    if Superbasis is None:
        if Nmax is None:
            SuperDim = dim_nblock(Lsites, Nquanta)
            Superbasis = gen_basis_nblock(Lsites, Nquanta, SuperDim)
        else:
            _, Superbasis = gen_basis_nblock_nmax(Lsites, Nmax, Nquanta)

    Dim = 0
    basis_list = list()
    periodicity_list = list()
    for s_state in Superbasis:
        Period = checkstate(s_state, Lsites, Kmoment)
        if Period >= 0:
            Dim += 1
            basis_list.append(s_state)
            periodicity_list.append(Period)

    return (Dim, np.array(basis_list, dtype=int), np.array(periodicity_list, dtype=int))


#-----------------------------------------------------------------------------------------------
## generate pkN-block basis
## with coservation of reflection parity (p)
## with conservation of the crystal momentum (k)
## with conservation of the total number of quanta (N)
## by creating kN-block basis (SCALABLE)
## from given kN-block basis (SCALABLE)
# Lsites = number of sites
# Nquanta = total number of quanta
# Kmoment = crystal momentum
# Parity = reflection parity
# superbasis = basis from which the subbasis is generated
# Nmax = maximum occupancy for every site
#-----------------------------------------------------------------------------------------------
def gen_basis_pknblock(Lsites, Nquanta, Kmoment, Parity, SuperDim=None, Superbasis=None, SuperPeriodicities=None, Nmax=None):
    if Superbasis is None:
            SuperDim, Superbasis, SuperPeriodicities = gen_basis_knblock(Lsites, Nquanta, Kmoment, Nmax=Nmax)

    Dim = 0
    basis_list = list()
    periodicity_list = list()
    reflection_periodicity_list = list()
    for j in range(SuperDim):
        s_state = Superbasis[j]
        Period = SuperPeriodicities[j]
        Period, ReflectionPeriod = checkstate_reflection(s_state, Lsites, Kmoment, Period)
        if ReflectionPeriod != -1:
            if 1 + Parity * np.cos(2.0 * np.pi / Lsites * Kmoment * ReflectionPeriod) == 0:
                Period = -1
        if Period >= 0:
            Dim += 1
            basis_list.append(s_state)
            periodicity_list.append(Period)
            reflection_periodicity_list.append(ReflectionPeriod)

    return (Dim, np.array(basis_list, dtype=int), np.array(periodicity_list, dtype=int), np.array(reflection_periodicity_list, dtype=int))


#-----------------------------------------------------------------------------------------------
## create a Hilbert space
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# *
# Diag = specify any fixed commuting quantum numbers to diagonalize a block
# Sym = symmetry type for block diagonalization
# Nquanta = conserved total number of quanta (N)
# Kmoment = conserved crystal momentum (k)
# Parity = conserved reflection parity (p)
# *
#-----------------------------------------------------------------------------------------------
class HilbertSpace:
    ## construct a Hilbert space with given parameters
    def __init__(self, Lsites, Nmax, Diag='full', Sym=None, Nquanta=None, Kmoment=None, Parity=None):
        self.Lsites = Lsites   # number of sites
        self.Nmax = Nmax       # maximum occupancy for any site
        
        if Diag == 'full':
            self.Dim = dim_full_nmax(Lsites, Nmax)                    # dimension of Hilbert space
            self.Basis = gen_basis_full_nmax(Lsites, Nmax, self.Dim)  # Nmax full basis
            if Sym in ('N', 'kN', 'pkN'):
                self.Subspaces = list()
                for n in range(Lsites * Nmax + 1):
                    self.Subspaces.append(HilbertSubspace(self.Dim, self.Basis, Lsites, Nmax, Diag='N', Sym=Sym, Nquanta=n)) # N-block Hilbert subspaces

        elif Diag == 'N':
            self.Nquanta = Nquanta  # total number of quanta
            if Nquanta <= Nmax:
                self.Dim = dim_nblock(Lsites, Nquanta)                    # dimension of Hilbert space
                self.Basis = gen_basis_nblock(Lsites, Nquanta, self.Dim)  # N-block basis
            else:
                self.Dim, self.Basis = gen_basis_nblock_nmax(Lsites, Nmax, Nquanta) # dimension of Hilbert space # Nmax N-block basis
            if Sym in ('kN', 'pkN'):
                self.Subspaces = list()
                for k in range(Lsites):
                    self.Subspaces.append(HilbertSubspace(self.Dim, self.Basis, Lsites, Nmax, Diag='kN', Sym=Sym, Nquanta=Nquanta, Kmoment=k))  # kN-block Hilbert subspaces
                
        elif Diag == 'kN':
            self.Nquanta = Nquanta   # total number of quanta
            self.Kmoment = Kmoment   # crystal momentum k
            if Nquanta <= Nmax:
                self.Dim, self.Basis, self.Periodicities = gen_basis_knblock(Lsites, Nquanta, Kmoment)             # dimension of Hilbert space # kN-block basis
            else:
                self.Dim, self.Basis, self.Periodicities = gen_basis_knblock(Lsites, Nquanta, Kmoment, Nmax=Nmax)  # dimension of Hilbert space # Nmax kN-block basis
            if Sym == 'pkN':
                self.Subspaces = list()
                for p in (-1, 1):
                    self.Subspaces.append(HilbertSubspace(self.Dim, self.Basis, Lsites, Nmax, Diag='pkN', Sym=Sym, Nquanta=Nquanta, Kmoment=Kmoment, SuperPeriodicities=self.Periodicities, Parity=p)) # pkN Hilbert subspaces

        elif Diag == 'pkN':
            self.Nquanta = Nquanta     # total number of quanta
            self.Kmoment = Kmoment     # crystal momentum k
            self.Parity = Parity       # reflection parity p
            if Nquanta <= Nmax:
                self.Dim, self.Basis, self.Periodicities, self.ReflectionPeriodicites = gen_basis_pknblock(Lsites, Nquanta, Kmoment, Parity)             # dimension of Hilbert space # pkN-block basis
            else:
                self.Dim, self.Basis, self.Periodicities, self.ReflectionPeriodicites = gen_basis_pknblock(Lsites, Nquanta, Kmoment, Parity, Nmax=Nmax)             # dimension of Hilbert space # Nmax pkN-block basis

        self.Findstate = dict()
        for a in range(self.Dim):
            self.Findstate[tuple(self.Basis[a])] = a       # mapping fock states to indices


    ## Coulomb interaction
    def op_interaction(self, U):
        Matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            state_a = self.Basis[a]            
            Matrika[a, a] = 0.5 * U * np.sum(state_a * (state_a - 1))

        return Matrika


    ## hopping operator
    def __op_hop(self, t, i, d, a, state_a, Matrika):
        if state_a[i] > 0:
            n_i = state_a[i]
            t_state = op_lower(i, state_a)
            for d_j in np.array(d, dtype=int):
                j = i + d_j
                j = j - round(j / self.Lsites) * self.Lsites # PBC IF NEEDED
                if t_state[j] < self.Nmax:
                    n_j = t_state[j]
                    state_b = op_raise(j, t_state)
                    b = self.Findstate[tuple(state_b)]
                    Matrika[a, b] -= t * np.sqrt(n_i * (n_j + 1))


    ## Hamiltonian with OBC
    def op_kinetic_obc(self, t):
        Matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            state_a = self.Basis[a]
            self.__op_hop(t, 0, 1, a, state_a, Matrika)
            self.__op_hop(t, self.Lsites - 1, -1, a, state_a, Matrika)               
            for i in range(1, self.Lsites - 1):
                self.__op_hop(t, i, (-1, 1), a, state_a, Matrika)

        return Matrika

    def op_hamiltonian_obc(self, t, inter=False, U=None):
        if inter:
            return self.op_kinetic_obc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_obc(t)


    ## Hamiltonian with PBC
    def op_kinetic_pbc(self, t):
        Matrika = np.zeros((self.Dim, self.Dim), dtype=float)
        for a in range(self.Dim):
            state_a = self.Basis[a]
            for i in range(self.Lsites):
                self.__op_hop(t, i, (-1, 1), a, state_a, Matrika)

        return Matrika

    def op_hamiltonian_pbc(self, t, inter=False, U=None):
        if inter:
            return self.op_kinetic_pbc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_pbc(t)


    ## hopping operator
    def __op_hop_k(self, t, i, d, a, state_a, Period_a, Matrika):
        if state_a[i] > 0:
            n_i = state_a[i]
            t_state = op_lower(i, state_a)
            for d_j in np.array(d, dtype=int):
                j = i + d_j
                j = j - round(j / self.Lsites) * self.Lsites # PBC IF NEEDED
                if t_state[j] < self.Nmax:
                    n_j = t_state[j]
                    state_b, Phase = representative(op_raise(j, t_state), self.Lsites)
                    if tuple(state_b) in self.Findstate:
                        b = self.Findstate[tuple(state_b)]
                        Period_b = self.Periodicities[b]
                        Arg = 2.0 * np.pi / self.Lsites * self.Kmoment * Phase
                        Matrika[a, b] -= t * np.sqrt(n_i * (n_j + 1) * Period_a / Period_b) * complex(np.cos(Arg), np.sin(Arg))


    ## kN-block Hamiltonian
    def op_kinetic_k(self, t):
        Matrika = np.zeros((self.Dim, self.Dim), dtype=complex)
        for a in range(self.Dim):
            state_a = self.Basis[a]
            Period_a = self.Periodicities[a]
            for i in range(self.Lsites):
                self.__op_hop_k(t, i, (-1, 1), a, state_a, Period_a, Matrika)
        
        return Matrika

    def op_hamiltonian_k(self, t, inter=False, U=None):
        if inter:
            return self.op_kinetic_k(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_k(t)


#-----------------------------------------------------------------------------------------------
## create a Hilbert subspace
# SuperDim = dimension of Superbasis
# Superbasis = basis from which the subbasis is generated
# Lsites = number of sites
# Nmax = maximum occupancy for every site
# Nquanta = conserved total number of quanta (N)
# *
# Diag = specify any fixed commuting quantum numbers to diagonalize a block
# Sym = symmetry type for block diagonalization
# Kmoment = conserved crystal momentum (k)
# Parity = conserved reflection parity (p)
# *
#-----------------------------------------------------------------------------------------------
class HilbertSubspace(HilbertSpace):
    ## construct a Hilbert subspace with given parameters
    def __init__(self, SuperDim, Superbasis, Lsites, Nmax, Nquanta, Diag='N', Sym='N', Kmoment=None, SuperPeriodicities=None, Parity=None):
        self.SuperDim = SuperDim         # dimension of Superbasis
        self.Superbasis = Superbasis     # basis from which the subbasis is generated
        self.Lsites = Lsites             # number of sites
        self.Nmax = Nmax                 # maximum occupancy for any site
        self.Nquanta = Nquanta           # total number of quanta

        if Diag == 'N':
            self.Dim, self.Basis = gen_basis_nblock_from_full(Superbasis, Nquanta)      # N-block basis
            if Sym in ('kN', 'pkN'):
                self.Subspaces = list()
                for k in range(Lsites):
                    self.Subspaces.append(HilbertSubspace(self.Basis, Lsites, Nmax, Diag='kN', Sym=Sym, Nquanta=Nquanta, Kmoment=k))  # kN-block Hilbert subspaces
                
        elif Diag == 'kN':
            self.Kmoment = Kmoment        # crystal momentum k
            self.Dim, self.Basis, self.Periodicities = gen_basis_knblock(Lsites, Nquanta, Kmoment, Superbasis=Superbasis)    # kN-block basis
            if Sym == 'pkN':
                self.Subspaces = list()
                for p in (-1, 1):
                    self.Subspaces.append(HilbertSubspace(self.Basis, Lsites, Nmax, Diag='pkN', Sym=Sym, Nquanta=Nquanta, Kmoment=Kmoment, SuperPeriodicites=self.Periodicities, Parity=p)) # pkN-block Hilbert subspaces

        elif Diag == 'pkN':
            self.Kmoment = Kmoment    # crystal momentum k
            self.Parity = Parity      # reflection parity p
            self.Dim, self.Basis, self.Periodicities, self.ReflectionPeriodicites = gen_basis_pknblock(Lsites, Nquanta, Kmoment, Parity, SuperDim=SuperDim, Superbasis=Superbasis, SuperPeriodicities=SuperPeriodicities)   # pkN-block basis

        self.Findstate = dict()
        for a in range(self.Dim):
            self.Findstate[tuple(self.Basis[a])] = a     # mapping fock states to indices


