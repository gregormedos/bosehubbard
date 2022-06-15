import numpy as np



def dim_hil(N, L):
    return np.math.factorial(N + L - 1) // np.math.factorial(N) // np.math.factorial(L - 1)


def gen_basis(N, L, D):
    if L > 1:
        basis = np.zeros((D, L), dtype=int)
        j = 0
        for n in range(N + 1):
            d = dim_hil(n, L - 1)
            basis[j:j + d, 0] = N - n
            basis[j:j + d, 1:] = gen_basis(n, L - 1, d)
            j += d
    else:
        basis = np.array([N], dtype=int)
    
    return basis


def op_lower(i, fock):
    copy = np.copy(fock)
    copy[i] -= 1
    return copy


def op_lower_n(i, n, fock):
    copy = np.copy(fock)
    if n == 0:
        return copy
    else:
        return op_lower(i, op_lower_n(i, n - 1, copy))


def op_b(i, fock):
    n = fock[i]
    return np.sqrt(n) * op_lower(i, fock)


def op_b_n(i, n, fock):
    copy = np.copy(fock)
    if n == 0:
        return copy
    else:
        return op_b(i, op_b_n(i, n - 1, copy))


def op_raise(i, fock):
    copy = np.copy(fock)
    copy[i] += 1
    return copy


def op_raise_n(i, n, fock):
    copy = np.copy(fock)
    if n == 0:
        return copy
    else:
        return op_raise(i, op_raise_n(i, n - 1, copy))


def op_bdagger(i, fock):
    n = fock[i]
    return np.sqrt(n + 1) * op_raise(i, fock)


def op_bdagger_n(i, n, fock):
    copy = np.copy(fock)
    if n == 0:
        return copy
    else:
        return op_bdagger(i, op_bdagger_n(i, n - 1, copy))


class HilbertSpace:
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.D = dim_hil(self.N, self.L)
        self.basis = gen_basis(self.N, self.L, self.D)
        self.map = dict()
        for j in range(self.D):
            self.map[tuple(self.basis[j])] = j


    def get_fock_state(self, j):
        fock = np.copy(self.basis[j])
        return fock


    def op_kinetic_pbc(self, t):
        new_states = np.zeros((self.D, self.D), dtype=float)
        for j in range(self.D):
            fock = self.get_fock_state(j)
            for i in range(self.L):
                if fock[i] > 0:
                    amplitude1 = np.sqrt(fock[i])
                    lower_fock = op_lower(i, fock)

                    indeks = (i + 1) % self.L # PBC
                    amplitude = amplitude1 * np.sqrt(lower_fock[indeks] + 1)
                    new_fock = op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

                    indeks = (i + self.L - 1) % self.L # PBC
                    amplitude = amplitude1 * np.sqrt(lower_fock[indeks] + 1)
                    new_fock = op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

        return new_states


    def op_interaction(self, U):
        new_states = np.zeros((self.D, self.D), dtype=float)
        for j in range(self.D):
            fock = self.get_fock_state(j)
            amplitude = 0.0
            for i in range(self.L):
                amplitude += fock[i] * (fock[i] - 1)
            
            new_states[j, j] = 0.5 * U * amplitude

        return new_states


    def op_hamiltonian_pbc(self, t, inter, U):
        if inter:
            return self.op_kinetic_pbc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_pbc(t)


    def op_kinetic_obc(self, t):
        new_states = np.zeros((self.D, self.D), dtype=float)
        for j in range(self.D):
            fock = self.get_fock_state(j)

            if fock[0] > 0:
                amplitude1 = np.sqrt(fock[0])
                lower_fock = op_lower(0, fock)
                indeks = 1 # OBC
                amplitude = amplitude1 * np.sqrt(lower_fock[indeks] + 1)
                new_fock = op_raise(indeks, lower_fock)
                new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

            if fock[self.L - 1] > 0:
                amplitude1 = np.sqrt(fock[self.L - 1])
                lower_fock = op_lower(self.L - 1, fock)
                indeks = self.L - 2 # OBC
                amplitude = amplitude1 * np.sqrt(lower_fock[indeks] + 1)
                new_fock = op_raise(indeks, lower_fock)
                new_states[j, self.map[tuple(new_fock)]] += - t * amplitude
                
            for i in range(1, self.L - 1):
                if fock[i] > 0:
                    amplitude1 = np.sqrt(fock[i])
                    lower_fock = op_lower(i, fock)

                    indeks = i + 1 # OBC
                    amplitude = amplitude1 * np.sqrt(lower_fock[indeks] + 1)
                    new_fock = op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

                    indeks = i - 1 # OBC
                    amplitude = amplitude1 * np.sqrt(lower_fock[indeks] + 1)
                    new_fock = op_raise(indeks, lower_fock)
                    new_states[j, self.map[tuple(new_fock)]] += - t * amplitude

        return new_states


    def op_hamiltonian_obc(self, t, inter, U):
        if inter:
            return self.op_kinetic_obc(t) + self.op_interaction(U)
        else:
            return self.op_kinetic_obc(t)


class DensityOfStates:
    def __init__(self, eigen_h, epsilon):
        self.eigen_h = eigen_h
        self.epsilon = epsilon


    def delta(self, eigen, E):
        return self.epsilon / ((E - eigen)**2 + self.epsilon**2) / np.pi

    def dos(self, E):
        res = 0.0
        for eigen in self.eigen_h:
            res += self.delta(eigen, E)
        
        res /= len(self.eigen_h)
        return res
