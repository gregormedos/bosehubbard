import numpy as np
import bosehubbard as bh
import h5py
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


HAMILTONIAN_DICT = {
    'PK': {
        't': bh.HilbertSpace.op_hamiltonian_tunnel_pk,
        'U': bh.HilbertSpace.op_hamiltonian_interaction,
        'g1': bh.HilbertSpace.op_hamiltonian_annihilate_create_pk,
        'g2': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pk
    },
    'K': {
        't': bh.HilbertSpace.op_hamiltonian_tunnel_k,
        'U': bh.HilbertSpace.op_hamiltonian_interaction,
        'g1': bh.HilbertSpace.op_hamiltonian_annihilate_create_k,
        'g2': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_k
    },
    'None': {
        't': bh.HilbertSpace.op_hamiltonian_tunnel_pbc,
        'U': bh.HilbertSpace.op_hamiltonian_interaction,
        'g1': bh.HilbertSpace.op_hamiltonian_annihilate_create,
        'g2': bh.HilbertSpace.op_hamiltonian_annihilate_create_pair_pbc
    }
}


def main():
    test('full', L=6, M=2, terms={'t': 1.0, 'U': 1.0})
    test('full', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g1': 1.0})
    test('full', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g2': 1.0})
    test('full', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g1': 1.0, 'g2': 1.0})
    test('Z2', L=6, M=2, terms={'t': 1.0, 'U': 1.0})
    test('Z2', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g2': 1.0})
    test('N', L=6, M=2, terms={'t': 1.0, 'U': 1.0})
    test('K', L=6, M=2, terms={'t': 1.0, 'U': 1.0})
    test('K', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g1': 1.0})
    test('K', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g2': 1.0})
    test('K', L=6, M=2, terms={'t': 1.0, 'U': 1.0, 'g1': 1.0, 'g2': 1.0})


def plot_dos(dir_name: str, file_name: str, eigen_energies: np.ndarray, reference_eigen_energies: np.ndarray = None):
    if reference_eigen_energies is None:
        plt.figure(dpi=300)
        plt.hist(eigen_energies, 50, color='b')
        plt.xlabel('$E$')
        plt.ylabel('DOS($E$)')
        plt.tight_layout()
        plt.savefig(f'{dir_name}{file_name}_dos.pdf')
        plt.close()
    else:
        fig, ax = plt.subplots(dpi=300)
        ax.hist(reference_eigen_energies, 50, color='b', alpha=0.5)
        ax.set_xlabel('$E$')
        ax.set_ylabel('ref DOS($E$)', color='b')
        ax2 = ax.twinx()
        ax2.hist(eigen_energies, 50, color='r', alpha=0.5)
        ax2.set_ylabel('DOS($E$)', color='r')
        fig.tight_layout()
        fig.savefig(f'{dir_name}{file_name}_dos.pdf')
        plt.close(fig)


def plot_eigen_energies(dir_name: str, file_name: str, eigen_energies: np.ndarray, reference_eigen_energies: np.ndarray = None):
    if reference_eigen_energies is None:
        plt.figure(dpi=300)
        plt.plot(eigen_energies, color='b')
        plt.xlabel('$n$')
        plt.ylabel('$E_n$')
        plt.tight_layout()
        plt.savefig(f'{dir_name}{file_name}_energies.pdf')
        plt.close()
    else:
        plt.figure(dpi=300)
        plt.plot(np.abs(eigen_energies - reference_eigen_energies), color='r')
        plt.xlabel('$n$')
        plt.ylabel(r'$\Delta E_n$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'{dir_name}{file_name}_energies.pdf')
        plt.close()    


def _test(file_name: str, reference_eigen_energies: np.ndarray = None, **kwargs):
    block_exact_diagonalization('test/data/', file_name, **kwargs)
    eigen_energies: dict = read_eigen_energies('test/data/', file_name)
    print(eigen_energies.keys())
    eigen_energies = np.hstack([vals for vals in eigen_energies.values()])
    print(eigen_energies.shape)
    eigen_energies.sort()
    print(eigen_energies[:10])
    plot_dos('test/plots/', file_name, eigen_energies, reference_eigen_energies)
    plot_eigen_energies('test/plots/', file_name, eigen_energies, reference_eigen_energies)
    return eigen_energies


def read_eigen_energies(dir_name: str, file_name: str):
    with h5py.File(f'{dir_name}{file_name}_output.h5', 'r') as file:
        group = file['data']
        eigen_energies = {}
        _read_eigen_energies(group, eigen_energies)
    return eigen_energies


def _read_eigen_energies(group: h5py.Group, eigen_energies: dict):
    if 'subspaces' in group:
        for subspace in group['subspaces']:
            _read_eigen_energies(group[f'subspaces/{subspace}'], eigen_energies)
    else:
        sym = '('
        if 'n_tot' in group['param']:
            N = group['param/n_tot'][()]
            sym += f'{N}|'
        if 'n_tot_parity' in group['param']:
            Z2 = group['param/n_tot_parity'][()]
            sym += f'{Z2}|'
        if 'crystal_momentum' in group['param']:
            K = group['param/crystal_momentum'][()]
            sym += f'{K}|'
        if 'reflection_parity' in group['param']:
            P = group['param/reflection_parity'][()]
            sym += f'{P}|'
        if len(sym) > 1:
            sym = sym[:-1] + ')'
        else:
            sym += ')'
        eigen_energies[sym] = group['spectrum/all_vals'][()]


def block_exact_diagonalization(
        dir_name: str,
        file_name: str,
        L: int,
        M: int,
        terms: dict,
        space: str = 'full',
        sym: str = None,
        N: int = None,
        Z2: int = None,
        K: int = None,
        P: int = None
):
    with h5py.File(f'{dir_name}{file_name}_output.h5', 'w') as file:
        group = file.create_group('data')
        hs = bh.DecomposedHilbertSpace(num_sites=L, n_max=M, space=space, sym=sym, n_tot=N, n_tot_parity=Z2, crystal_momentum=K, reflection_parity=P)
        _block_exact_diagonalization(file, group, hs, terms)


def _block_exact_diagonalization(
        file: h5py.File,
        group: h5py.Group,
        hs: bh.DecomposedHilbertSpace,
        terms: dict
):
    if hs.subspaces is not None:
        subspaces = group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            if subspace.dim > 0:
                _block_exact_diagonalization(
                    file,
                    subspaces.create_group(f'{i:04d}'),
                    subspace,
                    terms
                )
    else:
        param = group.create_group('param')
        param.create_dataset('num_sites', data=hs.num_sites)
        param.create_dataset('n_max', data=hs.n_max)
        param.create_dataset('space', data=hs.space)
        if hs.sym is not None:
            param.create_dataset('sym', data=hs.sym)
        if hs.n_tot is not None:
            param.create_dataset('n_tot', data=hs.n_tot)
        if hs.n_tot_parity is not None:
            param.create_dataset('n_tot_parity', data=hs.n_tot_parity)
        if hs.crystal_momentum is not None:
            param.create_dataset('crystal_momentum', data=hs.crystal_momentum)
        if hs.reflection_parity is not None:
            param.create_dataset('reflection_parity', data=hs.reflection_parity)
        file.flush()

        spectrum = group.create_group('spectrum')
        if hs.space in {'PK', 'PKN', 'PKZ2'}:
            hamiltonians = HAMILTONIAN_DICT['PK']
        elif hs.space in ('K', 'KN', 'KZ2'):
            hamiltonians = HAMILTONIAN_DICT['K']
        else:
            hamiltonians = HAMILTONIAN_DICT['None']
        hamiltonian = np.sum([rate * hamiltonians[term](hs) for term, rate in terms.items()], axis=0)
        vals = np.linalg.eigvalsh(hamiltonian)
        spectrum.create_dataset('dim', data=hs.dim)
        spectrum.create_dataset('all_vals', data=vals)
        file.flush()


def test(space, L, M, terms):
    file_name = f'L={L}_M={M}_'
    for term, rate in terms.items():
        file_name += f'{term}={rate}_'
    print(file_name)
    if space == 'full':
        reference_eigen_energies = _test(f'{file_name}space=full_sym=None', L=L, M=M, terms=terms)
        _test(f'{file_name}space=full_sym=N', reference_eigen_energies, L=L, M=M, terms=terms, sym='N')
        _test(f'{file_name}space=full_sym=Z2', reference_eigen_energies, L=L, M=M, terms=terms, sym='Z2')
        _test(f'{file_name}space=full_sym=K', reference_eigen_energies, L=L, M=M, terms=terms, sym='K')
        _test(f'{file_name}space=full_sym=KN', reference_eigen_energies, L=L, M=M, terms=terms, sym='KN')
        _test(f'{file_name}space=full_sym=KZ2', reference_eigen_energies, L=L, M=M, terms=terms, sym='KZ2')
        _test(f'{file_name}space=full_sym=PK', reference_eigen_energies, L=L, M=M, terms=terms, sym='PK')
        _test(f'{file_name}space=full_sym=PKN', reference_eigen_energies, L=L, M=M, terms=terms, sym='PKN')
        _test(f'{file_name}space=full_sym=PKZ2', reference_eigen_energies, L=L, M=M, terms=terms, sym='PKZ2')
    elif space == 'N':
        reference_eigen_energies = _test(f'{file_name}space=N_N={L}_sym=None', L=L, M=M, terms=terms, space='N', N=L)
        _test(f'{file_name}space=N_N={L}_sym=KN', reference_eigen_energies, L=L, M=M, terms=terms, space='N', sym='KN', N=L)
        _test(f'{file_name}space=N_N={L}_sym=PKN', reference_eigen_energies, L=L, M=M, terms=terms, space='N', sym='PKN', N=L)
        reference_eigen_energies = _test(f'{file_name}space=KN_N={L}_K={0}_sym=None', L=L, M=M, terms=terms, space='KN', N=L, K=0)
        _test(f'{file_name}space=KN_N={L}_K={0}_sym=PKN', reference_eigen_energies, L=L, M=M, terms=terms, space='KN', sym='PKN', N=L, K=0)
    elif space == 'Z2':
        reference_eigen_energies = _test(f'{file_name}space=Z2_Z2={1}_sym=None', L=L, M=M, terms=terms, space='Z2', Z2=1)
        _test(f'{file_name}space=Z2_Z2={1}_sym=KZ2', reference_eigen_energies, L=L, M=M, terms=terms, space='Z2', sym='KZ2', Z2=1)
        _test(f'{file_name}space=Z2_Z2={1}_sym=PKZ2', reference_eigen_energies, L=L, M=M, terms=terms, space='Z2', sym='PKZ2', Z2=1)
        reference_eigen_energies = _test(f'{file_name}space=KZ2_Z2={1}_K={0}_sym=None', L=L, M=M, terms=terms, space='KZ2', Z2=1, K=0)
        _test(f'{file_name}space=KZ2_Z2={1}_K={0}_sym=PKZ2', reference_eigen_energies, L=L, M=M, terms=terms, space='KZ2', sym='PKZ2', Z2=1, K=0)
    elif space == 'K':
        reference_eigen_energies = _test(f'{file_name}space=K_K={0}_sym=None', L=L, M=M, terms=terms, space='K', K=0)
        _test(f'{file_name}space=K_K={0}_sym=PK', reference_eigen_energies, L=L, M=M, terms=terms, space='K', sym='PK', K=0)
    else:
        raise ValueError("Value of `space` must be in `{'full', 'N', 'Z2', 'K'}`")


if __name__ == '__main__':
    main()
