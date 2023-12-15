import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt
import h5py

plt.rcParams.update({'text.usetex': True,
                     'font.size': 18})


def exact_diagonalization(
        file: h5py.File,
        group: h5py.Group,
        hs: bh.DecomposedHilbertSpace,
        tunneling_rate: float,
        repulsion_strength: float,
        particle_transfer_rate: float,
        pair_production_rate: float
):
    param = group.create_group('param')
    param.create_dataset('num_sites', data=hs.num_sites)
    param.create_dataset('n_max', data=hs.n_max)
    param.create_dataset('space', data=hs.space)
    group.create_dataset('basis', data=hs.basis)
    group.create_dataset('dim', data=hs.dim)
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
        group.create_dataset('reflection_translation_periods', data=hs.nums_translations_reflection)
    file.flush()
    if hs.subspaces is not None:
        group.create_group('subspaces')
        for i, subspace in enumerate(hs.subspaces):
            exact_diagonalization(
                file,
                group.create_group(f'subspaces/{i:04d}'),
                subspace,
                tunneling_rate,
                repulsion_strength,
                particle_transfer_rate,
                pair_production_rate
            )
    else:
        spectrum = group.create_group('spectrum')
        if hs.space in ('PK', 'PKN'):
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pk()
            hamiltonian_interaction = hs.op_hamiltonian_interaction_k()
            hamiltonian_annihilate_create = hs.op_hamiltonian_annihilate_create_pk()
            hamiltonian_annihilate_create_pair = hs.op_hamiltonian_annihilate_create_pair_pk()
        elif hs.space in ('K', 'KN'):
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_k()
            hamiltonian_interaction = hs.op_hamiltonian_interaction_k()
            hamiltonian_annihilate_create = hs.op_hamiltonian_annihilate_create_k()
            hamiltonian_annihilate_create_pair = hs.op_hamiltonian_annihilate_create_pair_k()
        else:
            hamiltonian_tunnel = hs.op_hamiltonian_tunnel_pbc()
            hamiltonian_interaction = hs.op_hamiltonian_interaction()
            hamiltonian_annihilate_create = hs.op_hamiltonian_annihilate_create()
            hamiltonian_annihilate_create_pair = hs.op_hamiltonian_annihilate_create_pair_pbc()
        hamiltonian = (- tunneling_rate * hamiltonian_tunnel
                       + repulsion_strength * hamiltonian_interaction
                       + particle_transfer_rate * hamiltonian_annihilate_create
                       + pair_production_rate * hamiltonian_annihilate_create_pair)
        eigen_energies, eigen_states = np.linalg.eigh(hamiltonian)
        spectrum.create_dataset('eigen_energies', data=eigen_energies)
        spectrum.create_dataset('eigen_states', data=eigen_states)
        file.flush()


def get_eigen_energies(group: h5py.Group, eigen_energies: np.ndarray, counter: list):
    if 'subspaces' in group:
        for subspace_name in group['subspaces']:
            get_eigen_energies(group[f'subspaces/{subspace_name}'], eigen_energies, counter)
    else:
        sub_space = group['param/space'][()].decode()
        if 'K' in sub_space:
            sub_dim = group['representative_dim'][()]
        else:
            sub_dim = group['dim'][()]
        sub_eigen_energies = group['spectrum/eigen_energies'][()]
        j = counter[0]
        eigen_energies[j: j + sub_dim] = sub_eigen_energies
        counter[0] += sub_dim


def plot_dos(file_name: str, eigen_energies: np.ndarray, reference_eigen_energies: np.ndarray = None):
    if reference_eigen_energies is None:
        plt.figure(dpi=300)
        plt.hist(eigen_energies, 50, color='b')
        plt.xlabel('$E$')
        plt.ylabel('DOS($E$)')
        plt.tight_layout()
        plt.savefig(f'test/{file_name}_dos.pdf')
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
        fig.savefig(f'test/{file_name}_dos.pdf')
        plt.close(fig)


def plot_eigen_energies(file_name: str, eigen_energies: np.ndarray, reference_eigen_energies: np.ndarray = None):
    if reference_eigen_energies is None:
        plt.figure(dpi=300)
        plt.plot(eigen_energies, color='b')
        plt.xlabel('$n$')
        plt.ylabel('$E_n$')
        plt.tight_layout()
        plt.savefig(f'test/{file_name}_energies.pdf')
        plt.close()
    else:
        plt.figure(dpi=300)
        plt.plot(np.abs(eigen_energies - reference_eigen_energies), color='r')
        plt.xlabel('$n$')
        plt.ylabel(r'$\Delta E_n$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'test/{file_name}_energies.pdf')
        plt.close()


def run(
        file_name: str,
        t: float = 1.0,
        U: float = 1.0,
        V1: float = 0.0,
        V2: float = 0.0,
        L: int = 8,
        M: int = 2,
        space: str = 'full',
        sym: str = None,
        N: int = None,
        K: int = None,
        P: int = None,
        reference_eigen_energies: np.ndarray = None
):
    with h5py.File(f'test/{file_name}.h5', 'w') as file:
        group = file.create_group('data')
        hs = bh.DecomposedHilbertSpace(num_sites=L, n_max=M, space=space, sym=sym, n_tot=N, crystal_momentum=K, reflection_parity=P)
        exact_diagonalization(file, group, hs, tunneling_rate=t, repulsion_strength=U, particle_transfer_rate=V1, pair_production_rate=V2)
    with h5py.File(f'test/{file_name}.h5', 'r') as file:
        group = file['data']
        basis = group['basis'][()]
        dim = group['dim'][()]
        findstate = dict()
        for a in range(dim):
            findstate[tuple(basis[a])] = a
        if 'K' in space:
            representative_dim = group['representative_dim'][()]
            eigen_energies = np.empty(representative_dim, dtype=float)
        else:
            eigen_energies = np.empty(dim, dtype=float)
        counter = [0]
        get_eigen_energies(group, eigen_energies, counter)
        eigen_energies.sort()
        plot_dos(file_name, eigen_energies, reference_eigen_energies)
        plot_eigen_energies(file_name, eigen_energies, reference_eigen_energies)

        return eigen_energies


def main(**kwargs):
    L = kwargs['L']
    file_name = ''.join(f'{key}={val}_' for key, val in kwargs.items())
    reference_eigen_energies = run(f'{file_name}space=full_sym=None', **kwargs)
    run(f'{file_name}space=full_sym=N', **kwargs, sym='N', reference_eigen_energies=reference_eigen_energies)
    run(f'{file_name}space=full_sym=K', **kwargs, sym='K', reference_eigen_energies=reference_eigen_energies)
    run(f'{file_name}space=full_sym=KN', **kwargs, sym='KN', reference_eigen_energies=reference_eigen_energies)
    run(f'{file_name}space=full_sym=PK', **kwargs, sym='PK', reference_eigen_energies=reference_eigen_energies)
    run(f'{file_name}space=full_sym=PKN', **kwargs, sym='PKN', reference_eigen_energies=reference_eigen_energies)
    reference_eigen_energies = run(f'{file_name}space=N_sym=None', **kwargs, space='N', N=L//2)
    run(f'{file_name}space=N_sym=KN', **kwargs, space='N', sym='KN', N=L//2, reference_eigen_energies=reference_eigen_energies)
    run(f'{file_name}space=N_sym=PKN', **kwargs, space='N', sym='PKN', N=L//2, reference_eigen_energies=reference_eigen_energies)
    reference_eigen_energies = run(f'{file_name}space=K_sym=None', **kwargs, space='K', K=0)
    run(f'{file_name}space=K_sym=PK', **kwargs, space='K', sym='PK', K=0, reference_eigen_energies=reference_eigen_energies)
    reference_eigen_energies = run(f'{file_name}space=KN_sym=None', **kwargs, space='KN', N=L//2, K=0)
    run(f'{file_name}space=KN_sym=PKN', **kwargs, space='KN', sym='PKN', N=L//2, K=0, reference_eigen_energies=reference_eigen_energies)
    run(f'{file_name}space=PK_sym=None', **kwargs, space='PK', K=0, P=1)
    run(f'{file_name}space=PKN_sym=None', **kwargs, space='PKN', sym='PKN', N=L//2, K=0, P=1)


if __name__ == '__main__':
    main(t=1.0, U=1.0, V1=1.0, V2=0.0, L=12, M=1)
    main(t=1.0, U=1.0, V1=1.0, V2=0.0, L=8, M=2)
