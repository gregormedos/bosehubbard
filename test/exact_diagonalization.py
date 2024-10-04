import numpy as np
import bosehubbard as bh
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': False,
                     'font.size': 18})


def main():
    test(L=6, M=2, terms=(('t', 1.0), ('U', 1.0)))
    test(L=6, M=2, terms=(('t', 1.0), ('U', 1.0), ('V1', 1.0), ('V2', 1.0)))


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


def test(**kwargs):
    L = kwargs['L']
    file_name = ''.join(f'{key}={val}_' for key, val in kwargs.items())
    bh.run_decomposed('test/data/', f'{file_name}space=full_sym=None', **kwargs)
    bh.run_decomposed('test/data/', f'{file_name}space=full_sym=N', **kwargs, sym='N')
    bh.run_decomposed('test/data/', f'{file_name}space=full_sym=K', **kwargs, sym='K')
    bh.run_decomposed('test/data/', f'{file_name}space=full_sym=KN', **kwargs, sym='KN')
    bh.run_decomposed('test/data/', f'{file_name}space=full_sym=PK', **kwargs, sym='PK')
    bh.run_decomposed('test/data/', f'{file_name}space=full_sym=PKN', **kwargs, sym='PKN')
    bh.run_decomposed('test/data/', f'{file_name}space=N_sym=None', **kwargs, space='N', N=L//2)
    bh.run_decomposed('test/data/', f'{file_name}space=N_sym=KN', **kwargs, space='N', sym='KN', N=L//2)
    bh.run_decomposed('test/data/', f'{file_name}space=N_sym=PKN', **kwargs, space='N', sym='PKN', N=L//2)
    bh.run_decomposed('test/data/', f'{file_name}space=K_sym=None', **kwargs, space='K', K=0)
    bh.run_decomposed('test/data/', f'{file_name}space=K_sym=PK', **kwargs, space='K', sym='PK', K=0)
    bh.run_decomposed('test/data/', f'{file_name}space=KN_sym=None', **kwargs, space='KN', N=L//2, K=0)
    bh.run_decomposed('test/data/', f'{file_name}space=KN_sym=PKN', **kwargs, space='KN', sym='PKN', N=L//2, K=0)
    bh.run_decomposed('test/data/', f'{file_name}space=PK_sym=None', **kwargs, space='PK', K=0, P=1)
    bh.run_decomposed('test/data/', f'{file_name}space=PKN_sym=None', **kwargs, space='PKN', sym='PKN', N=L//2, K=0, P=1)


if __name__ == '__main__':
    main()
