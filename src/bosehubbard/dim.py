from scipy import special


def dim_full_hardcore(num_sites: int):
    """
    Calculate the full hardcore bosonic Hilbert space dimension,
    given the number of sites `num_sites`.

    Parameters
    ----------
    num_sites : int
        Number of sites

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    return 2 ** num_sites


def dim_full_softcore(num_sites: int, n_max: int):
    """
    Calculate the full softcore bosonic Hilbert space dimension,
    given the number of sites `num_sites` and the maximum site occupation number `n_max`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_max : int
        Maximum site occupation number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    return (n_max + 1) ** num_sites


def dim_nblock_hardcore(num_sites: int, n_tot: int):
    """
    Calculate the N-block hardcore bosonic Hilbert space dimension,
    given the number of sites `num_sites` and the total particle number `n_tot`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total particle number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    if num_sites < n_tot:
        return 0
    else:
        return special.comb(num_sites, n_tot, exact=True)


def dim_nblock_bosonic(num_sites: int, n_tot: int):
    """
    Calculate the N-block bosonic Hilbert space dimension,
    given the number of sites `num_sites` and the total particle number `n_tot`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total particle number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    if num_sites == 0:
        if n_tot == 0:
            return 1
        else:
            return 0
    else:
        return special.comb(num_sites + n_tot - 1, n_tot, exact=True)


def dim_nblock_softcore(num_sites: int, n_tot: int, n_max: int):
    """
    Calculate the N-block softcore bosonic Hilbert space dimension,
    given the number of sites `num_sites`, the total particle number `n_tot`
    and the maximum site occupation number `n_max`.

    Parameters
    ----------
    num_sites : int
        Number of sites
    n_tot : int
        Total particle number
    n_max : int
        Maximum site occupation number

    Returns
    -------
    dim : int
        Hilbert space dimension

    """
    if num_sites * n_max < n_tot:
        return 0
    elif num_sites == 0 and n_tot == 0:
        return 1
    else:
        dimension = 0
        for k in range(num_sites + 1):
            dimension += (
                (-1) ** k
                * special.comb(num_sites, k, exact=True)
                * special.comb(num_sites + n_tot - k * (n_max + 1) - 1, num_sites - 1, exact=True)
            )
    
    return dimension


def dim_full(num_sites: int, n_max: int):
    if n_max == 1:
        return dim_full_hardcore(num_sites)
    elif n_max > 1:
        return dim_full_softcore(num_sites, n_max)
    else:
        raise ValueError("Argument n_max of dim_full must be greater than 0 but was given " + f"{n_max}")


def dim_nblock(num_sites: int, n_tot: int, n_max: int):
    if n_max == 1:
        return dim_nblock_hardcore(num_sites, n_tot)
    elif n_max >= n_tot:
        return dim_nblock_bosonic(num_sites, n_tot)
    elif n_max > 1:
        return dim_nblock_softcore(num_sites, n_tot, n_max)
    else:
        raise ValueError("Argument n_max of dim_nblock must be greater than 0 but was given " + f"{n_max}")
