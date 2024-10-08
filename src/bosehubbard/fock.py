import numpy as np


def fock_lower(s_state: np.ndarray, i: int):
    """
    Return a copy of a Fock state with one lowered site.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    i : int
        Site index

    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.copy(s_state)
    t_state[i] -= 1
    return t_state


def fock_raise(s_state: np.ndarray, i: int):
    """
    Return a copy of a Fock state with one raised site.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    i : int
        Site index

    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.copy(s_state)
    t_state[i] += 1
    return t_state


def fock_translation(s_state: np.ndarray, r: int):
    """
    Return a copy of a Fock state, translated in the right direction by `r` number of sites.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    r : int
        Number of sites by which to move in the right direction
        
    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.roll(s_state, r)
    return t_state


def fock_checkstate(s_state: np.ndarray, num_sites: int, crystal_momentum: int):
    """
    Check if the given Fock state is the representative state under translations.
    If so, calculate the translation period of the representative state.
    If not, return the translation period as -1 to dismiss the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
    crystal_momentum : int
        Crystal momentum

    Returns
    -------
    translation_period : int
        Translation period of the representative state

    """
    translation_period = -1  # not representative
    t_state = np.copy(s_state)
    for i in range(1, num_sites + 1):
        t_state = np.roll(t_state, 1)
        if tuple(t_state) > tuple(s_state):
            return translation_period
        elif tuple(t_state) == tuple(s_state):
            if (crystal_momentum % (num_sites / i)) == 0:
                translation_period = i
                return translation_period
            else:
                return translation_period


def fock_representative(s_state: np.ndarray, num_sites: int):
    """
    Find the representative state under translations for the given Fock state and
    calculate the number of translations from the representative state to the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites

    Returns
    -------
    representative_state : np.ndarray
        Representative state
    num_translations : int
        Number of translations

    """
    representative_state = np.copy(s_state)
    num_translations = 0
    t_state = np.copy(s_state)
    for i in range(1, num_sites):
        t_state = np.roll(t_state, 1)
        if tuple(t_state) > tuple(representative_state):
            representative_state = np.copy(t_state)
            num_translations = i

    return representative_state, num_translations


def fock_reflection(s_state: np.ndarray):
    """
    Return a copy of a Fock state, reflected through its center.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
        
    Returns
    -------
    t_state: np.ndarray
        Transformed Fock state

    """
    t_state = np.flipud(s_state)
    return t_state


def fock_checkstate_reflection(s_state: np.ndarray, num_sites: int, crystal_momentum: int):
    """
    Check if the given Fock state is the representative state under translations and reflection.
    If so, calculate number of translations need to get from the reflection of the representative state
    back to the representative state.
    If not, return the translation period as -1 to dismiss the given Fock state.
    If the reflections are not connected under translations,
    return -1 to indicate that two copies of this representative state will be stored.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations and reflection.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
    crystal_momentum : int
        Crystal momentum

    Returns
    -------
    translation_period : int
        Translation period of the representative state
    num_translations_reflection : int
        Number of translations after reflection

    """
    translation_period = fock_checkstate(s_state, num_sites, crystal_momentum)
    num_translations_reflection = -1  # the reflections are not connected under translations
    t_state = np.flipud(s_state)
    for i in range(translation_period):
        if tuple(t_state) > tuple(s_state):
            translation_period = -1  # not representative
            return translation_period, num_translations_reflection
        elif tuple(t_state) == tuple(s_state):
            num_translations_reflection = i
            return translation_period, num_translations_reflection
        t_state = np.roll(t_state, 1)

    return translation_period, num_translations_reflection


def fock_representative_reflection(s_state: np.ndarray, num_sites: int):
    """
    Find the representative state for the reflection of the given Fock
    state and calculate the number of translations and reflections from
    the representative state to the given Fock state.

    The representative state is the highest integer tuple among all
    Fock states, linked by translations and reflection.

    Parameters
    ----------
    s_state : np.ndarray
        Starting Fock state
    num_sites : int
        Number of sites
        
    Returns
    -------
    representative_state : np.ndarray
        Representative state
    num_translations : int
        Number of translations
    num_reflections : int
        Number of reflections

    """
    representative_state, num_translations = fock_representative(s_state, num_sites)
    num_reflections = 0
    t_state = np.flipud(s_state)
    for i in range(num_sites):
        if tuple(t_state) > tuple(representative_state):
            representative_state = np.copy(t_state)
            num_translations = i
            num_reflections = 1
        t_state = np.roll(t_state, 1)

    return representative_state, num_translations, num_reflections
