import os

import numpy as np


def read_dftd4(fname: str) -> np.ndarray:
    """Get the C6 parameter from the DFT-D4 output.

    Args:
        fname (str): Filename of the dftd4 output.

    Returns:
        np.ndarray: C6 parameter.
    """

    with open(fname, "r") as file:
        lines = file.readlines()

    for idx, line in enumerate(lines):
        if "Atomic properties (in atomic units):" in line:
            break

    init_idx = idx + 4

    for idx, line in enumerate(lines[init_idx:]):
        if "---" in line:
            break

    final_idx = idx + init_idx

    c6_params = []
    for line in lines[init_idx:final_idx]:
        c6_params.append(line.split()[-2])

    return np.array(c6_params, dtype=np.float64)


def read_txt_file(fname: str) -> list:
    """Reads a txt file and returns lines.

    Args:
        fname (str): Name of the file.

    Returns:
        list: Lines of the file.
    """
    if fname is None:
        return []

    if os.path.isfile(fname):
        with open(f"{fname}", "r") as f:
            lines = f.readlines()
        f.close()
        msg = f"Filenames are read from {fname}"
        print(msg)

        for idx, line in enumerate(lines):
            lines[idx] = line[:-1]
        return lines

    return []


def read_xyz(file: str) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz file.

    Args:
        file (str): Input xyz file.

    Returns:
        tuple: A tuple containing element symbols and coordinates.
    """
    elms, coords = [], []
    with open(file) as f:
        _ = f.readline()  # Skip nat
        next(f)  # Skip comment line
        for line in f:
            elms.append(line.split()[0])
            coords.append([float(a) for a in line.split()[1:]])
    return np.array(elms, dtype=str), np.array(coords)
