import numpy as np
from scipy import io


def save_xyz(filename, positions, frame):
    """
    Dump the movie as xyz files. Particle labels indicate the IDs.

    Args:
        filename (str): the name of the xyz file
        frame (np.ndarray): shape (n, dim)
    """
    n, dim = positions.shape
    labels = np.arange(n)[:, None]
    result = np.concatenate((labels, positions), axis=1)
    with open(filename, 'a') as f:
        np.savetxt(
            f, result,
            delimiter='\t',
            fmt="\t".join(
                ['%d\t%.8e'] + ['%.8e' for i in range(dim - 1)]
            ),
            comments='',
            header='%s\nframe %s' % (n, frame)
        )


data = io.loadmat("result.mat")

X = data['xol']  # (time, n_fish)
Y = data['yol']  # (time, n_fish)
P = data['phi_ol']  # (time, n_fish)


output = "result.xyz"

for t in range(X.shape[0]):
    pos = np.array((X[t], Y[t], P[t])).T
    save_xyz(output, pos, t+1)
