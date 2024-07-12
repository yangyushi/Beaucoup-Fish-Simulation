import os
import numpy as np
from numba import njit, prange
from tqdm import tqdm

N_FISH = 50
N_REPEAT = 3
N_PRE_STEPS = 0
N_SAMPLE_STEPS = 10000

dt = 0.067
noise_rot = 1e-2  # rotational noise
noise_trans = 0  # translational noise
v0 = 1  # fish speed
radius = 66.7  # boundary

at_str = 1.0  # attraction
al_str = 1.0  # align
hyd_str = 1.0  # hydrodynamic
ext_str = 3.0  # external force
max_rot = np.pi / 6
ap = 8.1 * 10**-3  # for external/wall force

xmin = 1       # minimum of attraction potential
lrep = 0.67  # length-scale of repuslion
latt = 5    # attraction length scale
catt = 3.0  # attraction coefficien
crep = catt * np.exp((1/lrep - 1/latt) * xmin)  # repulsion coefficient

epsilon = 0.3  # for vision cone

lal = 1.33   # alignment length scale
cal = 0.075  # alignment coefficient

# hydrodynamic
lhyd = 1.33    # hydrodynamic length scale, dummy
chyd = 0.25  # coefficient for hydrodynamic interaction


@njit(parallel=True)
def get_force(x, y, phi, n):
    f_attr = np.zeros((n, ))
    f_align = np.zeros((n,))
    f_hydro = np.zeros((n,))

    for i in prange(n):
        for j in range(n):
            if i != j:
                x_ij = x[i] - x[j]
                y_ij = y[i] - y[j]
                d_ij = np.sqrt(x_ij**2 + y_ij**2)
                d_phi = phi[j] - phi[i]

                cvis_an = -(
                    x_ij * np.cos(phi[i]) + y_ij * np.sin(phi[i])
                ) / d_ij

                vis_ij = (1 + epsilon * cvis_an)
                theta_ij = np.arctan2(-y_ij, -x_ij)

                f_attr[i] += (
                    crep * np.exp(-d_ij / lrep) - catt * np.exp(-d_ij / latt)
                ) * np.sin(phi[i] - theta_ij) * vis_ij

                f_align[i] += cal * np.exp(
                        -d_ij / lal
                    ) * np.sin(d_phi) * vis_ij

                f_hydro[i] += chyd * (d_ij ** -2.0) * np.sin(
                    2.0 * theta_ij - phi[i] - phi[j]
                )
    return f_attr, f_align, f_hydro


def wrap_to_pi(x):
    tmp = x / (np.pi * 2)
    tmp -= np.rint(tmp)
    mask = tmp > 0.5
    tmp[mask] = tmp[mask] - 1.0
    mask = tmp < -0.5
    tmp[mask] = tmp[mask] + 1.0
    return tmp * np.pi * 2


def advance(p, phi):
    """ advance one time point, modyinf inputs inplace

    Args:
        p: positions, shape (n, 2)
        v: velocities, shape (n, 2)
        phi: orientations, shape (n, 1)
    """
    n = p.shape[0]
    x, y = p.T
    r = np.linalg.norm(p, axis=1)

    f_wall = np.exp(
        -ap * (r - radius) ** 4
    ) * (x * np.sin(phi) - y * np.cos(phi)) / r

    f_attr, f_align, f_hydro = get_force(x, y, phi, n)

    d_orient = sum([
        np.sqrt(2 * noise_rot * dt) * np.random.normal(0, 1, n),
        ext_str * f_wall * dt,
        at_str * f_attr * dt,
        al_str * f_align * dt,
        hyd_str * f_hydro * dt,
    ])

    d_orient = wrap_to_pi(d_orient)

    d_orient[d_orient > max_rot] = max_rot
    d_orient[d_orient < -max_rot] = -max_rot

    phi += d_orient
    phi[:] = phi % (2 * np.pi)

    noise = np.sqrt(2.0 * noise_trans * dt) * \
        np.random.normal(0, 1, (2, n))

    p[:, 0] += dt * v0 * np.cos(phi) + noise[0]
    p[:, 1] += dt * v0 * np.sin(phi) + noise[1]


def save_xyz(filename, positions, orientations, frame):
    """
    Dump the movie as xyz files. Particle labels indicate the IDs.

    Args:
        filename (str): the name of the xyz file
        positions (np.ndarray): shape (n, 2)
        orientations (np.ndarray): shape (n, 1)
    """
    n, dim = positions.shape
    labels = np.arange(n)[:, None]
    result = np.concatenate(
        (labels, positions, orientations[:, None]), axis=1
    )
    with open(filename, 'a') as f:
        np.savetxt(
            f, result,
            delimiter='\t',
            fmt="\t".join(
                ['%d\t%.8e'] + ['%.8e' for i in range(dim)]
            ),
            comments='',
            header='%s\nframe %s' % (n, frame)
        )


if __name__ == "__main__":
    out_dir = "trajs-simulate-cpu"

    if out_dir not in os.listdir("."):
        os.mkdir(out_dir)

    for ir in range(N_REPEAT):
        out_file = os.path.join(
            out_dir,
            f"fish-{N_FISH:02}-repeat-{ir:02}.xyz"
        )

        # erase the existing xyz file
        f = open(out_file, "w")
        f.close()

        pos = np.array([
            radius / 2 * np.random.uniform(-0.5, 0.5, N_FISH),
            radius / 2 * np.random.uniform(-0.5, 0.5, N_FISH),
        ]).T  # (n, dim)
        phi = np.random.uniform(0, 2 * np.pi, N_FISH)

        for t in tqdm(range(N_PRE_STEPS)):
            advance(pos, phi)

        for t in tqdm(range(N_SAMPLE_STEPS)):
            advance(pos, phi)
            save_xyz(out_file, pos, phi, t+1)
