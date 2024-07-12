import os
from tqdm import tqdm
import numpy as np
import cupy as cp


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


def get_force(x, y, phi, n):
    X = cp.repeat(x[:, None], n, axis=1)
    Y = cp.repeat(y[:, None], n, axis=1)
    P = cp.repeat(phi[:, None], n, axis=1)

    x_ij = X - X.T
    y_ij = Y - Y.T
    d_phi = P.T - P
    d_ij = cp.sqrt(x_ij ** 2 + y_ij ** 2)

    cvis_an = -(x_ij * cp.cos(P) + y_ij * cp.sin(P)) / d_ij

    cp.fill_diagonal(cvis_an, -cp.inf)

    vis_ij = (1 + epsilon * cvis_an)
    theta_ij = cp.arctan2(-y_ij, -x_ij)

    f_attr = (
        crep * cp.exp(-d_ij / lrep) -
        catt * cp.exp(-d_ij / latt)
    ) * cp.sin(P - theta_ij) * vis_ij

    f_align = cal * cp.exp(-d_ij / lal) * cp.sin(d_phi) * vis_ij

    f_hydro = chyd * (d_ij ** -2.0) * cp.sin(
        2.0 * theta_ij - P - P.T
    )
    cp.fill_diagonal(f_attr, 0.0)
    cp.fill_diagonal(f_align, 0.0)
    cp.fill_diagonal(f_hydro, 0.0)

    return (
        cp.sum(f_attr, axis=1),
        cp.sum(f_align, axis=1),
        cp.sum(f_hydro, axis=1),
    )


def wrap_to_pi(x):
    tmp = x / (cp.pi * 2)
    tmp -= cp.rint(tmp)
    mask = tmp > 0.5
    tmp[mask] = tmp[mask] - 1.0
    mask = tmp < -0.5
    tmp[mask] = tmp[mask] + 1.0
    return tmp * cp.pi * 2


def advance(p, phi):
    """ advance one time point, modyinf inputs inplace

    Args:
        p: positions, shape (n, 2)
        v: velocities, shape (n, 2)
        phi: orientations, shape (n, 1)
    """
    n = p.shape[0]
    x, y = p.T
    r = cp.linalg.norm(p, axis=1)

    f_wall = cp.exp(
        -ap * (r - radius) ** 4
    ) * (x * cp.sin(phi) - y * cp.cos(phi)) / r

    f_attr, f_align, f_hydro = get_force(
        cp.asarray(x),
        cp.asarray(y),
        cp.asarray(phi),
        n
    )

    d_orient = sum([
        cp.sqrt(2 * noise_rot * dt) * cp.random.normal(0, 1, n),
        ext_str * f_wall * dt,
        at_str * f_attr * dt,
        al_str * f_align * dt,
        hyd_str * f_hydro * dt,
    ])

    d_orient = wrap_to_pi(d_orient)

    d_orient[d_orient > max_rot] = max_rot
    d_orient[d_orient < -max_rot] = -max_rot

    phi += d_orient
    phi[:] = phi % (2 * cp.pi)

    noise = cp.sqrt(2.0 * noise_trans * dt) * cp.random.normal(0, 1, (2, n))

    p[:, 0] += dt * v0 * cp.cos(phi) + noise[0]
    p[:, 1] += dt * v0 * cp.sin(phi) + noise[1]


def save_xyz(filename, positions, orientations, frame):
    """
    Dump the movie as xyz files. Particle labels indicate the IDs.

    Args:
        filename (str): the name of the xyz file
        positions (np.ndarray): shape (n, 2)
        orientations (np.ndarray): shape (n)
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
    out_dir = "trajs-simulate-gpu"

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

        pos = cp.asarray(
            np.array([
                radius / 2 * np.random.uniform(-0.5, 0.5, N_FISH),
                radius / 2 * np.random.uniform(-0.5, 0.5, N_FISH),
            ]).T  # (n, dim)
        )
        phi = cp.asarray(
            np.random.uniform(0, np.pi * 2, N_FISH)
        )

        for t in tqdm(range(N_PRE_STEPS)):
            advance(pos, phi)

        trajs = []
        for t in tqdm(range(N_SAMPLE_STEPS)):
            advance(pos, phi)
            save_xyz(out_file, pos.get(), phi.get(), t+1)
