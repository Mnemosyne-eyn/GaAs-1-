import numpy as np
from scipy.linalg import eigh


# 1. Loading graph data -------------------------------------------------------


def load_graph(path="graph_data.npz"):
    graph_data = np.load(path, allow_pickle=True)["graph"].item()
    data = list(graph_data.values())[0]
    return data


# 2. Build 3D k-mesh ----------------------------------------------------------


def build_kmesh(Nk1=10, Nk2=10, Nk3=10):
    k1 = np.linspace(0, 1, Nk1, endpoint=False)
    k2 = np.linspace(0, 1, Nk2, endpoint=False)
    k3 = np.linspace(0, 1, Nk3, endpoint=False)
    return k1, k2, k3


# 3. Solve H(k) ψ = E S(k) ψ and return S(k) ---------------------------------


def solve_eigenvectors_and_S_at_k(data, kfrac, eps=1e-8):
    Hon = data.Hon.numpy()
    Hoff = data.Hoff.numpy()
    Son = data.Son.numpy()
    Soff = data.Soff.numpy()
    nbr_shift = data.nbr_shift.numpy()
    edge_index = data.edge_index.numpy()
    cell = data.cell.numpy().reshape(3, 3)

    natoms = data.pos.shape[0]
    nao = int(np.sqrt(Hon.shape[1]))
    norb = natoms * nao

    Hon = Hon.reshape(natoms, nao, nao)
    Son = Son.reshape(natoms, nao, nao)
    Hoff = Hoff.reshape(-1, nao, nao)
    Soff = Soff.reshape(-1, nao, nao)

    # fractional -> cartesian k
    lat_inv = np.linalg.inv(cell).T
    kcart = kfrac @ lat_inv

    HK = np.zeros((natoms, natoms, nao, nao), dtype=np.complex128)
    SK = np.zeros((natoms, natoms, nao, nao), dtype=np.complex128)

    # on-site blocks
    for a in range(natoms):
        HK[a, a] += Hon[a]
        SK[a, a] += Son[a]

    # phase factors for hoppings
    phase = np.exp(2j * np.pi * np.sum(nbr_shift * kcart[None, :], axis=-1))

    for e in range(Hoff.shape[0]):
        u, v = edge_index[:, e]
        HK[u, v] += phase[e] * Hoff[e]
        SK[u, v] += phase[e] * Soff[e]

    # reshape to (norb, norb)
    HK = np.swapaxes(HK, 1, 2).reshape(norb, norb)
    SK = np.swapaxes(SK, 1, 2).reshape(norb, norb)

    # symmetrize S and regularize slightly
    SK = (SK + SK.conj().T) / 2
    SK += eps * np.eye(norb)

    evals, psi = eigh(HK, SK)  # psi: (norb, nbands), S-orthonormal
    return evals, psi, SK


# 4. Fukui F12 and Chern per band on one kz slice (non-orthogonal) -----------


def fukui_F12_and_chern_nonorth(psi_slice, S_slice):
    """
    psi_slice[i,j,:,n] = eigenvector of band n at (kx_i, ky_j, kz_fixed)
    S_slice[i,j,:,:]   = overlap matrix S(k) at (kx_i, ky_j, kz_fixed)
    """
    Nk1, Nk2, norb, nbands = psi_slice.shape

    def sx(a):
        return np.roll(a, -1, axis=0)

    def sy(a):
        return np.roll(a, -1, axis=1)

    Ux = np.zeros((Nk1, Nk2, nbands), dtype=np.complex128)
    Uy = np.zeros_like(Ux)

    for i in range(Nk1):
        ip = (i + 1) % Nk1
        for j in range(Nk2):
            jp = (j + 1) % Nk2

            S_ij = S_slice[i, j]  # (norb, norb)

            for n in range(nbands):
                v_ij = psi_slice[i, j, :, n]
                v_ipj = psi_slice[ip, j, :, n]
                v_ijp = psi_slice[i, jp, :, n]

                # S-metric overlaps
                Ux[i, j, n] = v_ij.conj() @ (S_ij @ v_ipj)
                Uy[i, j, n] = v_ij.conj() @ (S_ij @ v_ijp)

    # normalize to U(1)
    Ux /= np.abs(Ux)
    Uy /= np.abs(Uy)

    # plaquette product
    Uxy = Ux * sx(Uy) * np.conj(sy(Ux)) * np.conj(Uy)  # (Nk1,Nk2,nbands)

    F12 = np.angle(Uxy)  # (Nk1,Nk2,nbands)
    chern = np.sum(F12, axis=(0, 1)) / (2 * np.pi)  # (nbands,)

    return F12, chern


# MAIN: 3D Fukui Chern per band vs kz ----------------------------------------


if __name__ == "__main__":
    data = load_graph("graph_data.npz")

    Nk1 = Nk2 = Nk3 = 10
    k1, k2, k3 = build_kmesh(Nk1, Nk2, Nk3)

    # determine norb once
    natoms = data.pos.shape[0]
    nao = int(np.sqrt(data.Hon.shape[1]))
    norb = natoms * nao

    chern_all = []   # list of (nbands,) per kz
    F12_all = []     # list of (Nk1,Nk2,nbands) per kz

    for kz in k3:
        psi_slice = np.zeros((Nk1, Nk2, norb, norb), dtype=np.complex128)
        S_slice = np.zeros((Nk1, Nk2, norb, norb), dtype=np.complex128)

        for i, kx in enumerate(k1):
            for j, ky in enumerate(k2):
                kfrac = np.array([kx, ky, kz])
                evals, psi, Sk = solve_eigenvectors_and_S_at_k(data, kfrac)
                psi_slice[i, j] = psi   # (norb, nbands)
                S_slice[i, j] = Sk      # (norb, norb)

        F12_kz, chern_kz = fukui_F12_and_chern_nonorth(psi_slice, S_slice)

        chern_all.append(chern_kz)
        F12_all.append(F12_kz)

        print(f"kz={kz:.3f}, Chern per band (rounded):", np.rint(chern_kz).astype(int))

    chern_all = np.array(chern_all)   # (Nk3, nbands)
    np.savetxt("chern_slices_per_band.txt", chern_all, fmt="%.6f")

    F12_all = np.array(F12_all)       # (Nk3, Nk1, Nk2, nbands)
    np.save("F12_all_kz.npy", F12_all)

    print("Saved chern_slices_per_band.txt and F12_all_kz.npy")
