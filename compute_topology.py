import numpy as np
import torch
import math
from utils import basis_def_26  # and num_valence_* from your project

# 
graph_data_path  = "/Users/joshibhumika/Downloads/GaAs 12/graph_data.npz"
hamiltonian_path = "/Users/joshibhumika/Downloads/GaAs 12/prediction_hamiltonian.npy"
output_dir       = "/Users/joshibhumika/Downloads/GaAs 12/"
nao_max          = 26
Nk               = 10
eps_reg          = 1e-6
Ham_type         = "openmx"   # or "abacus"
# 


def load_graph(path):
    graph_data = np.load(path, allow_pickle=True)["graph"].item()
    return list(graph_data.values())[0]


def build_orbital_mask(species):
    basis_definition = np.zeros((99, nao_max))
    for Z, idxs in basis_def_26.items():
        basis_definition[Z][idxs] = 1
    orb_mask = basis_definition[species].reshape(-1)
    return orb_mask[:, None] * orb_mask[None, :]


def load_hamiltonian(data, hamiltonian_path):
    len_H = [len(data.Hon), len(data.Hoff)]
    H = np.load(hamiltonian_path)

    idx = 0
    Hon = H[idx:idx + len_H[0]]; idx += len_H[0]
    Hoff = H[idx:idx + len_H[1]]

    return Hon.reshape(-1, nao_max, nao_max), Hoff.reshape(-1, nao_max, nao_max)


def build_H_and_S(kfrac, data, Hon, Hoff, orb_mask):
    Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
    Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
    nbr_shift = data.nbr_shift.numpy()
    edge_index = data.edge_index.numpy()
    species = data.z.numpy()
    natoms = len(species)

    HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex128)
    SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex128)

    na = np.arange(natoms)
    HK[na, na] += Hon[na]
    SK[na, na] += Son[na]

    phase = np.exp(2j * np.pi * np.sum(nbr_shift * kfrac[None, :], axis=-1))

    for e in range(Hoff.shape[0]):
        i, j = edge_index[:, e]
        HK[i, j] += phase[e] * Hoff[e]
        SK[i, j] += phase[e] * Soff[e]

    HK = np.swapaxes(HK, -2, -3).reshape(natoms * nao_max, natoms * nao_max)
    SK = np.swapaxes(SK, -2, -3).reshape(natoms * nao_max, natoms * nao_max)

    HK = HK[orb_mask > 0]
    SK = SK[orb_mask > 0]
    norbs = int(math.sqrt(HK.size))

    HK = HK.reshape(norbs, norbs)
    SK = SK.reshape(norbs, norbs)

    SK = (SK + SK.conj().T) / 2
    SK += eps_reg * np.eye(norbs)

    return HK, SK


def fukui_abelian(evecs_slice):
    Nkx, Nky, nbands, norb = evecs_slice.shape
    v = evecs_slice / np.linalg.norm(evecs_slice, axis=-1, keepdims=True)

    def sx(a): return np.roll(a, -1, axis=0)
    def sy(a): return np.roll(a, -1, axis=1)

    Ux = np.sum(np.conj(v) * sx(v), axis=-1)
    Uy = np.sum(np.conj(v) * sy(v), axis=-1)

    Ux /= np.abs(Ux)
    Uy /= np.abs(Uy)

    Uxy = Ux * sx(Uy) * np.conj(sy(Ux)) * np.conj(Uy)
    F12 = np.angle(Uxy)

    chern = np.sum(F12, axis=(0, 1)) / (2 * np.pi)
    return F12, chern


def fukui_nonabelian(evecs_slice, n_occ):
    Nkx, Nky, norb, _ = evecs_slice.shape
    psi = evecs_slice[:, :, :, :n_occ]

    def sx(a): return np.roll(a, -1, axis=0)
    def sy(a): return np.roll(a, -1, axis=1)

    Ux = np.zeros((Nkx, Nky), dtype=np.complex128)
    Uy = np.zeros((Nkx, Nky), dtype=np.complex128)

    for i in range(Nkx):
        for j in range(Nky):
            Mx = psi[i, j].conj().T @ sx(psi)[i, j]
            My = psi[i, j].conj().T @ sy(psi)[i, j]
            Ux[i, j] = np.linalg.det(Mx)
            Uy[i, j] = np.linalg.det(My)

    Ux /= np.abs(Ux)
    Uy /= np.abs(Uy)

    Uxy = Ux * sx(Uy) * np.conj(sy(Ux)) * np.conj(Uy)
    F12 = np.angle(Uxy)
    chern = np.sum(F12) / (2 * np.pi)
    return F12, chern


def compute_n_occ(data, Ham_type):
    species = data.z.numpy()  # atomic numbers
    num_val = np.zeros((99,), dtype=int)

    # fill from your existing dictionaries
    if Ham_type.lower() == "openmx":
        from utils import num_valence_openmx as num_valence
    elif Ham_type.lower() == "abacus":
        from utils import num_valence_abacus as num_valence
    else:
        raise NotImplementedError

    for Z, v in num_valence.items():
        num_val[Z] = v

    num_electrons = int(np.sum(num_val[species]))
    # spinless model → 1 electron per band
    n_occ = num_electrons
    return n_occ


def main():
    data = load_graph(graph_data_path)
    species = data.z.numpy()
    orb_mask = build_orbital_mask(species)
    Hon, Hoff = load_hamiltonian(data, hamiltonian_path)

    n_occ = compute_n_occ(data, Ham_type)
    print("Detected n_occ =", n_occ)

    kx = np.linspace(0, 1, Nk, endpoint=False)
    ky = np.linspace(0, 1, Nk, endpoint=False)
    kz = np.linspace(0, 1, Nk, endpoint=False)

    F12_all_abel = []
    chern_all_abel = []
    chern_occ_all = []

    for iz, kz_val in enumerate(kz):
        print(f"=== kz slice {iz}/{Nk-1}, kz={kz_val:.3f} ===")
        evecs_slice = []

        for ix, kx_val in enumerate(kx):
            row = []
            for iy, ky_val in enumerate(ky):
                kfrac = np.array([kx_val, ky_val, kz_val])

                HK, SK = build_H_and_S(kfrac, data, Hon, Hoff, orb_mask)

                SK_t = torch.complex(torch.tensor(SK.real), torch.tensor(SK.imag)).unsqueeze(0)
                HK_t = torch.complex(torch.tensor(HK.real), torch.tensor(HK.imag)).unsqueeze(0)

                L = torch.linalg.cholesky(SK_t)
                L_inv = torch.linalg.inv(L)
                L_t = L.conj().transpose(-1, -2)
                L_t_inv = torch.linalg.inv(L_t)

                Hs = L_inv @ HK_t @ L_t_inv
                vals, vecs = torch.linalg.eigh(Hs)

                vecs_phys = L_t_inv @ vecs
                row.append(vecs_phys.squeeze(0).cpu().numpy())

            evecs_slice.append(row)

        evecs_slice = np.array(evecs_slice)  # (Nk,Nk,norb,norb)

        F12_kz_abel, chern_kz_abel = fukui_abelian(evecs_slice)
        F12_kz_occ, chern_kz_occ = fukui_nonabelian(evecs_slice, n_occ)

        F12_all_abel.append(F12_kz_abel)
        chern_all_abel.append(chern_kz_abel)
        chern_occ_all.append(chern_kz_occ)

        print("  Abelian Chern (rounded):", np.rint(chern_kz_abel).astype(int))
        print("  Occupied-subspace Chern:", chern_kz_occ)

    F12_all_abel = np.array(F12_all_abel)
    chern_all_abel = np.array(chern_all_abel)
    chern_occ_all = np.array(chern_occ_all)

    np.savetxt(output_dir + "chern_slices_abelian.txt", chern_all_abel, fmt="%.6f")
    np.savetxt(output_dir + "chern_slices_occ.txt", chern_occ_all, fmt="%.6f")

    chern_3D_abel = chern_all_abel.mean(axis=0)
    np.savetxt(output_dir + "chern_3D_abelian.txt", chern_3D_abel, fmt="%.6f")

    F12_int_abel = F12_all_abel.mean(axis=0)
    np.save(output_dir + "F12_3D_integrated_abelian.npy", F12_int_abel)

    print("Saved chern_slices_abelian.txt, chern_slices_occ.txt, chern_3D_abelian.txt, F12_3D_integrated_abelian.npy")


if __name__ == "__main__":
    main()
