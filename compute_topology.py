import numpy as np
import torch
import math
from utils import basis_def_26  # and num_valence_* from your project
 
# ── CONFIG ──────────────────────────────────────────────────────────────────
graph_data_path  = "/Users/joshibhumika/Documents/GitHub/GaAs-1-/graph_data.npz"
hamiltonian_path = "/Users/joshibhumika/Documents/GitHub/GaAs-1-/prediction_hamiltonian.npy"
output_dir       = "/Users/joshibhumika/Documents/GitHub/GaAs-1-/"
nao_max          = 26
Nk               = 20      # increased from 10 — coarse mesh causes noisy Chern numbers
eps_reg          = 1e-6
Ham_type         = "openmx"   # or "abacus"
spinful          = False       # set True only if Hamiltonian has explicit spin doubling
# ────────────────────────────────────────────────────────────────────────────
 
 
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
    Hon  = H[idx:idx + len_H[0]]; idx += len_H[0]
    Hoff = H[idx:idx + len_H[1]]
 
    return Hon.reshape(-1, nao_max, nao_max), Hoff.reshape(-1, nao_max, nao_max)
 
 
def build_H_and_S(kfrac, data, Hon, Hoff, orb_mask):
    Son        = data.Son.numpy().reshape(-1, nao_max, nao_max)
    Soff       = data.Soff.numpy().reshape(-1, nao_max, nao_max)
    nbr_shift  = data.nbr_shift.numpy()
    edge_index = data.edge_index.numpy()
    species    = data.z.numpy()
    natoms     = len(species)
 
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
 
    # Enforce Hermiticity on both, check error
    h_err = np.max(np.abs(HK - HK.conj().T))
    s_err = np.max(np.abs(SK - SK.conj().T))
    if h_err > 1e-6 or s_err > 1e-6:
        print(f"  WARNING: Hermiticity errors — H: {h_err:.2e}, S: {s_err:.2e}")
    HK = (HK + HK.conj().T) / 2
    SK = (SK + SK.conj().T) / 2
    SK += eps_reg * np.eye(norbs)
 
    return HK, SK
 
 
def solve_generalized_eig(HK, SK):
    """
    Solve H c = E S c via Cholesky orthogonalization.
    Returns eigenvalues and eigenvectors in the ORTHONORMAL basis
    (not physical basis) — these are what the Fukui routines need.
    """
    SK_t = torch.complex(torch.tensor(SK.real), torch.tensor(SK.imag)).unsqueeze(0)
    HK_t = torch.complex(torch.tensor(HK.real), torch.tensor(HK.imag)).unsqueeze(0)
 
    L       = torch.linalg.cholesky(SK_t)
    L_inv   = torch.linalg.inv(L)
    L_t_inv = torch.linalg.inv(L.conj().transpose(-1, -2))
 
    Hs = L_inv @ HK_t @ L_t_inv
    # Symmetrize Hs to remove any tiny numerical asymmetry before eigh
    Hs = (Hs + Hs.conj().transpose(-1, -2)) / 2
 
    vals, vecs = torch.linalg.eigh(Hs)
    # vecs columns are orthonormal eigenvectors — use directly for topology
    return vals.squeeze(0).cpu().numpy(), vecs.squeeze(0).cpu().numpy()
 
 
def fix_gauge(evecs_slice):
    """
    Smooth the U(1) gauge of each eigenvector across the BZ by rotating
    each vector so its largest-magnitude component is real and positive.
    This is a simple but effective parallel-transport-free gauge fixing
    that reduces random phase jumps between neighboring k-points.
 
    evecs_slice: (Nkx, Nky, norb, nbands)
    """
    Nkx, Nky, norb, nbands = evecs_slice.shape
    out = evecs_slice.copy()
    for n in range(nbands):
        for i in range(Nkx):
            for j in range(Nky):
                v = out[i, j, :, n]
                # Find component with largest magnitude
                idx = np.argmax(np.abs(v))
                phase = v[idx] / np.abs(v[idx])
                out[i, j, :, n] = v / phase
    return out
 
 
def fukui_abelian(evecs_slice, eps=1e-14):
    """
    Abelian Fukui method — per-band Chern numbers.
    evecs_slice: (Nkx, Nky, norb, nbands)
    Orbital index = axis 2, band index = axis 3.
    """
    Nkx, Nky, norb, nbands = evecs_slice.shape
 
    # Normalize each eigenvector over the orbital axis (axis=2)
    norms = np.linalg.norm(evecs_slice, axis=2, keepdims=True)
    v = evecs_slice / np.maximum(norms, eps)
 
    # Precompute rolls — do NOT recompute inside inner loops
    vx = np.roll(v, -1, axis=0)   # v at k + x̂
    vy = np.roll(v, -1, axis=1)   # v at k + ŷ
 
    # Link variables: inner product over orbital axis → (Nkx, Nky, nbands)
    Ux = np.sum(np.conj(v) * vx, axis=2)
    Uy = np.sum(np.conj(v) * vy, axis=2)
 
    # Normalize to U(1)
    Ux /= np.maximum(np.abs(Ux), eps)
    Uy /= np.maximum(np.abs(Uy), eps)
 
    # Precompute shifted link variables
    Ux_shy = np.roll(Ux, -1, axis=1)   # Ux at k + ŷ
    Uy_shx = np.roll(Uy, -1, axis=0)   # Uy at k + x̂
 
    # Correct plaquette formula:
    # F12(k) = arg[ Ux(k) * Uy(k+x̂) * Ux(k+ŷ)^{-1} * Uy(k)^{-1} ]
    # For U(1) numbers, inverse = conjugate
    Uxy = Ux * Uy_shx * np.conj(Ux_shy) * np.conj(Uy)
    F12  = np.angle(Uxy)                             # (Nkx, Nky, nbands)
    chern = np.sum(F12, axis=(0, 1)) / (2 * np.pi)  # (nbands,)
    return F12, chern
 
 
def fukui_nonabelian(evecs_slice, n_occ, eps=1e-14):
    """
    Non-Abelian Fukui method — total Chern number of occupied subspace.
    evecs_slice: (Nkx, Nky, norb, nbands)
    Uses first n_occ columns as the occupied subspace.
    """
    Nkx, Nky, norb, nbands = evecs_slice.shape
    psi = evecs_slice[:, :, :, :n_occ]   # (Nkx, Nky, norb, n_occ)
 
    # Precompute rolled arrays ONCE — not inside the loop
    psi_x = np.roll(psi, -1, axis=0)    # psi at k + x̂
    psi_y = np.roll(psi, -1, axis=1)    # psi at k + ŷ
 
    Ux = np.zeros((Nkx, Nky), dtype=np.complex128)
    Uy = np.zeros((Nkx, Nky), dtype=np.complex128)
 
    for i in range(Nkx):
        for j in range(Nky):
            # Overlap matrices (n_occ x n_occ)
            # Valid because psi is in orthonormal basis
            Mx = psi[i, j].conj().T @ psi_x[i, j]   # (n_occ, n_occ)
            My = psi[i, j].conj().T @ psi_y[i, j]
            Ux[i, j] = np.linalg.det(Mx)
            Uy[i, j] = np.linalg.det(My)
 
    # Normalize determinants to U(1)
    Ux /= np.maximum(np.abs(Ux), eps)
    Uy /= np.maximum(np.abs(Uy), eps)
 
    # Precompute shifted determinants
    Ux_shy = np.roll(Ux, -1, axis=1)
    Uy_shx = np.roll(Uy, -1, axis=0)
 
    # Correct plaquette formula (same structure as Abelian)
    Uxy  = Ux * Uy_shx * np.conj(Ux_shy) * np.conj(Uy)
    F12  = np.angle(Uxy)
    chern = np.sum(F12) / (2 * np.pi)
    return F12, chern
 
 
def compute_n_occ(data, Ham_type, spinful=False):
    species = data.z.numpy()
    num_val = np.zeros((99,), dtype=int)
 
    if Ham_type.lower() == "openmx":
        from utils import num_valence_openmx as num_valence
    elif Ham_type.lower() == "abacus":
        from utils import num_valence_abacus as num_valence
    else:
        raise NotImplementedError(f"Unknown Ham_type: {Ham_type}")
 
    for Z, v in num_valence.items():
        num_val[Z] = v
 
    num_electrons = int(np.sum(num_val[species]))
 
    if spinful:
        n_occ = num_electrons
    else:
        if num_electrons % 2 != 0:
            raise ValueError(
                f"Odd electron count ({num_electrons}) in spinless model."
            )
        n_occ = num_electrons // 2
 
    return n_occ
 
 
def check_gap(vals_slice, n_occ, kx, ky, kz_val):
    """
    Check whether there is a clear gap between band n_occ-1 and n_occ
    across the entire kz slice. A small or zero gap invalidates Chern numbers.
    """
    gaps = []
    for ix in range(len(kx)):
        for iy in range(len(ky)):
            e_occ  = vals_slice[ix][iy][n_occ - 1]
            e_unocc = vals_slice[ix][iy][n_occ]
            gaps.append(e_unocc - e_occ)
    min_gap = min(gaps)
    if min_gap < 1e-3:
        print(f"  WARNING: min gap at kz={kz_val:.3f} is {min_gap:.2e} eV — "
              f"bands may be touching, Chern numbers unreliable.")
    else:
        print(f"  Min gap at kz={kz_val:.3f}: {min_gap:.4f} eV")
    return min_gap
 
 
def main():
    data     = load_graph(graph_data_path)
    species  = data.z.numpy()
    orb_mask = build_orbital_mask(species)
    Hon, Hoff = load_hamiltonian(data, hamiltonian_path)
 
    n_occ = compute_n_occ(data, Ham_type, spinful=spinful)
    print(f"Detected n_occ = {n_occ}  (spinful={spinful})")
 
    kx = np.linspace(0, 1, Nk, endpoint=False)
    ky = np.linspace(0, 1, Nk, endpoint=False)
    kz = np.linspace(0, 1, Nk, endpoint=False)
 
    F12_all_abel   = []
    chern_all_abel = []
    chern_occ_all  = []
 
    for iz, kz_val in enumerate(kz):
        print(f"\n=== kz slice {iz}/{Nk-1}, kz={kz_val:.3f} ===")
        evecs_slice = []
        vals_slice  = []
 
        for ix, kx_val in enumerate(kx):
            evecs_row = []
            vals_row  = []
            for iy, ky_val in enumerate(ky):
                kfrac = np.array([kx_val, ky_val, kz_val])
                HK, SK = build_H_and_S(kfrac, data, Hon, Hoff, orb_mask)
 
                assert 0 < n_occ <= HK.shape[0], (
                    f"n_occ={n_occ} out of range for norb={HK.shape[0]}"
                )
 
                vals, vecs = solve_generalized_eig(HK, SK)
                evecs_row.append(vecs)
                vals_row.append(vals)
 
            evecs_slice.append(evecs_row)
            vals_slice.append(vals_row)
 
        evecs_slice = np.array(evecs_slice)   # (Nk, Nk, norb, norb)
        print(f"  evecs_slice shape: {evecs_slice.shape}")
 
        # Check gap before computing topology
        check_gap(vals_slice, n_occ, kx, ky, kz_val)
 
        # Fix gauge to reduce random phase noise
        evecs_slice = fix_gauge(evecs_slice)
 
        F12_kz_abel, chern_kz_abel = fukui_abelian(evecs_slice)
        F12_kz_occ,  chern_kz_occ  = fukui_nonabelian(evecs_slice, n_occ)
 
        F12_all_abel.append(F12_kz_abel)
        chern_all_abel.append(chern_kz_abel)
        chern_occ_all.append(chern_kz_occ)
 
        print(f"  Abelian Chern (rounded): {np.rint(chern_kz_abel).astype(int)}")
        print(f"  Occupied-subspace Chern: {round(chern_kz_occ, 4)}")
 
    F12_all_abel   = np.array(F12_all_abel)
    chern_all_abel = np.array(chern_all_abel)
    chern_occ_all  = np.array(chern_occ_all)
 
    # Save results
    np.savetxt(output_dir + "chern_slices_abelian.txt",  chern_all_abel,  fmt="%.6f")
    np.savetxt(output_dir + "chern_slices_occ.txt",      chern_occ_all,   fmt="%.6f")
    np.save(output_dir + "F12_slices_abelian.npy",        F12_all_abel)
 
    # Summary
    chern_spread = np.max(chern_all_abel, axis=0) - np.min(chern_all_abel, axis=0)
    print("\n── Slice Chern summary ──")
    print("  Non-Abelian (occupied) Chern per kz slice:")
    print(" ", np.round(chern_occ_all, 4))
    print("  Non-Abelian Chern (rounded):", np.rint(chern_occ_all).astype(int))
 
    if np.any(chern_spread > 0.5):
        print("\n  WARNING: Abelian Chern varies across kz slices.")
        print("  Check the gap output above — if gap is small, bands are touching.")
        print("  Non-Abelian result is more reliable for degenerate systems.")
    else:
        chern_consensus = np.rint(chern_all_abel[0]).astype(int)
        print(f"\n  Slice Chern numbers are consistent: {chern_consensus}")
        np.savetxt(output_dir + "chern_number.txt", chern_consensus, fmt="%d")
 
    # Non-Abelian is the trusted result for GaAs
    nonabl_consensus = int(np.rint(np.mean(chern_occ_all)))
    print(f"\n  FINAL: Non-Abelian total Chern number = {nonabl_consensus}")
    np.savetxt(output_dir + "chern_nonabalian_final.txt",
               np.array([nonabl_consensus]), fmt="%d")
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 