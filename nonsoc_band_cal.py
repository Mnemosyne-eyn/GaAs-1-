import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from pymatgen.core.periodic_table import Element
import math
import os
from utils import *
import argparse
import yaml
import torch

def main():
    parser = argparse.ArgumentParser(description='band calculation')
    parser.add_argument('--config', default='band_cal.yaml', type=str, metavar='N')
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as rstream:
        input = yaml.load(rstream, yaml.SafeLoader)

    nao_max = input['nao_max']
    graph_data_path = input['graph_data_path']
    hamiltonian_path = input['hamiltonian_path']
    nk = input['nk']
    save_dir = input['save_dir']
    filename = input['strcture_name']

    Ham_type = input.get('Ham_type', 'openmx').lower()
    soc_switch = input.get('soc_switch', False)
    spin_colinear = input.get('spin_colinear', False)
    auto_mode = input['auto_mode']

    if not auto_mode:
        k_path = input['k_path']
        label = input['label']

    os.makedirs(save_dir, exist_ok=True)

    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())

    num_val = np.zeros((99,), dtype=int)
    for k in num_valence_openmx.keys():
        num_val[k] = num_valence_openmx[k]

    basis_definition = np.zeros((99, nao_max))
    if nao_max == 14:
        basis_def = basis_def_14
    elif nao_max == 19:
        basis_def = basis_def_19
    else:
        basis_def = basis_def_26

    for k in basis_def.keys():
        basis_definition[k][basis_def[k]] = 1

    # ---------------- NON-SOC, NON-SPIN BRANCH ----------------
    len_H = []
    for i in range(len(graph_dataset)):
        len_H.append(len(graph_dataset[i].Hon))
        len_H.append(len(graph_dataset[i].Hoff))
               
    if hamiltonian_path is not None:
        H = np.load(hamiltonian_path)
        Hon_all, Hoff_all = [], []
        idx = 0
        for i in range(0, len(len_H), 2):
            Hon_all.append(H[idx:idx + len_H[i]])
            idx = idx+len_H[i]
            Hoff_all.append(H[idx:idx + len_H[i+1]])
            idx = idx+len_H[i+1]
    else:
        Hon_all, Hoff_all = [], []
        for data in graph_dataset:
            Hon_all.append(data.Hon.numpy())
            Hoff_all.append(data.Hoff.numpy())
    
    for idx, data in enumerate(graph_dataset):

        Son = data.Son.numpy().reshape(-1, nao_max, nao_max)
        Soff = data.Soff.numpy().reshape(-1, nao_max, nao_max)
        Hon = Hon_all[idx].reshape(-1, nao_max, nao_max)
        Hoff = Hoff_all[idx].reshape(-1, nao_max, nao_max)
        latt = data.cell.numpy().reshape(3,3)
        pos = data.pos.numpy()*au2ang
        nbr_shift = data.nbr_shift.numpy()
        edge_index = data.edge_index.numpy()
        species = data.z.numpy()

        struct = Structure(
            lattice=latt*au2ang,
            species=[Element.from_Z(k).symbol for k in species],
            coords=pos,
            coords_are_cartesian=True
        )
        struct.to(filename=os.path.join(save_dir, filename+f'_{idx+1}.cif'))

        if auto_mode:
            kpath_seek = KPathSeek(structure = struct)
            klabels = []
            for lbs in kpath_seek.kpath['path']:
                klabels += lbs
            res = [klabels[0]]
            [res.append(x) for x in klabels[1:] if x != res[-1]]
            klabels = res
            k_path = [kpath_seek.kpath['kpoints'][k] for k in klabels]
            label = [rf'${lb}$' for lb in klabels]            

        orb_mask = basis_definition[species].reshape(-1)
        orb_mask = orb_mask[:,None] * orb_mask[None,:]

        kpts = kpoints_generator(dim_k=3, lat=latt)
        k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)

        k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:])
        k_vec = k_vec.reshape(-1,3)

        natoms = len(struct)
        eigen = []
        eigvals_all = []
        eigvecs_all = []

        for ik in range(nk):
            HK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
            SK = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
        
            na = np.arange(natoms)
            HK[na,na,:,:] +=  Hon[na,:,:]
            SK[na,na,:,:] +=  Son[na,:,:]
        
            coe = np.exp(2j*np.pi*np.sum(nbr_shift*k_vec[ik][None,:], axis=-1))
        
            for iedge in range(len(Hoff)):
                HK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Hoff[iedge,:,:]
                SK[edge_index[0, iedge],edge_index[1, iedge]] += coe[iedge,None,None] * Soff[iedge,:,:]
        
            HK = np.swapaxes(HK,-2,-3)
            HK = HK.reshape(natoms*nao_max, natoms*nao_max)
            SK = np.swapaxes(SK,-2,-3)
            SK = SK.reshape(natoms*nao_max, natoms*nao_max)
        
            HK = HK[orb_mask > 0]
            SK = SK[orb_mask > 0]
            norbs = int(math.sqrt(HK.size))
            HK = HK.reshape(norbs, norbs)
            SK = SK.reshape(norbs, norbs)

            SK_cuda = torch.complex(torch.Tensor(SK.real), torch.Tensor(SK.imag)).unsqueeze(0)
            HK_cuda = torch.complex(torch.Tensor(HK.real), torch.Tensor(HK.imag)).unsqueeze(0)
            L = torch.linalg.cholesky(SK_cuda)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK_cuda), L_t_inv)
            orbital_energies, orbital_vecs = torch.linalg.eigh(Hs)
            orbital_energies = orbital_energies.squeeze(0)
            orbital_vecs = orbital_vecs.squeeze(0)

            eigvals_all.append(orbital_energies.cpu().numpy())
            eigvecs_all.append(orbital_vecs.cpu().numpy())
            eigen.append(orbital_energies.cpu().numpy())

        eigvals_all = np.array(eigvals_all) * au2ev
        eigvecs_all = np.array(eigvecs_all)

        np.savez(
            os.path.join(save_dir, "bands_gmkg.npz"),
            kpoints=k_vec,
            eigenvalues=eigvals_all,
            eigenvectors=eigvecs_all,
        )

        eigen = np.swapaxes(np.array(eigen), 0, 1)*au2ev

        num_electrons = np.sum(num_val[species])
        max_val = np.max(eigen[math.ceil(num_electrons/2)-1])
        min_con = np.min(eigen[math.ceil(num_electrons/2)])
        eigen = eigen - max_val
        print(f"max_val = {max_val} eV")
        print(f"band gap = {min_con - max_val} eV")
        
        if nk > 1:
            print('Plotting bandstructure...')
            fig, ax = plt.subplots()
            ax.set_xlim(k_node[0],k_node[-1])
            ax.set_xticks(k_node)
            ax.set_xticklabels(label)
            for n in range(len(k_node)):
                ax.axvline(x=k_node[n], linewidth=0.5, color='k')
            for n in range(norbs):
                ax.plot(k_dist, eigen[n])
            ax.plot(k_dist, nk*[0.0], linestyle='--')
            ax.set_title("Band structure")
            ax.set_xlabel("Path in k-space")
            ax.set_ylabel("Band energy (eV)")
            ax.set_ylim(-4, 6)
            fig.tight_layout()
            plt.savefig(os.path.join(save_dir, f'band_{idx+1}.png'))
            print('Done.\n')

if __name__ == '__main__':
    main()
