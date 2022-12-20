import math
import torch
import torch_dct as dct

def get_dct(nb_atoms, nb_channels, atom_size):
    atoms = torch.eye(nb_channels * atom_size ** 2)
    atoms = atoms.reshape(nb_channels * atom_size ** 2, nb_channels, atom_size, atom_size)
    atoms = dct.dct_2d(atoms).repeat(math.ceil(nb_atoms / atom_size**2), 1, 1, 1)[:nb_atoms,...]
    atoms = atoms - torch.mean(atoms, dim=(1,2,3)).reshape(-1, 1, 1, 1) + torch.randn(atoms.shape) * 0.05
    return atoms