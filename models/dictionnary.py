import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from init_atoms import get_dct
from models.proximal_operators import choose_prox

def softhreshold(x, threshold):
    return x + threshold - F.relu(x + threshold) + F.relu(x - threshold)

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

class Dictionnary(nn.Module):
    """Learning the atoms of a dictionnary to reconstruct image with l1 regularization on the coefficients"""
    def __init__(self, prox_params, nb_atoms, nb_channels, atom_size, stride,  nondiff_steps, diff_steps, init='random'):
        
        super().__init__()

        if init == 'random':
            self.atoms = nn.Parameter(torch.randn(nb_atoms, nb_channels, atom_size, atom_size) \
                                        / math.sqrt(nb_channels * atom_size ** 2))             
        elif init == 'diracs':
            if nb_atoms != nb_channels * atom_size ** 2:
                raise ValueError('Number of atoms should be the same as their dimension')
            else:
                self.atoms = nn.Parameter(torch.eye(nb_channels * atom_size ** 2).reshape(nb_channels * atom_size ** 2, \
                                                    nb_channels, atom_size, atom_size))
        elif init == 'dct':
            self.atoms = nn.Parameter(get_dct(nb_atoms, nb_channels, atom_size))
        else:
            raise ValueError('Init type is not valid')
            
        self.nb_atoms = nb_atoms
        self.nb_channels = nb_channels
        self.atom_size = atom_size
        self.nondiff_steps = nondiff_steps
        self.diff_steps = diff_steps
        self.eigenvector = torch.randn(self.nb_atoms)
        self.stride = stride
        self.proximal = choose_prox(prox_params, nb_atoms)

    def forward(self, y):
        """ """

        batch_size, y_height, y_width = y.shape[0], y.shape[2], y.shape[3]
        y = y.unfold(2, self.atom_size, self.stride).unfold(3, self.atom_size, self.stride)
        n_patch_height, n_patch_width = y.shape[2], y.shape[3]
        y = y.permute(0, 2, 3, 1, 4, 5).reshape(-1, y.shape[1], y.shape[4], y.shape[5])
        y_mean = torch.mean(y, axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
        y = y - y_mean

        atoms = self.atoms / torch.linalg.norm(self.atoms.reshape(self.nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
        X = atoms.reshape(self.nb_atoms, -1) @ atoms.reshape(self.nb_atoms, -1).T
        q = torch.einsum('ijkl,mjkl->im', atoms, y)
        L = torch.linalg.matrix_norm(X, 2)

        self.proximal.project()
        new_coeffs = torch.zeros(self.nb_atoms, y.shape[0], device=X.device)
        with torch.no_grad():
            for _ in range(self.nondiff_steps):
                new_coeffs = new_coeffs - (X @ new_coeffs - q) / L
                new_coeffs = self.proximal(new_coeffs, L)
        for _ in range(self.diff_steps):
            new_coeffs = new_coeffs - (X @ new_coeffs - q) / L
            new_coeffs = self.proximal(new_coeffs, L)

        y = torch.einsum('ij,ilmn->jlmn', new_coeffs, atoms) + y_mean
        y = y.view(batch_size, n_patch_height*n_patch_width, self.nb_channels*self.atom_size**2).permute(0, 2, 1)
        y = F.fold(y, output_size=(y_height, y_width), kernel_size=self.atom_size, stride=self.stride)

        divisors = F.unfold(torch.ones(y.shape, device=y.device), kernel_size=self.atom_size, stride=self.stride)
        divisors = F.fold(divisors, output_size=(y_height, y_width), kernel_size=self.atom_size, stride=self.stride)

        return y / divisors