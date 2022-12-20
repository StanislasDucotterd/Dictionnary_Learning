import torch
import torch.nn as nn
import math


class L2_Dictionnary(nn.Module):
    """Learning the atoms of a dictionnary to reconstruct image with l2 regularization on the coefficients"""
    def __init__(self, nb_atoms, nb_channels, atom_size, l2_reg, init='random'):
        
        super().__init__()

        if init == 'random':
            self.atoms = nn.Parameter(torch.randn(nb_atoms, nb_channels, atom_size, atom_size) / \
                                        (atom_size * math.sqrt(nb_channels)))
        elif init == 'diracs':
            if nb_atoms != nb_channels * atom_size ** 2:
                raise ValueError('Number of atoms should be the same as their dimension')
            else:
                self.atoms = nn.Parameter(torch.eye(nb_atoms).reshape(nb_atoms, nb_channels, atom_size, atom_size))
        else:
            raise ValueError('Init type is not valid')
        self.nb_atoms = nb_atoms
        self.l2_reg = l2_reg
        

    def forward(self, y):
        """ """
        
        atoms = self.atoms / torch.linalg.norm(self.atoms.reshape(self.nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
        X = atoms.reshape(self.nb_atoms, -1) @ atoms.reshape(self.nb_atoms, -1).T
        q = torch.einsum('ijkl,mjkl->im', atoms, y)
        optimal_coeffs = torch.linalg.inv(X + self.l2_reg * torch.eye(self.nb_atoms, device=X.device)) @ q
        return torch.einsum('ij,ilmn->jlmn', optimal_coeffs, atoms)

