import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvHull_Dictionnary(nn.Module):
    """Learning the atoms of a dictionnary to reconstruct image with simplex constraints on the coeffs"""
    def __init__(self, nb_atoms, nb_channels, atom_size, nondiff_steps, diff_steps, temp, init='random'):
        
        super().__init__()

        if init == 'random':
            self.atoms = nn.Parameter(torch.randn(nb_atoms, nb_channels, atom_size, atom_size) / \
                                        (atom_size * math.sqrt(nb_channels)))
        elif init == 'diracs':
            if nb_atoms != nb_channels * atom_size ** 2:
                raise ValueError('Number of atoms should be the same as their dimension')
            else:
                self.atoms = nn.Parameter(torch.eye(nb_channels * atom_size ** 2).reshape(nb_channels * atom_size ** 2, \
                                                    nb_channels, atom_size, atom_size))
        else:
            raise ValueError('Init type is not valid')
            
        self.nb_atoms = nb_atoms
        self.nondiff_steps = nondiff_steps
        self.diff_steps = diff_steps
        self.temp = temp

    def forward(self, y):
        """ """
        y_norms = torch.linalg.norm(y.reshape(y.shape[0], -1), axis=1, ord=1).reshape(-1, 1, 1, 1)
        atoms = self.atoms / torch.linalg.norm(self.atoms.reshape(self.nb_atoms, -1), axis=1, ord=1).reshape(-1, 1, 1, 1)
        X = atoms.reshape(self.nb_atoms, -1) @ atoms.reshape(self.nb_atoms, -1).T
        q = torch.einsum('ijkl,mjkl->im', atoms, y / y_norms)

        new_coeffs = torch.ones(self.nb_atoms, y.shape[0], device=X.device) / self.nb_atoms
        with torch.no_grad():
            for _ in range(self.nondiff_steps):
                grad = X @ new_coeffs - q
                argmin = torch.zeros(self.nb_atoms, y.shape[0], device=X.device)
                argmin[torch.arange(self.nb_atoms), torch.argmin(grad, dim=1)] = 1.0
                lambda_update = torch.clip(-torch.sum((argmin - new_coeffs) * grad, dim=0).unsqueeze(0) / \
                                (torch.sum((argmin - new_coeffs) * (X @ (argmin - new_coeffs))) + 1e-8), 0.0, 1.0)
                new_coeffs = new_coeffs + (argmin - new_coeffs) * lambda_update
        for _ in range(self.diff_steps):
            grad = X @ new_coeffs - q
            sargmin = F.softmax(-self.temp * grad, dim=0)
            lambda_update = torch.clip(-torch.sum((sargmin - new_coeffs) * grad, dim=0).unsqueeze(0) / \
                                (torch.sum((sargmin - new_coeffs) * (X @ (sargmin - new_coeffs))) + 1e-8), 0.0, 1.0)
            new_coeffs = new_coeffs + (sargmin - new_coeffs) * lambda_update

        return torch.einsum('ij,ilmn->jlmn', new_coeffs, atoms) * y_norms