import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from models.proximal_operators import choose_prox
from optimization import apply_op, get_coeffs_fista

class Dictionnary(nn.Module):

    def __init__(self, prox_params, nb_atoms, nb_channels, atom_size, res, beta, mu, hyper_learnable, unroll_steps):
        super().__init__()

        self.atoms = nn.Parameter(torch.randn(nb_atoms, nb_channels, atom_size, atom_size) / 1500)
        self.mean_filter = torch.ones(1, 1, atom_size, atom_size) / (atom_size**2)         
        self.nb_atoms = nb_atoms
        self.nb_channels = nb_channels
        self.atom_size = atom_size
        self.res = res
        self.beta = torch.tensor(beta)
        self.mu = torch.tensor(mu)
        if hyper_learnable: 
            self.beta = nn.Parameter(self.beta)
            self.mu = nn.Parameter(self.mu)
        self.unroll_steps = unroll_steps
        self.proximal = choose_prox(nb_atoms, prox_params)

    def forward(self, y):
        self.backward_count = 0
        self.s = []

        batch_size, y_height, y_width = y.shape[0], y.shape[2], y.shape[3]
        atoms = self.atoms - torch.mean(self.atoms, dim=(1,2,3), keepdim=True)

        beta = F.relu(self.beta)
        mu = F.relu(self.mu)

        atoms = atoms / torch.linalg.norm(atoms.reshape(self.nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
        atoms = atoms / (torch.linalg.matrix_norm(atoms.reshape(self.nb_atoms, -1), 2) * torch.sqrt(mu))
        X = atoms.reshape(self.nb_atoms, -1) @ atoms.reshape(self.nb_atoms, -1).T

        n_patch = (y.shape[2]-self.atom_size+1)*(y.shape[3]-self.atom_size+1)*batch_size
        patch_y_mean = F.conv2d(y, self.mean_filter.to(y.device)).reshape(n_patch, 1, 1, 1)

        goal_img = y.clone()
        starting_coeffs = torch.zeros(n_patch, self.nb_atoms, device=y.device)
        for _ in range(self.unroll_steps):
            q = F.conv2d(goal_img, atoms).permute(0, 2, 3, 1).reshape(n_patch, self.nb_atoms)
            with torch.no_grad():
                final_coeffs = get_coeffs_fista(starting_coeffs, X, q, self.proximal, mu, self.res)

            final_coeffs = self.proximal(final_coeffs - (final_coeffs @ X - q) * mu)
            z0 = final_coeffs.clone().detach().requires_grad_()
            self.s.append(self.proximal.jacobian(z0 - (z0 @ X - q) * mu))
            final_coeffs.register_hook(lambda grad: self.backward_hook(grad, X, mu))

            dict_pred = torch.einsum('ij,jlmn->ilmn', final_coeffs, atoms) + patch_y_mean 
            dict_pred = dict_pred.view(batch_size, n_patch//batch_size, self.nb_channels*self.atom_size**2).permute(0, 2, 1)
            dict_pred = F.fold(dict_pred, output_size=(y_height, y_width), kernel_size=self.atom_size)

            divisors = F.unfold(torch.ones(y.shape, device=y.device), kernel_size=self.atom_size)
            divisors = F.fold(divisors, output_size=(y_height, y_width), kernel_size=self.atom_size)
            goal_img = (y + beta * dict_pred) / (1.0 + beta * divisors) 
            starting_coeffs = final_coeffs.clone()

        return goal_img

    def val_forward(self, y):

        batch_size, y_height, y_width = y.shape[0], y.shape[2], y.shape[3]
        atoms = self.atoms - torch.mean(self.atoms, dim=(1,2,3), keepdim=True)

        beta = F.relu(self.beta)
        mu = F.relu(self.mu)

        atoms = atoms / torch.linalg.norm(atoms.reshape(self.nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
        atoms = atoms / (torch.linalg.matrix_norm(atoms.reshape(self.nb_atoms, -1), 2) * torch.sqrt(mu))
        X = atoms.reshape(self.nb_atoms, -1) @ atoms.reshape(self.nb_atoms, -1).T

        n_patch = (y_height-self.atom_size+1)*(y_width-self.atom_size+1)*batch_size
        patch_y_mean = F.conv2d(y, self.mean_filter.to(y.device)).reshape(n_patch, 1, 1, 1)

        #ISTA to find the global minimum
        goal_img = y.clone()
        starting_coeffs = torch.zeros(n_patch, self.nb_atoms, device=y.device)
        for _ in range(self.unroll_steps):
            q = F.conv2d(goal_img, atoms).permute(0, 2, 3, 1).reshape(n_patch, self.nb_atoms)
            with torch.no_grad():
                final_coeffs = get_coeffs_fista(starting_coeffs, X, q, lambda x: self.proximal(x), mu, 1e-4)

            dict_pred = torch.einsum('ij,jlmn->ilmn', final_coeffs, atoms) + patch_y_mean
            dict_pred = dict_pred.view(batch_size, n_patch//batch_size, self.nb_channels*self.atom_size**2).permute(0, 2, 1)
            dict_pred = F.fold(dict_pred, output_size=(y_height, y_width), kernel_size=self.atom_size)

            divisors = F.unfold(torch.ones(y.shape, device=y.device), kernel_size=self.atom_size)
            divisors = F.fold(divisors, output_size=(y_height, y_width), kernel_size=self.atom_size)
            goal_img = (y + beta * dict_pred) / (1.0 + beta * divisors) 
            starting_coeffs = final_coeffs.clone()

        return goal_img


    def backward_hook(self, grad, X, mu):
        real_grad = invert_jacobian(grad, self.s[self.unroll_steps-self.backward_count-1], X, mu)
        self.backward_count += 1
        return real_grad

    def extra_repr(self):
        return f"""nb_atoms={self.nb_atoms}, nb_channels={self.nb_channels}, atom_size={self.atom_size},
        res={self.res}, beta={self.beta}, mu={self.mu}, unroll_steps={self.unroll_steps}"""

def invert_jacobian(grad, s, X, mu):
    equation_batch = int((grad.shape[0] / 32) * (256 / grad.shape[1])**2)
    real_grad = torch.zeros(grad.shape, device=grad.device)
    Id = torch.eye(grad.shape[1], device=grad.device)
    eps = 1e-3
    for i in range(grad.shape[0] // equation_batch):
        real_grad[i*equation_batch:(i+1)*equation_batch,:] = torch.linalg.solve((1+eps)*Id - (Id - X*mu) @ \
                torch.diag_embed(s[i*equation_batch:(i+1)*equation_batch,:]), grad[i*equation_batch:(i+1)*equation_batch,:])
    real_grad[(i+1)*equation_batch:-1,:] = torch.linalg.solve((1+eps)*Id - (Id - X*mu) @ torch.diag_embed(s[(i+1)*equation_batch:-1,:]),\
                                            grad[(i+1)*equation_batch:-1,:])
    return real_grad