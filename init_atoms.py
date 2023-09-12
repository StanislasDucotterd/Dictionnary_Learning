import torch

def get_gabor(nb_atoms, nb_channels, atom_size):
    a = 2.0*torch.rand(nb_atoms, nb_channels, 1, 1, 2)
    w0 = torch.pi*torch.rand(nb_atoms, nb_channels, 1, 1, 2)
    x0 = (atom_size-1)*torch.rand(nb_atoms, nb_channels, 1, 1, 2)
    psi = 2*torch.pi*torch.rand(nb_atoms, nb_channels, 1, 1)

    i = torch.arange(atom_size).to(a.device)
    x = torch.stack(torch.meshgrid(i,i, indexing='ij'), dim=2)[None,None,...]

    atoms = torch.exp(-torch.sum((a*(x-x0))**2, dim=-1)) * torch.cos(torch.sum(w0*(x-x0), dim=-1) + psi)
    atoms = (atoms - torch.mean(atoms, dim=(1,2,3)).reshape(-1, 1, 1, 1)) / 40
    return atoms