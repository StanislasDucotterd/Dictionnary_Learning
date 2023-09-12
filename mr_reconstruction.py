import os
import math
import time
import torch
import argparse
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F
from utils import utilities
from matplotlib import pyplot as plt
from models.proximal_operators import choose_prox
torch.manual_seed(0)

def apply_op(x, HtH, atom_size, beta):
    patches = F.unfold(x, kernel_size=atom_size)
    if new_method: patches = patches - patches.mean(dim=1, keepdim=True)
    return HtH(x) + beta*F.fold(patches, output_size=(x.shape[2], x.shape[3]), kernel_size=atom_size)

def compute_obj(x, y, dict_patches, alphas):
    patches = F.unfold(x, kernel_size=atom_size)
    if new_method: patches = patches - patches.mean(dim=1, keepdim=True)
    dict_patches = dict_patches.view(batch_size, n_patch, atom_size**2).permute(0, 2, 1)
    alphas = alphas.view(batch_size, n_patch, nb_atoms)
    Hx = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')*mask
    data_fitting = ((Hx-y)*(Hx-y).conj()).sum(dim=(1,2,3)).real
    return 0.5*data_fitting + 0.5*beta*((patches-dict_patches)**2).sum(dim=(1,2)) \
           + beta*lambda_*alphas.abs().sum(dim=(1, 2))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device
batch_size = 1
#model_name = 'sigma_5/DEQL1Norm_128_atomsize_10_beta_0.3_step_2_threshold_0.0015'
#model_name = 'sigma_5/OldDEQL1Norm_128_atomsize_10_beta_0.3_step_2_threshold_0.0015'
#model_name = 'sigma_25/DEQL1Norm_128_atomsize_10_beta_2.0_step_2_threshold_0.0075'
model_name = 'sigma_25/OldDEQL1Norm_128_atomsize_10_beta_2.0_step_2_threshold_0.0075'
new_method = not model_name.split('/')[1].startswith('Old')
infos = torch.load('exps/'+model_name+'/checkpoints/best_checkpoint.pth')
config = infos['config']

atoms = infos['state_dict']['atoms'].to(device)
atom_size = config['dict_params']['atom_size']
nb_atoms = config['dict_params']['nb_atoms']

mean_filter = torch.ones(1, 1, atom_size, atom_size, device=device) / (atom_size**2)
atoms = atoms - torch.mean(atoms, dim=(1,2,3), keepdim=True)
atoms = atoms / torch.linalg.norm(atoms.reshape(nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
atoms = atoms / torch.linalg.matrix_norm(atoms.reshape(nb_atoms, -1), 2)
X = atoms.reshape(nb_atoms, -1) @ atoms.reshape(nb_atoms, -1).T

img_name ='Bust'
mask_type = 'Radial30'

mask = torch.tensor(sio.loadmat(f'cs_mri/Q_{mask_type}.mat').get('Q1')).float().to(device)
true_img = torch.tensor(plt.imread(f'cs_mri/{img_name}.jpg')).view(1, 1, 256, 256).float().to(device) / 255
img_height, img_width = true_img.shape[2], true_img.shape[3]

# betas = 10**torch.linspace(1.0, -5.0, 61)
# lambdas = 10**torch.linspace(1.0, -5.0, 61)
betas = [2.75e-3]
lambdas = [3e-3]
psnrs = torch.zeros(len(betas), len(lambdas))

n_patch = (img_height-atom_size+1)*(img_width-atom_size+1)*batch_size
y = torch.fft.fft2(true_img, dim=(-2, -1), norm='ortho')*mask + \
    (10/255)*torch.randn(true_img.shape, dtype=torch.complex128, device=device)
Hty = torch.fft.ifft2(y*mask, dim=(-2, -1), norm='ortho').real.type(torch.float32)
L1 = 1.01

def H(x):
    return torch.fft.fft2(x, dim=(-2, -1), norm='ortho')*mask

def HtH(x):
    Htx = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')*mask
    return torch.fft.ifft2(Htx, dim=(-2, -1), norm='ortho').real


for j, lambda_ in enumerate(lambdas):
    prox = nn.Softshrink(lambd=lambda_/L1)
    for i, beta in enumerate(betas):
        L2 = 1.01*(beta*atom_size**2+1)
        patch_y_mean = F.conv2d(Hty, mean_filter).reshape(batch_size*n_patch, 1, 1, 1)
        img, new_img, temp_img = Hty.clone(), Hty.clone(), Hty.clone()
        coeffs = torch.zeros(n_patch, nb_atoms, device=device)
        new_coeffs, temp_coeffs = coeffs.clone(), coeffs.clone()
        max_res, obj_value, k = 1.0, 1e6, 1
        start = time.time()
        while max_res > 1e-12:
            q = F.conv2d(img, atoms).permute(0, 2, 3, 1).reshape(batch_size*n_patch, nb_atoms)
            temp_coeffs = new_coeffs + (k-1) * (new_coeffs - coeffs) / (k+2)
            coeffs = new_coeffs
            new_coeffs = prox(temp_coeffs - (temp_coeffs @ X - q) / L1)
            new_dict_patches = torch.einsum('ij,jlmn->ilmn', new_coeffs, atoms)
            if not new_method: new_dict_patches = new_dict_patches + patch_y_mean
            dict_pred = new_dict_patches.view(batch_size, n_patch, atom_size**2).permute(0, 2, 1)
            dict_pred = F.fold(dict_pred, output_size=(img_height, img_width), kernel_size=atom_size)
            temp_img = new_img + (k-1) * (new_img - img) / (k+2)
            img = new_img
            new_img = torch.clip(temp_img - (apply_op(temp_img, HtH, atom_size, beta) - Hty - beta*dict_pred) / L2, 0., 1.)
            dict_patches = torch.einsum('ij,jlmn->ilmn', coeffs, atoms)
            if not new_method: dict_patches = dict_patches + patch_y_mean
            new_obj_value = compute_obj(img, y, dict_patches, coeffs)
            max_res = (obj_value - new_obj_value).abs() / obj_value
            obj_value = new_obj_value
            k += 1
        psnr = utilities.batch_PSNR(img, true_img, 1.)
        print(f'Beta: {beta:.6}, Lambda: {lambda_:.6}, PSNR: {psnr:.6}, it took {time.time()-start:.2f}s')
        psnrs[i, j] = utilities.batch_PSNR(img, true_img, 1.)
best_psnr = psnrs[~psnrs.isnan()].max()
best_beta = betas[(psnrs == best_psnr).nonzero()[0,0]]
best_lambda = lambdas[(psnrs == best_psnr).nonzero()[0,1]]
print(f'Best PSNR is {best_psnr:.6} with beta={best_beta:.6} and lambda={best_lambda:.6}')
performances = {'psnrs': psnrs, 'betas': betas, 'lambdas': lambdas}
plt.imsave(f'cs_mri/{img_name}_{mask_type}_sigma=25.png', img[0,0,:,:].cpu().numpy(), cmap='gray')

# path = f'mr_reconstruction/{img_name}_{mask_type}/' + model_name
# if not os.path.exists(path):
#     os.makedirs(path)
# torch.save(performances, path + f'/perf_{best_psnr:.4}_beta_{best_beta:.2e}_lmbda_{best_lambda:.2e}.pth')