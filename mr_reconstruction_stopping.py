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
from optimization import apply_op, get_coeffs_ista, get_coeffs_fista, get_img_fista, get_img_ista
torch.manual_seed(0)

def compute_obj(x, y, dict_patches, alphas):
    patches = F.unfold(x, kernel_size=atom_size)
    if new_method:
        patches = patches - patches.mean(dim=1, keepdim=True)
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
model_name = 'sigma_25/DEQL1Norm_128_atomsize_10_beta_2.0_step_2_threshold_0.0075'
#model_name = 'sigma_25/OldDEQL1Norm_128_atomsize_10_beta_2.0_step_2_threshold_0.0075'
new_method = not model_name.split('/')[1].startswith('Old')
infos = torch.load('exps/'+model_name+'/checkpoints/best_checkpoint.pth')
config = infos['config']

atoms = infos['state_dict']['atoms'].to(device)
atom_size = config['dict_params']['atom_size']
nb_atoms = config['dict_params']['nb_atoms']

mean_filter = torch.ones(1, 1, atom_size, atom_size) / (atom_size**2)
atoms = atoms - torch.mean(atoms, dim=(1,2,3), keepdim=True)
atoms = atoms / torch.linalg.norm(atoms.reshape(nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
atoms = atoms / (torch.linalg.matrix_norm(atoms.reshape(nb_atoms, -1), 2))
X = atoms.reshape(nb_atoms, -1) @ atoms.reshape(nb_atoms, -1).T
res_criteria = 1e-12

img_name ='Brain'
mask_type = 'Cartesian30'

mask = torch.tensor(sio.loadmat(f'cs_mri/Q_{mask_type}.mat').get('Q1')).float().to(device)
img = torch.tensor(plt.imread(f'cs_mri/{img_name}.jpg')).view(1, 1, 256, 256).float().to(device) / 255
img_height, img_width = img.shape[2], img.shape[3]

base_beta = 3e-3
base_lambda = 2.5e-3
gamma = 1.05
betas = [base_beta]
lambdas = [base_lambda]
psnrs = torch.zeros(len(betas), len(lambdas))

n_patch = (img_height-atom_size+1)*(img_width-atom_size+1)*batch_size
y = torch.fft.fft2(img, dim=(-2, -1), norm='ortho')*mask + \
    (10/255)*torch.randn(img.shape, dtype=torch.complex128, device=device)
Hty = torch.fft.ifft2(y*mask, dim=(-2, -1), norm='ortho').real.type(torch.float32)
patch_y_mean = F.conv2d(Hty, mean_filter.to(y.device)).reshape(batch_size*n_patch, 1, 1, 1)
first_q = F.conv2d(Hty, atoms).permute(0, 2, 3, 1).reshape(batch_size*n_patch, nb_atoms)

def H(x):
    return torch.fft.fft2(x, dim=(-2, -1), norm='ortho')*mask

def HtH(x):
    Htx = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')*mask
    return torch.fft.ifft2(Htx, dim=(-2, -1), norm='ortho').real

start = time.time()
for j, lambda_ in enumerate(lambdas):
    prox = nn.Softshrink(lambd=lambda_)
    first_lambda = True
    for i, beta in enumerate(betas):
        goal_img = Hty.clone()
        starting_coeffs = torch.zeros(n_patch, nb_atoms, device=device)
        max_res, obj_value, iteration = 1.0, 1e6, 1
        while max_res > res_criteria:
            if first_lambda and iteration == 1:
                coeffs = get_coeffs_fista(starting_coeffs, X, first_q, prox, 1e-4)
                first_coeffs, first_lambda = coeffs.clone(), False
            elif iteration == 1: coeffs = first_coeffs.clone()
            else:
                q = F.conv2d(goal_img, atoms).permute(0, 2, 3, 1).reshape(batch_size*n_patch, nb_atoms)
                if iteration > 50: 
                    coeffs = get_coeffs_ista(starting_coeffs, X, q, prox, 1e-4)
                else: coeffs = get_coeffs_fista(starting_coeffs, X, q, prox, 1e-4)
            dict_patches = torch.einsum('ij,jlmn->ilmn', coeffs, atoms)
            if not new_method: dict_patches = dict_patches + patch_y_mean
            dict_pred = dict_patches.view(batch_size, n_patch, atom_size**2).permute(0, 2, 1)
            dict_pred = F.fold(dict_pred, output_size=(img_height, img_width), kernel_size=atom_size)
            if new_method: 
                new_img = get_img_fista(goal_img, HtH, dict_pred, Hty, beta, atom_size, 1e-6)
            else: new_img = get_img_fista_old(goal_img, HtH, dict_pred, Hty, beta, atom_size, 1e-6)
            max_res = torch.sqrt((((goal_img-new_img)**2).sum(dim=(1,2,3))+((coeffs-starting_coeffs)**2).view(batch_size,-1).sum(dim=1))/((goal_img**2).sum(dim=(1,2,3)) + ((starting_coeffs)**2).view(batch_size, -1).sum(dim=1)))
            starting_coeffs = coeffs.clone()
            goal_img = new_img.clone()
            iteration += 1
            print(f'Iteration {iteration}, PSNR of {utilities.batch_PSNR(goal_img, img, 1.)}')
        print('PSNR = ', utilities.batch_PSNR(goal_img, img, 1.), ' for beta=', beta, ' and lambda=', lambda_)
        psnrs[i, j] = utilities.batch_PSNR(goal_img, img, 1.)
print(f'Optimization done, it took {time.time()-start:.2f}s') 
best_psnr = psnrs[~psnrs.isnan()].max()
best_beta = betas[(psnrs == best_psnr).nonzero()[0,0]]
best_lambda = lambdas[(psnrs == best_psnr).nonzero()[0,1]]
print(f'Best PSNR is {best_psnr:.6} with beta={best_beta:.6} and lambda={best_lambda:.6}')
performances = {'psnrs': psnrs, 'betas': betas, 'lambdas': lambdas}

path = f'mr_reconstruction/{img_name}_{mask_type}/stopping_' + model_name
if not os.path.exists(path):
    os.makedirs(path)
torch.save(performances, path + f'/perf_{best_psnr:.4}_beta_{best_beta:.2e}_lmbda_{best_lambda:.2e}.pth')