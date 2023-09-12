import os
import time
import math
import torch
import argparse
import torch.nn as nn
import scipy.io as sio
from utils import utilities
import torch.nn.functional as F
from dataloader.BSD500 import BSD500
from matplotlib import pyplot as plt
from optimization import apply_op, get_coeffs_ista, get_coeffs_fista, get_img_fista
torch.manual_seed(0)

def compute_obj(x, y, dict_patches, alphas):
    patches = F.unfold(x, kernel_size=atom_size)
    if new_method:
        patches = patches - patches.mean(dim=1, keepdim=True)
    dict_patches = dict_patches.view(batch_size, n_patch, atom_size**2).permute(0, 2, 1)
    alphas = alphas.view(batch_size, n_patch, nb_atoms)
    return 0.5*((x-y)**2).sum(dim=(1,2,3)) + 0.5*beta*((patches-dict_patches)**2).sum(dim=(1,2)) \
           + beta*lambda_*alphas.abs().sum(dim=(1, 2))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device
model_name = 'sigma_5/DEQL1Norm_128_atomsize_10_beta_0.3_step_2_threshold_0.0015'
#model_name = 'sigma_5/OldDEQL1Norm_128_atomsize_10_beta_0.3_step_2_threshold_0.0015'
#model_name = 'sigma_25/DEQL1Norm_128_atomsize_10_beta_2.0_step_2_threshold_0.0075'
#model_name = 'sigma_25/OldDEQL1Norm_128_atomsize_10_beta_2.0_step_2_threshold_0.0075'
new_method = not model_name.split('/')[1].startswith('Old')
infos = torch.load('exps/'+model_name+'/checkpoints/best_checkpoint.pth')
config = infos['config']
dataset = BSD500('/home/ducotter/nerf_dip/images/test.h5')
sigma = float(model_name.split('/')[0].split('_')[1])

batch1 = torch.zeros(15, 1, 481, 321)
batch2 = torch.zeros(27, 1, 321, 481)
batch3 = torch.zeros(26, 1, 321, 481)
index1, index2 = 0, 0
for i in range(dataset.__len__()):
    img = dataset.__getitem__(i)
    if img.shape[1] == 321:
        if index2 >= 27: batch3[index2-27] = img
        else: batch2[index2] = img
        index2 += 1
    else:
        batch1[index1] = img
        index1 += 1
imgs = [batch1, batch2, batch3]

# batch1 = torch.zeros(7, 1, 256, 256)
# batch2 = torch.zeros(5, 1, 512, 512)
# index1, index2 = 0, 0
# for i in range(dataset.__len__()):
#    img = dataset.__getitem__(i)
#    if img.shape[1] == 256:
#        batch1[index1] = img
#        index1 += 1
#    else:
#        batch2[index2] = img
#        index2 += 1

# imgs = [batch1, batch2]

atoms = infos['state_dict']['atoms'].to(device)
atom_size = config['dict_params']['atom_size']
nb_atoms = config['dict_params']['nb_atoms']

mean_filter = torch.ones(1, 1, atom_size, atom_size) / (atom_size**2)
atoms = atoms - torch.mean(atoms, dim=(1,2,3), keepdim=True)
atoms = atoms / torch.linalg.norm(atoms.reshape(nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
atoms = atoms / (torch.linalg.matrix_norm(atoms.reshape(nb_atoms, -1), 2))
X = atoms.reshape(nb_atoms, -1) @ atoms.reshape(nb_atoms, -1).T
res_criteria = 1e-4

def HtH(x):
    return x
base_beta = 0.6
base_lambda = 7.75e-5
gamma = 1.1
betas = [base_beta/gamma, base_beta, base_beta*gamma]
lambdas = [base_lambda/gamma, base_lambda, base_lambda*gamma]
psnrs = torch.zeros(len(betas), len(lambdas))

start = time.time()
for img in imgs:
    img = img.to(device)
    y = img + (sigma/255) * torch.randn(img.shape, device=device)
    y_height, y_width = y.shape[2], y.shape[3]
    n_patch = (y_height-atom_size+1)*(y_width-atom_size+1)
    for j, lambda_ in enumerate(lambdas):
        prox = nn.Softshrink(lambda_)
        first_lambda = True
        for i, beta in enumerate(betas):
            batch_size = img.shape[0]
            goal_img, temp_y = y.clone(), y.clone()
            patch_y_mean = F.conv2d(y, mean_filter.to(y.device)).reshape(-1, 1, 1, 1)
            starting_coeffs = torch.zeros(batch_size*n_patch, nb_atoms, device=device)
            final_preds = torch.zeros(img.shape, device=device) * float('nan')
            max_res, obj_value, iteration = 1.0, 1e6, 1
            while max_res > res_criteria:
                q = F.conv2d(goal_img, atoms).permute(0, 2, 3, 1).reshape(batch_size*n_patch, nb_atoms)
                if first_lambda and iteration == 1:
                    coeffs = get_coeffs_fista(starting_coeffs, X, q, prox, 1e-4)
                    first_coeffs, first_lambda = coeffs.clone(), False
                elif iteration == 1: coeffs = first_coeffs.clone()
                else:
                    if iteration > 50: coeffs = get_coeffs_ista(starting_coeffs, X, q, prox, 1e-4)
                    else: coeffs = get_coeffs_fista(starting_coeffs, X, q, prox, 1e-4)
                dict_patches = torch.einsum('ij,jlmn->ilmn', coeffs, atoms)
                if not new_method: dict_patches = dict_patches + patch_y_mean
                dict_pred = dict_patches.view(batch_size, n_patch, atom_size**2).permute(0, 2, 1)
                dict_pred = F.fold(dict_pred, output_size=(y_height, y_width), kernel_size=atom_size)
                new_img = get_img_fista(goal_img, HtH, dict_pred, temp_y, beta, atom_size, 1e-6, new_method)
                res = torch.sqrt((((goal_img-new_img)**2).sum(dim=(1,2,3))+((coeffs-starting_coeffs)**2).view(batch_size,-1).sum(dim=1))/((goal_img**2).sum(dim=(1,2,3)) + ((starting_coeffs)**2).view(batch_size, -1).sum(dim=1)))
                starting_coeffs = coeffs.clone()
                goal_img = new_img.clone()
                condition = res < res_criteria
                if condition.sum().item() > 0:
                    final_nan_indices = final_preds.view(final_preds.shape[0], -1).isnan().any(dim=1)
                    temp_var = torch.zeros(final_nan_indices.sum(), final_preds.shape[1], final_preds.shape[2], final_preds.shape[3], device=y.device)*float('nan')
                    temp_var[condition,:] = goal_img[condition,:]
                    final_preds[final_nan_indices,:] = temp_var
                    starting_coeffs = starting_coeffs.view(batch_size, n_patch, nb_atoms)
                    starting_coeffs = starting_coeffs[~condition,:].view(-1, nb_atoms)
                    patch_y_mean = patch_y_mean.view(batch_size, n_patch)
                    patch_y_mean = patch_y_mean[~condition,:].view(-1, 1, 1, 1)
                    goal_img, temp_y, new_img = goal_img[~condition,:], temp_y[~condition,:], new_img[~condition,:]
                    batch_size = temp_y.shape[0]
                max_res = torch.max(res)
                iteration += 1
            psnrs[i,j] += final_preds.shape[0]*utilities.batch_PSNR(final_preds, img, 1.)/12
            print('PSNR = ', utilities.batch_PSNR(final_preds, img, 1.), ' for beta=', beta, ' and lambda=', lambda_, ' in ', iteration, ' iterations')
print(f'Optimization done, it took {time.time()-start:.2f}s')
best_psnr = psnrs[~psnrs.isnan()].max()
best_beta = betas[(psnrs == best_psnr).nonzero()[0,0]]
best_lambda = lambdas[(psnrs == best_psnr).nonzero()[0,1]]
print(f'Best PSNR is {best_psnr:.6} with beta={best_beta:.6} and lambda={best_lambda:.6}')
performances = {'psnrs': psnrs, 'betas': betas, 'lambdas': lambdas}

# path = 'denoising/' + model_name
# if not os.path.exists(path):
#     os.makedirs(path)
# torch.save(performances, path + f'/perf_{best_psnr:.4}_beta_{best_beta:.2e}_lmbda_{best_lambda:.2e}.pth')
# torch.save(final_preds, 'final_img.pth')
# torch.save(final_coeffs, 'final_coeffs.pth')