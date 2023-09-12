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
dataset = BSD500('/home/ducotter/nerf_dip/images/val.h5')
sigma = float(model_name.split('/')[0].split('_')[1])

def HtH(x):
    return x

batch1 = torch.zeros(15, 1, 481, 321)
batch2 = torch.zeros(18, 1, 321, 481)
batch3 = torch.zeros(18, 1, 321, 481)
batch4 = torch.zeros(17, 1, 321, 481)
index1, index2 = 0, 0

# for i in range(dataset.__len__()):
#     img = dataset.__getitem__(i)
#     if img.shape[1] == 321:
#         if index2 >= 18 and index2 < 36: batch3[index2-18] = img
#         elif index2 >= 36: batch4[index2-36] = img
#         else: batch2[index2] = img
#         index2 += 1
#     else:
#         batch1[index1] = img
#         index1 += 1
# imgs = [batch1, batch2, batch3, batch4]

batch1 = torch.zeros(7, 1, 256, 256)
batch2 = torch.zeros(5, 1, 512, 512)
index1, index2 = 0, 0
for i in range(dataset.__len__()):
    img = dataset.__getitem__(i)
    if img.shape[1] == 256:
        batch1[index1] = img
        index1 += 1
    else:
        batch2[index2] = img
        index2 += 1

imgs = [batch1, batch2]
imgs = [dataset.__getitem__(1).unsqueeze(0)]

atoms = infos['state_dict']['atoms'].to(device)
atom_size = config['dict_params']['atom_size']
nb_atoms = config['dict_params']['nb_atoms']

mean_filter = torch.ones(1, 1, atom_size, atom_size, device=device) / (atom_size**2)
atoms = atoms - torch.mean(atoms, dim=(1,2,3), keepdim=True)
atoms = atoms / torch.linalg.norm(atoms.reshape(nb_atoms, -1), axis=1, ord=2).reshape(-1, 1, 1, 1)
atoms = (atoms / (torch.linalg.matrix_norm(atoms.reshape(nb_atoms, -1), 2)))
X = atoms.reshape(nb_atoms, -1) @ atoms.reshape(nb_atoms, -1).T
L1 = 1.01
res_criteria = 1e-8

base_beta = 0.6
base_lambda = 7.75e-5
gamma = 1.1
betas = [base_beta/gamma, base_beta, base_beta*gamma]
lambdas = [base_lambda/gamma, base_lambda, base_lambda*gamma]
betas = [0.6]
lambdas = [7.75e-5]
psnrs = torch.zeros(len(betas), len(lambdas))
 
for j, lambda_ in enumerate(lambdas):
    prox = nn.Softshrink(lambda_/L1)
    for i, beta in enumerate(betas):
        L2 = 1.01*(beta*atom_size**2+1)
        start = time.time()   
        for true_img in imgs:
            boat_infos = {}
            boat_infos['obj_values'] = []
            boat_infos['psnr_values'] = []
            true_img = true_img.to(device)
            y = true_img + (sigma/255) * torch.randn(true_img.shape, device=device)
            img_height, img_width = true_img.shape[2], true_img.shape[3]
            n_patch = (img_height-atom_size+1)*(img_width-atom_size+1)
            batch_size = true_img.shape[0]
            patch_y_mean = F.conv2d(y, mean_filter).reshape(batch_size*n_patch, 1, 1, 1)
            img, new_img, temp_img, temp_y = y.clone(), y.clone(), y.clone(), y.clone()
            coeffs = torch.zeros(batch_size*n_patch, nb_atoms, device=device)
            new_coeffs, temp_coeffs = coeffs.clone(), coeffs.clone()
            final_preds = torch.zeros(true_img.shape, device=device) * float('nan')
            max_res, obj_value, k = 1.0, 1e6, 1
            while max_res > res_criteria:
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
                new_img = torch.clip(temp_img - (apply_op(temp_img, HtH, atom_size, beta) - temp_y - beta*dict_pred) / L2, 0., 1.)
                dict_patches = torch.einsum('ij,jlmn->ilmn', coeffs, atoms)
                if not new_method: dict_patches = dict_patches + patch_y_mean
                new_obj_value = compute_obj(img, temp_y, dict_patches, coeffs)
                res = (obj_value - new_obj_value).abs() / obj_value
                obj_value = new_obj_value
                boat_infos['obj_values'].append(obj_value)
                boat_infos['psnr_values'].append(utilities.batch_PSNR(img, true_img, 1.))
                condition = res < res_criteria
                if condition.sum().item() > 0:
                    final_nan_indices = final_preds.view(final_preds.shape[0], -1).isnan().any(dim=1)
                    temp_var = torch.zeros(final_nan_indices.sum(), final_preds.shape[1], final_preds.shape[2], final_preds.shape[3], device=y.device)*float('nan')
                    temp_var[condition,:] = img[condition,:]
                    final_preds[final_nan_indices,:] = temp_var
                    coeffs, new_coeffs = coeffs.view(batch_size, n_patch, nb_atoms), new_coeffs.view(batch_size, n_patch, nb_atoms)
                    temp_coeffs = temp_coeffs.view(batch_size, n_patch, nb_atoms)
                    coeffs, new_coeffs = coeffs[~condition,:].view(-1, nb_atoms), new_coeffs[~condition,:].view(-1, nb_atoms)
                    temp_coeffs = temp_coeffs[~condition,:].view(-1, nb_atoms)
                    patch_y_mean = patch_y_mean.view(batch_size, n_patch, 1, 1)
                    patch_y_mean = patch_y_mean[~condition,:].view(-1, 1, 1, 1)
                    img, new_img, temp_img, temp_y = img[~condition,:], new_img[~condition,:], temp_img[~condition,:], temp_y[~condition,:]
                    obj_value = obj_value[~condition]
                    batch_size = temp_y.shape[0]
                if k > 1e4:
                    final_nan_indices = final_preds.view(final_preds.shape[0], -1).isnan().any(dim=1)
                    final_preds[final_nan_indices,:] = img
                    break
                    print("SOME IMAGE DID NOT CONVERGE")
                max_res = torch.max(res)
                k += 1
            psnrs[i,j] += final_preds.shape[0]*utilities.batch_PSNR(final_preds, true_img, 1.)/12
            torch.save(boat_infos, f'ipalm_boat_infos_sigma_{sigma}_beta_{beta}_lambda_{lambda_}.pth')
        print(f'Beta: {beta:.6}, Lambda: {lambda_:.6}, PSNR: {psnrs[i,j]:.6}, it took {time.time()-start:.2f}s')
best_psnr = psnrs[~psnrs.isnan()].max()
best_beta = betas[(psnrs == best_psnr).nonzero()[0,0]]
best_lambda = lambdas[(psnrs == best_psnr).nonzero()[0,1]]
print(f'Best PSNR is {best_psnr:.6} with beta={best_beta:.6} and lambda={best_lambda:.6}')
performances = {'psnrs': psnrs, 'betas': betas, 'lambdas': lambdas}

# path = 'denoising/' + model_name
# if not os.path.exists(path):
#     os.makedirs(path)
# torch.save(performances, path + f'/perf_{best_psnr:.4}_beta_{best_beta:.2e}_lmbda_{best_lambda:.2e}.pth')