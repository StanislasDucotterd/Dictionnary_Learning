import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_obj(x, y, atom_size, nb_atoms, beta, lambda_, dict_patches, alphas, new_method):
    n_patch = (x.shape[2]-atom_size+1)*(x.shape[3]-atom_size+1)
    patches = F.unfold(x, kernel_size=atom_size)
    if new_method: patches = patches - patches.mean(dim=1, keepdim=True)
    dict_patches = dict_patches.view(x.shape[0], n_patch, atom_size**2).permute(0, 2, 1)
    alphas = alphas.view(x.shape[0], n_patch, nb_atoms)
    return 0.5*((x-y)**2).sum(dim=(1,2,3)) + 0.5*beta*((patches-dict_patches)**2).sum(dim=(1,2)) \
           + beta*lambda_*alphas.abs().sum(dim=(1, 2))

def apply_op(x, HtH, atom_size, beta, new_method):
    patches = F.unfold(x, kernel_size=atom_size)
    if new_method: patches = patches - patches.mean(dim=1, keepdim=True)
    return HtH(x) + beta*F.fold(patches, output_size=(x.shape[2], x.shape[3]), kernel_size=atom_size)

def get_coeffs_ista(starting_coeffs, X, q, prox, res_criteria):
    coeffs, new_coeffs = starting_coeffs.clone(), starting_coeffs.clone()
    final_coeffs = torch.zeros(q.shape, device=X.device) * float('nan')
    temp_q, max_res, obj_value = q, 1.0, 1e6
    iteration = 0
    while max_res > res_criteria:
        iteration += 1
        new_coeffs = prox(coeffs - (coeffs @ X - temp_q))
        res = torch.linalg.norm(coeffs-new_coeffs, ord=1, dim=1) / (torch.linalg.norm(coeffs, ord=1, dim=1) + 1e-8)
        condition = res < res_criteria
        if condition.sum().item() > 0:
            temp_var = torch.zeros(final_coeffs[final_coeffs.isnan().any(dim=1),:].shape, device=X.device) * float('nan')
            temp_var[condition,:] = new_coeffs[condition,:]
            final_coeffs[final_coeffs.isnan().any(dim=1),:] = temp_var
            coeffs = new_coeffs[~condition,:]
            temp_q = temp_q[~condition,:]
        else:
            coeffs = new_coeffs
        max_res = torch.max(res)
        if iteration > 1e4:
            final_coeffs[final_coeffs.isnan().any(dim=1),:] = new_coeffs
            print('SOME COEFFICIENTS DID NOT CONVERGE')
            break
    return final_coeffs

def get_coeffs_fista(starting_coeffs, X, q, prox, res_criteria):
    coeffs, new_coeffs, temp_coeffs = starting_coeffs.clone(), starting_coeffs.clone(), starting_coeffs.clone()
    final_coeffs = torch.zeros(q.shape, device=X.device) * float('nan')
    temp_q, max_res, t = q, 1.0, 1.0
    iteration = 0
    while max_res > res_criteria:
        iteration += 1
        new_coeffs = prox(temp_coeffs - (temp_coeffs @ X - temp_q))
        res = torch.linalg.norm(coeffs-new_coeffs, ord=1, dim=1) / (torch.linalg.norm(coeffs, ord=1, dim=1) + 1e-8)
        new_t = (1 + math.sqrt(4*t**2 + 1)) / 2
        temp_coeffs = new_coeffs + (t-1) * (new_coeffs - coeffs) / new_t
        condition = res < res_criteria
        if condition.sum().item() > 0:
            temp_var = torch.zeros(final_coeffs[final_coeffs.isnan().any(dim=1),:].shape, device=X.device) * float('nan')
            temp_var[condition,:] = new_coeffs[condition,:]
            final_coeffs[final_coeffs.isnan().any(dim=1),:] = temp_var
            coeffs = new_coeffs[~condition,:]
            temp_q = temp_q[~condition,:]
            temp_coeffs = temp_coeffs[~condition,:]
        else:
            coeffs = new_coeffs
        max_res = torch.max(res)
        t = new_t
        if iteration > 1e4:
            final_coeffs[final_coeffs.isnan().any(dim=1),:] = new_coeffs
            print('SOME COEFFICIENTS DID NOT CONVERGE')
            break
    return final_coeffs

def get_img_fista(starting_img, HtH, dict_pred, Hty, beta, atom_size, res_criteria, new_method):
    L, max_res_img, t = beta*atom_size**2+1, 1.0, 1.0
    img, new_img, temp_img, temp_Hty = starting_img.clone(), starting_img.clone(), starting_img.clone(), Hty.clone()
    final_img = torch.zeros(starting_img.shape, device=starting_img.device) * float('nan')
    while max_res_img > res_criteria:
        new_img = torch.clip(temp_img-(apply_op(temp_img, HtH, atom_size, beta, new_method)-temp_Hty-beta*dict_pred)/L, 0., 1.)
        res = (img-new_img).abs().sum(dim=(1,2,3))/img.abs().sum(dim=(1,2,3))
        new_t = (1 + math.sqrt(4*t**2 + 1)) / 2
        temp_img = new_img + (t-1)*(new_img-img)/new_t
        condition = res < res_criteria
        if condition.sum().item() > 0:
            nan_indices = final_img.view(final_img.shape[0], -1).isnan().any(dim=1)
            temp_var = torch.zeros(nan_indices.sum(), final_img.shape[1], final_img.shape[2],\
                                    final_img.shape[3], device=Hty.device)*float('nan')
            temp_var[condition,:] = new_img[condition,:]
            final_img[nan_indices,:] = temp_var
            img = new_img[~condition,:]
            dict_pred = dict_pred[~condition,:]
            temp_Hty = temp_Hty[~condition,:]
            temp_img = temp_img[~condition,:]
        else:
            img = new_img
        max_res_img = torch.max(res)
        t = new_t
    return final_img

def ipalm(y, beta, lambda_, atoms, X, HtH, new_method, res_criteria):
    img_height, img_width = y.shape[2], y.shape[3]
    atom_size = atoms.shape[2]
    n_patch = (img_height-atom_size+1)*(img_width-atom_size+1)
    batch_size = y.shape[0]
    mean_filter = torch.ones(1, 1, atom_size, atom_size, device=y.device) / (atom_size**2)
    patch_y_mean = F.conv2d(y, mean_filter).reshape(batch_size*n_patch, 1, 1, 1)
    img, new_img, temp_img, temp_y = y.clone(), y.clone(), y.clone(), y.clone()
    nb_atoms = X.shape[1]
    coeffs = torch.zeros(batch_size*n_patch, nb_atoms, device=y.device)
    new_coeffs, temp_coeffs = coeffs.clone(), coeffs.clone()
    L1, L2 = 1.01, 1.01*(beta*atom_size**2+1)
    final_preds = torch.zeros(y.shape, device=y.device) * float('nan')
    final_obj_values = torch.zeros(batch_size, device=y.device)
    prox = nn.Softshrink(lambda_/L1)
    max_res, obj_value, k = 1.0, 1e6, 1
    while max_res > res_criteria:
        q = F.conv2d(img, atoms).permute(0, 2, 3, 1).reshape(batch_size*n_patch, nb_atoms)
        temp_coeffs = new_coeffs + (k-1) * (new_coeffs - coeffs) / (k+2)
        coeffs = new_coeffs
        new_coeffs = prox(temp_coeffs - (temp_coeffs @ X - q) / L1)
        dict_patches = torch.einsum('ij,jlmn->ilmn', new_coeffs, atoms)
        if not new_method: dict_patches = dict_patches + patch_y_mean
        dict_pred = dict_patches.view(batch_size, n_patch, atom_size**2).permute(0, 2, 1)
        dict_pred = F.fold(dict_pred, output_size=(img_height, img_width), kernel_size=atom_size)
        temp_img = new_img + (k-1) * (new_img - img) / (k+2)
        img = new_img
        new_img = torch.clip(temp_img - (apply_op(temp_img, HtH, atom_size, beta) - temp_y - beta*dict_pred) / L2, 0., 1.)
        dict_patches = torch.einsum('ij,jlmn->ilmn', coeffs, atoms)
        if not new_method: dict_patches = dict_patches + patch_y_mean
        new_obj_value = compute_obj(img, temp_y, atom_size, nb_atoms, beta, lambda_, dict_patches, coeffs, new_method)
        res = (obj_value - new_obj_value).abs() / obj_value
        obj_value = new_obj_value
        condition = res < res_criteria
        if condition.sum().item() > 0:
            final_nan_indices = final_preds.view(final_preds.shape[0], -1).isnan().any(dim=1)
            temp_var = torch.zeros(final_nan_indices.sum(), final_preds.shape[1], final_preds.shape[2], final_preds.shape[3], device=y.device)*float('nan')
            temp_obj_values = torch.zeros(final_nan_indices.sum(), device=y.device)*float('nan')
            temp_var[condition,:] = img[condition,:]
            temp_obj_values[condition] = obj_value[condition]
            final_preds[final_nan_indices,:] = temp_var
            final_obj_values[final_nan_indices] = temp_obj_values
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
        max_res = torch.max(res)
        k += 1
    return final_preds, final_obj_values