import math
import torch
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from utils import utilities
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0)


def L_tv(x, filters): 
    filters1 = filters[0]; filters2 = filters[1]
    
    L1 = F.pad(F.conv2d(x, filters1), (0, 1), "constant", 0)
    L2 = F.pad(F.conv2d(x, filters2), (0, 0, 0, 1), "constant", 0)

    Lx = torch.cat((L1, L2), dim=1)

    return Lx 

def Lt_tv(y, filters): 
    filters1 = filters[0]; filters2 = filters[1]

    L1t = F.conv_transpose2d(y[:, 0:1, :, :-1], filters1)
    L2t = F.conv_transpose2d(y[:, 1:2, :-1, :], filters2)

    Lty = L1t + L2t

    return Lty

def prox_tv(y, niter, lmbda): 

    filters1 = torch.Tensor([[[[1., -1]]]]).to(device).double()
    filters2 = torch.Tensor([[[[1], [-1]]]]).to(device).double()
    filters = [filters1, filters2]

    v_k = torch.zeros((1, 2, y.shape[0], y.shape[1]), requires_grad=False, device=device).double()
    u_k = torch.zeros((1, 2, y.shape[0], y.shape[1]), requires_grad=False, device=device).double()

    t_k = 1
    alpha = 1/(8*lmbda)

    for _ in range(niter):
        Ltv = Lt_tv(v_k, filters)
        pc = torch.clip(y - lmbda * Ltv, 0, 1)
        Lpc = L_tv(pc, filters)

        temp = v_k + alpha * Lpc
        
        u_kp1 = torch.nn.functional.normalize(temp, eps=1, dim=1, p=2)

        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

        u_k = u_kp1
        v_k = v_kp1
        t_k = t_kp1

    Ltu = Lt_tv(u_k, filters)
    c = torch.clip(y - lmbda * Ltu, 0, 1)

    return c  

device = 'cuda:3'
mask = torch.tensor(sio.loadmat('cs_mri/Q_Cartesian30.mat').get('Q1')).float().to(device)
img = torch.tensor(plt.imread('cs_mri/Bust.jpg')).view(1, 1, 256, 256).float().to(device) / 255
y = torch.fft.fft2(img, dim=(-2, -1), norm='ortho')*mask + (10/255)*torch.randn(img.shape, dtype=torch.complex128, device=device)
Hty = torch.fft.ifft2(y*mask, dim=(-2, -1), norm='ortho').real.type(torch.float32)

max_iter = 5000
tol = 1e-6
lmbd = 7.25e-3
alpha = 1.0
tv_iter = 100

x = Hty.clone()
z = Hty.clone()
t = 1

with torch.no_grad():
    pbar = tqdm(range(max_iter), dynamic_ncols=True)
    for i in pbar:
        x_old = torch.clone(x)
        HtHx = torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1), norm='ortho')*mask, dim=(-2, -1), norm='ortho').real
        x = torch.clip(z - alpha*(HtHx - Hty), 0., 1.)
        x = prox_tv(x.squeeze(), tv_iter, lmbd*alpha)

        t_old = t 
        t = 0.5 * (1 + math.sqrt(1 + 4*t**2))

        z = x + (t_old - 1)/t * (x - x_old)

        # relative change of norm for terminating
        res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
        pbar.set_description(f"res: {res:.2e}")
        print(f'PSNR is {utilities.batch_PSNR(x, img, 1.):.4f}')
        if res < tol:
            break
    
plt.imsave(f'cs_mri/tv_reconstruction_{lmbd:.2e}_psnr_{utilities.batch_PSNR(x, img, 1.):.4f}.png', x.cpu()[0,0,...], cmap='gray')