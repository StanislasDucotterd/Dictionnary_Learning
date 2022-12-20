import torch
from dataloader.BSD500 import BSD500
from models.dictionnary import Dictionnary
from utils import metrics, utilities
torch.manual_seed(42)
device = 'cpu'

infos = torch.load('exps/sigma_5/Norm_reg_0.0275_stride_1_lr_1e-3_1e-3_nd_50_d_50_gamma_0.9_batch_4/checkpoints/checkpoint_best_epoch.pth')
config = infos['config']
weights = infos['state_dict']


model = Dictionnary(config['prox_params'], config['dict_params']['nb_atoms'], config['dict_params']['nb_channels'],\
                    config['dict_params']['atom_size'], config['dict_params']['stride'],config['dict_params']['nondiff_steps'],\
                    config['dict_params']['diff_steps'],, config['dict_params']['init'])
model.load_state_dict(weights)
model.to(device)
model.eval()

val_dataset = BSD500('/home/ducotter/Lipschitz_DSNN/data/BSD500/test.h5')
psnr_val = 0
ssim_val = 0

for i in range(68):
    print(i)
    data = val_dataset.__getitem__(i).unsqueeze(0).to(device)
    noisy_data = data + (5/255) * torch.randn(data.shape, device=device)
    output = model(noisy_data)
    out_val = torch.clamp(output, 0., 1.)
    psnr_val += utilities.batch_PSNR(out_val, data, 1.) / 68
    ssim_val += utilities.batch_SSIM(out_val, data, 1.) / 68
print(psnr_val)
print(ssim_val)