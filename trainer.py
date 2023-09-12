import torch
import os
import json
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from dataloader.BSD500 import BSD500
from models.dictionnary import Dictionnary
from utils import metrics, utilities
import matplotlib.pyplot as plt
import gc

class TrainerDictionnary:
    """
    """
    def __init__(self, config, device):

        self.config = config
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        train_dataset = BSD500(config['training_options']['train_data_file'])
        val_dataset = BSD500(config['training_options']['val_data_file'])

        print('Preparing the dataloaders')
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True,\
                                             num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        self.batch_size = config["training_options"]["batch_size"]

        print('Building the model')
        self.model = Dictionnary(config['prox_params'], config['dict_params']['nb_atoms'], config['dict_params']['nb_channels'],\
                    config['dict_params']['atom_size'], config['dict_params']['res'], config['dict_params']['beta'],\
                    config['dict_params']['mu'], config['dict_params']['hyper_learnable'], config['dict_params']['unroll_steps'])
        self.model = self.model.to(device)
        print(self.model)
        
        self.epochs = config["training_options"]['epochs']
        
        dict_params = [{'params': self.model.atoms, 'lr': config['training_options']['lr'], \
                        'weight_decay': config['training_options']['weight_decay']}]
        model_params = []
        if config['prox_params']['prox_type'] == 'learn':
            model_params.append({'params': self.model.proximal.linearspline.coefficients_vect, 'lr': config['training_options']['lr_spline']})
            model_params.append({'params': self.model.proximal.linearspline.scaling_coeffs_vect, 'lr': config['training_options']['lr_spline_scaling']})
        if config['dict_params']['hyper_learnable']:
            #model_params.append({'params': [self.model.beta, self.model.mu], 'lr': config['training_options']['lr_hyper']})
            model_params.append({'params': [self.model.mu], 'lr': config['training_options']['lr_hyper']})

        self.update_model = False
        self.optimizer_dict = torch.optim.Adam(dict_params, betas=(0.9, 0.999))
        self.scheduler_dict = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_dict, gamma=config['training_options']['lr_decay'])
        if len(model_params) > 0:
            self.optimizer_model = torch.optim.Adam(model_params)
            self.scheduler_model = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_model, gamma=config['training_options']['lr_decay'])
            self.update_model = True
        
        if self.config['training_options']['loss'] == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.L1Loss()

        # CHECKPOINTS & TENSOBOARD
        run_name = config['exp_name']
        self.checkpoint_dir = os.path.join(config['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(config['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

        self.total_training_step = 0
        self.total_eval_step = 0
        self.best_psnr = 0
        

    def train(self):
        
        #self.valid_epoch()
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)

        if self.config['prox_params']['prox_type'] == 'learn':
            with torch.no_grad():
                x = self.model.proximal.linearspline.grid_tensor
                y = self.model.proximal.linearspline.lipschitz_coefficients
                figures_list = []
                for kk in range(x.shape[0]):
                    fig, ax = plt.subplots()
                    ax.grid()
                    ax.plot(x[kk,:].cpu().numpy(), y[kk,:].cpu().numpy())
                    figures_list.append(fig)
                    plt.close()
                self.writer.add_figure('Activation functions', figures_list, global_step=self.total_eval_step)

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):

        self.model.train()
        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for idx, data in enumerate(tbar):
            image = data.to(self.device)
            noisy_image = image + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

            self.optimizer_dict.zero_grad()
            if self.update_model: self.optimizer_model.zero_grad()
            output = self.model(noisy_image)
            data_fitting = self.criterion(output, image)

            regularization = torch.zeros_like(data_fitting)
            if self.config['prox_params']['prox_type'] == 'learn' and self.config['prox_params']['spline_lambda'] > 0:
                regularization = self.config['prox_params']['spline_lambda'] * self.model.proximal.linearspline.totalVariation().sum()
            loss = data_fitting + regularization
            loss.backward()
            self.optimizer_dict.step()
            if self.update_model: self.optimizer_model.step()
                
            log['train_loss'] = data_fitting.detach().cpu().item()

            if self.config['dict_params']['hyper_learnable']:
                log['beta'] = F.relu(self.model.beta).item()
                log['mu'] = torch.mean(F.relu(self.model.mu)).item()

            if self.config['prox_params']['prox_type'] == 'learn':
                log['coeff_mean'] = torch.mean(self.model.proximal.linearspline.coefficients_vect).item()
                log['coeff_std'] = torch.std(self.model.proximal.linearspline.coefficients_vect).item()
                log['alpha_mean'] = torch.mean(self.model.proximal.linearspline.scaling_coeffs_vect).item()
                log['alpha_std'] = torch.std(self.model.proximal.linearspline.scaling_coeffs_vect).item()

            # We report the metrics after the network has seen a certain amount of data
            # This was done to compare the training loss with different gradient steps
            if (self.total_training_step) % max((len(tbar) // 1000), 1)  == 0:
                self.wrt_step = self.total_training_step * self.batch_size
                self.write_scalars_tb(log)

            if (idx+1) % (len(tbar) // 5)  == 0:
                val_psnr, output = self.valid_epoch()
                self.scheduler_dict.step()
                if self.update_model: self.scheduler_model.step()
                self.writer.add_image('my_image', output[0,0,...], self.total_eval_step, dataformats='HW')
                
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.save_checkpoint('/best_checkpoint')

            tbar.set_description('T ({}) | TotalLoss {:.5f} |'.format(epoch, log['train_loss'])) 
            self.total_training_step += 1

    def valid_epoch(self):
        
        self.model.eval()
        loss_val = 0.0
        psnr_val, ssim_val = [], []
        tbar_val = tqdm(self.val_dataloader, ncols=130, position=0, leave=True)
        
        with torch.no_grad():
            for data in tbar_val:
                image = data.to(self.device)
                noisy_image = image + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)
                output = self.model.val_forward(noisy_image)
                loss = self.criterion(output, image)
                loss_val = loss_val + loss.cpu().item()
                out_val = torch.clamp(output, 0., 1.)

                psnr_val.append(utilities.batch_PSNR(out_val, data, 1.))
                ssim_val.append(utilities.batch_SSIM(out_val, data, 1.))
            
            # PRINT INFO
            loss_val = loss_val/len(self.val_dataloader)
            tbar_val.set_description('EVAL ({}) | MSELoss: {:.5f} |'.format(self.total_eval_step, loss_val))

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'
            self.writer.add_scalar(f'{self.wrt_mode}/loss', loss_val, self.total_eval_step)
            self.writer.add_scalar(f'{self.wrt_mode}/Test PSNR Mean', np.mean(psnr_val), self.total_eval_step)
            self.writer.add_scalar(f'{self.wrt_mode}/Test SSIM Mean', np.mean(ssim_val), self.total_eval_step)

        self.total_eval_step += 1
        self.model.train()
        
        return np.mean(psnr_val), out_val


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self, name):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer_dict.state_dict(),
            'config': self.config
        }
        if self.update_model: state['optimizer_model'] = self.optimizer_model.state_dict()

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + name + '.pth'
        torch.save(state, filename)