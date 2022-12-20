import torch
import os
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from dataloader.BSD500 import BSD500
from models.dictionnary import Dictionnary
from models.convhull_dictionnary import ConvHull_Dictionnary
from utils import metrics, utilities
import matplotlib.pyplot as plt

class TrainerDictionnary:
    """
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        self.train_dataset = BSD500(config['training_options']['train_data_file'])
        val_dataset = BSD500(config['training_options']['val_data_file'])

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True,\
                                             num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.batch_size = config["training_options"]["batch_size"]
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        print('Building the model')
        # Build the model

        self.model = Dictionnary(config['prox_params'], config['dict_params']['nb_atoms'], config['dict_params']['nb_channels'],\
                    config['dict_params']['atom_size'], config['dict_params']['stride'],config['dict_params']['nondiff_steps'],\
                    config['dict_params']['diff_steps'], config['dict_params']['init'])
        
        self.model = self.model.to(device)

        print(self.model)
        
        self.epochs = config["training_options"]['epochs']
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        params_list = [{'params': self.model.atoms, 'lr': config['training_options']['lr']}]
        if config['prox_params']['learn_threshold']:
            params_list.append({'params': self.model.proximal.threshold, 'lr': config['training_options']['lr_threshold']})
        if config['prox_params']['prox_type'] == 'learnable':
            params_list.append({'params': self.model.proximal.coefficients_vect, 'lr': config['training_options']['lr_spline']})
        if config['prox_params']['linearity'] != 'none':
            params_list.append({'params': self.model.proximal.weights, 'lr': config['training_options']['lr_linearity']})

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training_options']['lr'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['training_options']['gamma'])

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
        self.total_eval_step = 1
        self.best_psnr = 0
        

    def train(self):
        
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):
        """
        """
        self.model.train()

        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for idx, data in enumerate(tbar):
            data = data.to(self.device)
            noisy_image = data + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

            self.optimizer.zero_grad()
            output = self.model(noisy_image)

            loss = (self.criterion(output, data))/(self.batch_size)
            loss.backward()
            self.optimizer.step()
                
            log['train_loss'] = loss.detach().cpu().item()

            # We report the metrics after the network has seen a certain amount of data
            # This was done to compare the training loss with different gradient steps
            if (self.total_training_step) % (len(tbar) // 1000)  == 0:
                self.wrt_step = self.total_training_step * self.batch_size
                self.write_scalars_tb(log)

            tbar.set_description('T ({}) | TotalLoss {:.5f} |'.format(epoch, log['train_loss'])) 
            self.total_training_step += 1

            if (idx+1) % (len(tbar) // 10)  == 0:
                if self.total_eval_step % 1 == 0:
                    self.scheduler.step()
                val_psnr = self.valid_epoch()
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.save_checkpoint(self.total_eval_step)



    def valid_epoch(self):
        
        self.model.eval()
        loss_val = 0.0
        psnr_val = []
        ssim_val = []

        tbar_val = tqdm(self.val_dataloader, ncols=130, position=0, leave=True)
        
        with torch.no_grad():
            for data in tbar_val:
                data = data.to(self.device)
                noisy_image = data + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

                output = self.model(noisy_image)
                loss = self.criterion(output, data)

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
        
        return np.mean(psnr_val)


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint_best_epoch.pth'
        torch.save(state, filename)