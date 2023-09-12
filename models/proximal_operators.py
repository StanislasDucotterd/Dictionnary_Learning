import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractproperty, abstractmethod
from models.linearspline import LinearSpline

def choose_prox(nb_atoms, config):
    if config['prox_type'] == 'l1norm':
        return L1NormProx(config['threshold'])
    elif config['prox_type'] == 'learn':
        return LearnProx(nb_atoms, config['spline_size'], config['spline_range'], config['spline_init'])
    else:
        raise ValueError('Prox type is not valid')

class L1NormProx(nn.Module):

    def __init__(self, threshold):
        super(L1NormProx, self).__init__()

        self.threshold = threshold
        self.soft_threshold = nn.Softshrink(lambd=self.threshold)

    def forward(self, x):
        return self.soft_threshold(x)

    def jacobian(self, x):
        return torch.where(torch.abs(x) > self.threshold, torch.tensor(1., device=x.device), torch.tensor(0., device=x.device))
        
class LearnProx(ABC, nn.Module):
    
    def __init__(self, nb_atoms, spline_size, spline_range, spline_init):
        super().__init__()

        self.linearspline = LinearSpline('fc', nb_atoms, spline_size, spline_range, spline_init)

    def forward(self, x):
        return self.linearspline(x)

    def jacobian(self, x):

        max_range = (self.linearspline.grid.item() * (self.linearspline.size // 2 - 1))
        x_clamped = x.clamp(min=-(self.linearspline.grid.item() * (self.linearspline.size // 2)), max=max_range)
        floored_x = torch.floor(x_clamped / self.linearspline.grid.to(x.device))  
        indexes = (self.linearspline.zero_knot_indexes.to(x.device) + floored_x).long()
        x = (self.linearspline.lipschitz_coefficients_vect[indexes + 1] - \
             self.linearspline.lipschitz_coefficients_vect[indexes]) / self.linearspline.grid.item()

        return x