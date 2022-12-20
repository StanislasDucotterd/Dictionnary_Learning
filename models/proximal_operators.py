import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def choose_prox(config, nb_atoms):
    if config['prox_type'] == 'norm':
        return NormProx(nb_atoms, config['groupsize'], config['threshold'], config['learn_threshold'], config['linearity'])
    elif config['prox_type'] == 'learnable':
        return LearnProx(nb_atoms, config['groupsize'], config['spline_size'], config['spline_range'], config['linearity'])
    else:
        raise ValueError('Prox type is not valid')

class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):

        # The value of the spline at any x is a combination 
        # of at most two coefficients
        max_range = (grid.item() * (size // 2 - 1))
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=max_range)

        floored_x = torch.floor(x_clamped / grid)  #left coefficient
        fracs = x / grid - floored_x  # distance to left coefficient
        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()
        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + coefficients_vect[indexes] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (coefficients_vect[indexes + 1] - coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.reshape(-1) + 1, (fracs * grad_out).reshape(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.reshape(-1), ((1 - fracs) * grad_out).reshape(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

class NormProx(torch.nn.Module):

    def __init__(self, nb_atoms, groupsize, threshold, learn_threshold, linearity):
        super(NormProx, self).__init__()

        self.groupsize = groupsize
        self.nb_groups = nb_atoms // groupsize
        self.linearity = linearity
        self.learn_threshold = learn_threshold
        if learn_threshold:
            self.threshold = nn.Parameter(threshold * torch.ones(nb_atoms, 1))
        else:
            self.threshold = threshold
        if linearity != 'none':
            self.weights = nn.Parameter(torch.eye(nb_atoms))

    def project(self):
        if self.linearity == 'orthonormal':
            w = self.weights / math.sqrt(self.weights.shape[0] * self.weights.shape[1])
            for _ in range(25):
                w_t_w = w.t().mm(w)
                w = 1.5 * w - 0.5 * w.mm(w_t_w)
            self.projected_weights = w

    def forward(self, x, L):
        if self.learn_threshold: 
            threshold = F.relu(self.threshold) 
            #threshold = 0.0275 * self.nb_groups * threshold / torch.sum(threshold)
        else: threshold = self.threshold

        if self.linearity != 'none': x = self.projected_weights @ x

        if self.groupsize == 1: x = x + threshold / L - F.relu(x + threshold / L) + F.relu(x - threshold / L)
        else:
            batch_size = x.shape[1]
            x = x.reshape(self.nb_groups, self.groupsize, batch_size)
            norm_x = torch.linalg.norm(x, dim=1, ord=2).unsqueeze(1)
            x = x * F.relu(norm_x - threshold / L) / (norm_x + 1e-8)
            x = x.reshape(self.nb_groups * self.groupsize, batch_size)

        if self.linearity != 'none': x = self.projected_weights.T @ x
    
        return x

class LearnProx(torch.nn.Module):
    
    def __init__(self, nb_atoms, groupsize, spline_size, spline_range, init):
        super(LearnProx, self).__init__()

        self.nb_atoms = nb_atoms
        self.spline_size = spline_size
        self.grid = torch.Tensor([2 * spline_range / (spline_size-1)])
        self.zero_knot_indexes = (torch.arange(0, nb_atoms) * spline_size + (spline_size // 2))
        self.D2_filter = torch.Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)
        self.grid_tensor = torch.linspace(-spline_range, spline_range, spline_size).expand((nb_atoms, spline_size))
        coefficients = self.grid_tensor
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))

    @property
    def coefficients(self):
        return self.coefficients_vect.view(self.nb_atoms, self.spline_size)
    
    def project(self):

        new_slopes = torch.clamp(self.coefficients[:,1:] - self.coefficients[:,:-1], 0, self.grid.item())
        self.projected_coeffs = torch.zeros(self.coefficients.shape, device=self.coefficients.device)
        self.projected_coeffs[:,1:] = torch.cumsum(new_slopes, dim=1)
        self.projected_coeffs = self.projected_coeffs + torch.mean(self.coefficients - self.projected_coeffs, dim=1).unsqueeze(1)
        self.projected_coeffs = self.projected_coeffs.contiguous().view(-1)

    def forward(self, x, L):

        x = x.T.reshape(*x.T.shape, 1, 1)
        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        output = LinearSpline_Func.apply(x, self.projected_coeffs, grid, zero_knot_indexes, self.spline_size)
        output = torch.squeeze(output).T

        return output