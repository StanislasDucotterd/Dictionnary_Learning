import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod


def slope_clipping(cs, T):
    device = cs.device
    new_slopes = torch.clamp(cs[:,1:] - cs[:,:-1], 0., T)
    new_cs = torch.zeros(cs.shape, device=device)
    new_cs[:,1:] = torch.cumsum(new_slopes, dim=1)
    new_cs = new_cs - new_cs[:,new_cs.shape[1]//2].unsqueeze(1)

    return new_cs

def initialize_coeffs(init, grid_tensor, grid, size):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        
        if init == 'identity':
            coefficients = grid_tensor
        elif init == 'relu':
            coefficients = F.relu(grid_tensor)
        elif init == 'zero':
            coefficients = torch.zeros(grid_tensor.shape)
        elif init == 'soft_threshold':
            coefficients = F.softshrink(grid_tensor, 3e-3)
        else:
            raise ValueError('init should be in [identity, relu, zero].')
        
        return coefficients


class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size, even):
        # The value of the spline at any x is a combination 
        # of at most two coefficients
        max_range = (grid.item() * (size // 2 - 1))
        if even:
            x = x - grid / 2
            max_range = (grid.item() * (size // 2 - 2))
        x_clamped = x.clamp(min=-(grid.item() * (size // 2)), max=max_range)

        floored_x = torch.floor(x_clamped / grid)  #left coefficient
        #fracs = x_clamped / grid - floored_x
        fracs = x / grid - floored_x  # distance to left coefficient
        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()
        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)
        if even:
            activation_output = activation_output + grid / 2

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

class LinearSpline(ABC, nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        mode (str): 'conv' (convolutional) or 'fc' (fully-connected).
        num_activations (int) : number of activation functions
        size (int): number of coefficients of spline grid; the number of knots K = size - 2.
        range_ (float) : positive range of the B-spline expansion. B-splines range = [-range_, range_].
        init (str): Function to initialize activations as (e.g. 'relu', 'identity', 'absolute_value').
    """

    def __init__(self, mode, num_activations, size, range_, init, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations needs to be a '
                            'positive integer...')

        super().__init__()

        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)
        grid = 2 * self.range_ / (self.size-1)
        self.grid = torch.Tensor([grid])

        self.init_zero_knot_indexes()
        self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)

        # tensor with locations of spline coefficients
        self.grid_tensor = torch.linspace(-self.range_, self.range_, self.size).expand((self.num_activations, self.size))
        coefficients = initialize_coeffs(init, self.grid_tensor, self.grid, self.size)  # spline coefficients
        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))

        self.scaling_coeffs_vect = nn.Parameter(torch.ones((1, self.num_activations, 1, 1)))

    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size + (self.size // 2))

    @property
    def coefficients(self):
        """ B-spline coefficients. """
        return self.coefficients_vect.view(self.num_activations, self.size)
    
    @property
    def lipschitz_coefficients(self):
        """Projection of B-spline coefficients such that they are 1-Lipschitz"""
        return slope_clipping(self.coefficients, self.grid.item())
    
    @property
    def lipschitz_coefficients_vect(self):
        """Projection of B-spline coefficients such that they are 1-Lipschitz"""
        return self.lipschitz_coefficients.contiguous().view(-1)

    @property
    def relu_slopes(self):
        """ Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        D2_filter = self.D2_filter.to(device=self.coefficients.device)

        slopes = F.conv1d(self.lipschitz_coefficients.unsqueeze(1), D2_filter).squeeze(1)
        return slopes

    def reshape_forward(self, x):
        """
        Reshape inputs for deepspline activation forward pass, depending on
        mode ('conv' or 'fc').
        """
        input_size = x.size()
        if self.mode == 'fc':
            if len(input_size) == 2:
                # one activation per conv channel
                # transform to 4D size (N, num_units=num_activations, 1, 1)
                x = x.view(*input_size, 1, 1)
            else:
                raise ValueError(f'input size is {len(input_size)}D but should be 2D')
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'

        return x

    def reshape_back(self, output, input_size):
        """
        Reshape back outputs after deepspline activation forward pass,
        depending on mode ('conv' or 'fc').
        """
        if self.mode == 'fc':
            # transform back to 2D size (N, num_units)
            output = output.view(*input_size)

        return output


    def forward(self, input):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        input_size = input.size()
        x = self.reshape_forward(input)
        assert x.size(1) == self.num_activations, \
            'Wrong shape of input: {} != {}.'.format(x.size(1), self.num_activations)

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        x = x.mul(self.scaling_coeffs_vect)

        output = LinearSpline_Func.apply(x, self.lipschitz_coefficients_vect, grid, zero_knot_indexes, \
                                        self.size, self.even)

        output = output.div(self.scaling_coeffs_vect)
        output = self.reshape_back(output, input_size)

        return output


    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]:.5f}. ')

        return s.format(**self.__dict__)

    def totalVariation(self, **kwargs):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1)