"""Implement custom Layer Norm to reduce memory footprint"""
import time
import typing

from copy import deepcopy

import torch

import torch.nn as nn
from torch import Tensor, optim

"""
# TODO:
1) Do more checks to avoid any silent errors
2) Compute backprop as in DeepSpeed in attempt to reduce memory consumption and enhance speed
"""
@torch.jit.script
def backward_helper(
    dout: Tensor,
    var: Tensor,
    weight: Tensor,
    bias: Tensor,
    pred: Tensor,
    eps: Tensor
) -> typing.Tuple[Tensor, Tensor, Tensor]:
    """Backpropagation jit script"""
    features = dout.shape[-1]

    x_hat = (pred - bias)/weight
    xmu = x_hat*torch.sqrt(var + eps)
    dx_hat = dout * weight

    # dalpha and dbeta
    dbeta = dout.sum(dim=(0, 1))
    dalpha = (dout * x_hat).sum(dim=(0, 1))

    # As in https://www.deepspeed.ai/news/2020/05/27/fastest-bert-training.html
    # and https://usmanr149.github.io/urmlblog/cs231n%20assignments/2020/04/03/Batchnorm.html
    inv_std = 1/torch.sqrt(var + eps)
    dx = dx_hat*inv_std - \
        (inv_std**3/features) * \
        (var*dx_hat.sum(dim=-1, keepdim=True) + xmu*((dx_hat*xmu).sum(dim=-1, keepdim=True)))

    return dx, dalpha, dbeta

@torch.jit.script
def forward_helper(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: Tensor
) -> typing.Tuple[Tensor, Tensor]:
    """Forward propagation jit script"""
    mean = x.mean(dim=-1, keepdim=True)
    xmu = (x - mean)
    var  = (xmu ** 2).mean(dim=-1, keepdim=True)
    x_hat = xmu / torch.sqrt(var + eps)

    # TODO: add elementwise_affine
    pred = weight*x_hat + bias

    return var, pred

class LayerNormFunction(torch.autograd.Function):
    """Custom forward pass and backward pass"""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        var, pred = forward_helper(x, weight, bias, eps)

        # DeepSpeed stores var, weight, bias, and pred
        ctx.save_for_backward(var, weight, bias, pred, eps)

        return pred

    @staticmethod
    def backward(ctx, dout):
        var, weight, bias, pred, eps = ctx.saved_tensors
        dx, dalpha, dbeta = backward_helper(dout, var, weight, bias, pred, eps)
        return dx, dalpha, dbeta, None, None

class CustomLayerNorm(nn.LayerNorm):
    """Custom Layer Norm to reduce memory footprint"""

    def __init__(
        self,
        normalized_shape,
        device,
        eps=1e-05,
    ):
        super(CustomLayerNorm, self).__init__(
            normalized_shape,
            elementwise_affine=True,
            device=device,
            eps=eps
        )

        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, Tensor([self.eps]).to(self.device))

class CustomNN(nn.Module):
    """Simple linear layer with layer norm for debugging purposes"""
    # TODO: add elementwise_affine and set its default value to False
    def __init__(self, embed_size, custom_layer_norm, device, eps=1e-5):
        super(CustomNN, self).__init__()
        self.linear_layer = nn.Linear(embed_size, embed_size, device=device)
        if not custom_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_size, device=device, eps=eps)
        else:
            self.layer_norm = CustomLayerNorm(embed_size, device=device, eps=eps)

    def forward(self, x):
        """ Feedforward function """
        return self.layer_norm(self.linear_layer(x))

def custom_train(norm, input_x, target_y, criterion, optimizer, epochs=1000):
    """Custom train with backprop"""
    print('\nCustom train:')
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = norm(input_x)
        loss = criterion(output, target_y)
        if epoch % 100 == 0:
            print(f'Loss at epoch {epoch:05d}:', loss.item())
        loss.backward()
        optimizer.step()

def main():
    """Main function"""
    train = True
    torch.manual_seed(0)
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    batch, sequence, features = 16, 64, 128
    input_x  = torch.randn(batch, sequence, features, requires_grad=True).to(device)
    target_y = torch.randn(batch, sequence, features, requires_grad=True).to(device)

    # Activating the module 1
    original_ln = CustomNN(features, custom_layer_norm=False, device=device)
    output1 = original_ln(input_x)
    loss1 = (output1 - target_y).pow(2).sum()
    loss1.backward()

    print(output1[0:5, 0, 0])
    print(loss1)

    # Activating the module 2
    custom_ln = CustomNN(features, custom_layer_norm=True, device=device)
    custom_ln.linear_layer = deepcopy(original_ln.linear_layer)
    output2 = custom_ln(input_x)
    loss2 = (output2 - target_y).pow(2).sum()
    loss2.backward()

    print(output2[0:5, 0, 0])
    print(loss2)

    if train:
        # Original
        start1 = time.time()
        learning_rate = 1e-2
        optimizer1 = optim.SGD(original_ln.parameters(), lr=learning_rate)
        custom_train(original_ln, input_x, target_y, nn.MSELoss(), optimizer1)
        end1 = time.time()

        # Custom
        start2 = time.time()
        learning_rate = 1e-2
        optimizer2 = optim.SGD(custom_ln.parameters(), lr=learning_rate)
        custom_train(custom_ln, input_x, target_y, nn.MSELoss(), optimizer2)
        end2 = time.time()

        print(f'Elapsed time: {end1 - start1:.2f} seconds')
        print(f'Elapsed time: {end2 - start2:.2f} seconds')


if __name__ == '__main__':
    main()
