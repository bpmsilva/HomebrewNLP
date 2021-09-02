"""Implement custom Layer Norm to reduce memory footprint"""
import time

import torch

import torch.nn as nn
from torch import Tensor, optim

class LayerNormFunction(torch.autograd.Function):
    """Custom forward pass and backward pass"""
    @staticmethod
    def forward(ctx, x, weight, bias, elementwise_affine, eps):

        mean = x.mean(axis=-1, keepdim=True)
        xmu = (x - mean)
        var  = (xmu ** 2).mean(axis=-1, keepdim=True)
        x_hat = xmu / torch.sqrt(var + eps)

        if elementwise_affine:
            pred = x_hat*weight + bias
        else:
            pred = x_hat

        ctx.save_for_backward(xmu, var, weight, x_hat, Tensor([eps]))

        return pred

    @staticmethod
    def backward(ctx, dout):
        batch, _, _ = dout.shape

        xmu, var, alpha, x_hat, eps = ctx.saved_tensors
        eps = eps.item()

        # dalpha and dbeta
        dbeta = dout.sum(axis=(0, 1))
        dalpha = (x_hat*dout).sum(axis=(0, 1))

        dx_hat = (dout * alpha)
        inv_var = 1/torch.sqrt(var + eps)
        dx_hat_sum = dx_hat.sum(axis=-1, keepdim=True)

        # As in https://usmanr149.github.io/urmlblog/cs231n%20assignments/2020/04/03/Batchnorm.html
        dx = dx_hat*inv_var + (inv_var**3/batch)*(-var**2*dx_hat_sum - xmu*dx_hat_sum*xmu)

        return dx, dalpha, dbeta, None, None

class CustomLayerNorm(nn.LayerNorm):
    """Custom Layer Norm to reduce memory footprint"""

    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        # TODO: solve argument issue
        # device=None,
        # dtype=None
    ):
        super(CustomLayerNorm, self).__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            # TODO: solve argument issue
            # device=device,
            # dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.elementwise_affine, self.eps)

def custom_train(norm, input_x, target_y, criterion, optimizer, epochs=10000):
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

    batch, sequence, features = 16, 64, 128
    input_x  = torch.randn(batch, sequence, features, requires_grad=True)
    target_y = torch.randn(batch, sequence, features, requires_grad=True)

    # # Activating the module 1
    original_ln = nn.LayerNorm(features)
    output1 = original_ln(input_x)
    loss1 = (output1 - target_y).pow(2).sum()
    loss1.backward()

    print(output1[0, 0, 0])
    print(loss1)

    # # Activating the module 2
    custom_ln = CustomLayerNorm(features)
    output2 = custom_ln(input_x)
    loss2 = (output2 - target_y).pow(2).sum()
    loss2.backward()

    print(output2[0, 0, 0])
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
