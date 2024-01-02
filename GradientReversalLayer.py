from typing import Tuple

import torch
from torch.autograd import Function


class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, any]:
        output = grad_output.neg() * ctx.alpha
        return output, None
