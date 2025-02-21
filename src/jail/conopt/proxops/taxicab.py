from typing import Tuple

import torch


def joint_soft_thresholding(
    x: torch.Tensor,
    lamda: float,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Joint soft thresholding with mixed l1/l2 norms.

    This function applies soft thresholding on the input tensor `x` along the specified
    `axes`. The l2 norm is computed across the given axes, and the thresholding is
    applied based on the threshold value `lamda`. This is often used in group sparse
    regularization methods.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor to be thresholded.
    lamda : float
        Threshold parameter.
    axes : tuple of ints
        Axes along which to compute the l2 norm.

    Returns:
    -------
    torch.Tensor
        The thresholded tensor.
    """
    # Compute the l2 norm along the specified axes
    s = torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=axes, keepdim=True))

    # Avoid division by zero by clamping the norm
    s = torch.clamp(s, min=lamda)

    # Apply soft thresholding
    x = x * (1 - lamda / s)

    return x


class JointSoftThresholdingOperator:
    """
    Joint Soft Thresholding Operator using PyTorch.

    This operator applies joint soft thresholding with mixed l1/l2 norms on a tensor.
    It is typically used for structured sparsity regularization, such as group lasso.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        lamda: float,
        axes: Tuple[int, ...],
    ):
        """
        Initialize the JointSoftThresholdingOperator.

        Parameters:
        ----------
        shape : tuple
            Shape of the input tensor.
        lamda : float
            Threshold parameter for soft thresholding.
        axes : tuple
            Axes along which to compute the l2 norm.
        """
        self.shape = shape
        self.lamda = lamda
        self.axes = axes

        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

    def solve(
        self,
        x: torch.Tensor,
        rho: float,
        *args,
    ) -> torch.Tensor:
        """
        Solve the joint soft thresholding problem.

        This function applies the joint soft thresholding operation on the input tensor
        `x` using the provided regularization parameter `rho`. The thresholding is
        performed along the axes specified during initialization.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor to which soft thresholding is applied.
        rho : float
            Regularization parameter.
        axes : tuple of ints
            Axes along which the l2 norm is computed.

        Returns:
        -------
        torch.Tensor
            The thresholded tensor.
        """
        x = x.view(self.internal_dtype).reshape(self.shape)
        x = joint_soft_thresholding(x, self.lamda / rho, axes=self.axes)
        x = x.ravel().view(self.dtype)

        return x
