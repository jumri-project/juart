from typing import List

import torch


class SeparableProximalOperator:
    """
    Class for applying separable proximal operators in PyTorch.

    This class handles a list of separable proximal operators that are applied
    to different parts of a tensor, as defined by their corresponding indices.
    """

    def __init__(
        self,
        proximal_operators: List,
        indices: List,
        verbose: bool = False,
    ):
        """
        Initialize the SeparableProximalOperator.

        Parameters:
        ----------
        proximal_operators : list
            A list of proximal operator objects that will be applied to different parts
            of the tensor.
        indices : list
            A list of indices corresponding to the regions in the tensor where each
            proximal operator will be applied.
        verbose : bool, optional
            Whether to print information during the solving process (default is False).
        """
        self.proximal_operators = proximal_operators
        self.indices = indices
        self.verbose = verbose
        self.r_norms = list()
        self.d_norms = list()

    def solve(
        self,
        v: torch.Tensor,
        lamda: float,
    ) -> torch.Tensor:
        """
        Apply the separable proximal operators to the input tensor `v`.

        This method applies the proximal operators on different parts of `v`, as
        specified by the indices. The proximal operator modifies the tensor in-place.

        Parameters:
        ----------
        v : torch.Tensor
            Input tensor on which the proximal operators are applied.
        lamda : float
            Regularization parameter.

        Returns:
        -------
        v : torch.Tensor
            The updated tensor after applying the proximal operators.
        """
        for index, prox_op in zip(self.indices, self.proximal_operators):
            if self.verbose:
                print(f"[SeparableProximalOperator] Applying {type(prox_op).__name__}")
            v[index] = prox_op.solve(v[index], lamda)

        return v
