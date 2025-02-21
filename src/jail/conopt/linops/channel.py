from typing import Tuple

import torch

from . import LinearOperator


class ChannelOperator(LinearOperator):
    """
    A linear operator that applies forward and adjoint operations using coil sensitivity
    maps.
    """

    def __init__(
        self,
        coil_sensitivities: torch.Tensor,
        data_shape: Tuple[int, ...],
        normalize: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize the ChannelOperator.

        Parameters:
        ----------
        coil_sensitivities : torch.Tensor
            Coil sensitivity maps (nC, nX, nY, nZ).
        data_shape : tuple of int
            Shape of the data (nX, nY, nZ, nS, nTI, nTE).
        normalize : bool, optional
            Whether to normalize to unit spectral norm (default is True).
        device : str, optional
            Device on which the operator will run ('cpu' or 'cuda').
        """
        # Unpack the data shape
        nX, nY, nZ, nS, nTI, nTE = data_shape

        # Number of channels (coils)
        nC = coil_sensitivities.shape[0]

        # Device and dtype
        self.device = device
        self.dtype = torch.float32
        self.internal_dtype = torch.complex64

        # Define forward and adjoint shapes
        self.forward_shape = (1, nX, nY, nZ, nS, nTI, nTE)
        self.adjoint_shape = (nC, nX, nY, nZ, nS, nTI, nTE)

        # Define the shape for complex-to-real operations
        self.shape = (
            2 * torch.prod(torch.tensor(self.adjoint_shape)),
            2 * torch.prod(torch.tensor(self.forward_shape)),
        )

        # Store coil sensitivities, ensuring correct dimensions and device
        self.coil_sensitivities = coil_sensitivities[..., None, None].to(device)

        # Normalize to unit spectral norm if required
        if normalize:
            self.coil_sensitivities = (
                self.coil_sensitivities
                / torch.sqrt(
                    torch.sum(torch.abs(self.coil_sensitivities) ** 2, dim=0)
                ).max()
            )

    def _matvec(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the forward channel operator (coil sensitivities).

        This operation multiplies the input tensor by the coil sensitivity maps.

        Parameters:
        ----------
        input_tensor : torch.Tensor
            Input data tensor to be transformed.

        Returns:
        -------
        torch.Tensor
            The result after applying the forward channel operator.
        """
        # Reshape the input data to the appropriate internal dtype and shape
        input_tensor = input_tensor.view(self.internal_dtype).reshape(
            self.forward_shape
        )

        # Apply the coil sensitivities
        output_tensor = self.coil_sensitivities * input_tensor

        # Flatten the result and cast back to the external dtype
        output_tensor = output_tensor.view(self.dtype).ravel()

        return output_tensor

    def _rmatvec(
        self,
        coil_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the adjoint channel operator (combining coil signals).

        This operation sums the input data weighted by the conjugate of the coil
        sensitivity maps.

        Parameters:
        ----------
        coil_data : torch.Tensor
            Input data tensor representing the coil signals.

        Returns:
        -------
        torch.Tensor
            The result after applying the adjoint channel operator.
        """
        # Reshape the input data to the appropriate internal dtype and shape
        coil_data = coil_data.view(self.internal_dtype).reshape(self.adjoint_shape)

        # Apply the adjoint (combining the signals using the conjugate of the
        # sensitivities)
        output_tensor = torch.sum(
            torch.conj(self.coil_sensitivities) * coil_data, dim=0
        )

        # Flatten the result and cast back to the external dtype
        output_tensor = output_tensor.view(self.dtype).ravel()

        return output_tensor
