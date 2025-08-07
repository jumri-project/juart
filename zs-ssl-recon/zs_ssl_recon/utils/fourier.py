import math
import warnings
from typing import Optional, Tuple, Union

# import finufft
import torch
import tqdm
from pytorch_finufft.functional import finufft_type1, finufft_type2


def fourier_transform_forward(
    x: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Compute the fast Fourier transform of an array using torch.fft on the GPU/CPU.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    axes : tuple of ints
        Axes over which to compute the FFT.

    Returns
    -------
    torch.Tensor
        FFT of the input tensor.
    """
    x = torch.fft.ifftshift(x, dim=axes)
    x = torch.fft.fftn(x, dim=axes, norm="ortho")
    x = torch.fft.fftshift(x, dim=axes)
    return x


def fourier_transform_adjoint(
    x: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Compute the inverse fast Fourier transform of an array using torch.fft on the
    GPU/CPU.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    axes : tuple of ints
        Axes over which to compute the IFFT.

    Returns
    -------
    torch.Tensor
        IFFT of the input tensor.
    """
    x = torch.fft.ifftshift(x, dim=axes)
    x = torch.fft.ifftn(x, dim=axes, norm="ortho")
    x = torch.fft.fftshift(x, dim=axes)
    return x


def nufft2d_type1(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Tuple[int, ...] = None,
    eps: float = 1e-6,
    nthreads: int = 1,
) -> torch.Tensor:
    """
    Compute the NUFFT (type 1) converting non-uniform samples to a uniform grid.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (2, num_samples).
    x : torch.Tensor
        Input tensor of shape (num_samples, ...).
    n_modes : tuple of ints, optional
        Output grid size.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).
    nthreads : int, optional
        Number of threads to use (default: 1).

    Returns
    -------
    torch.Tensor
        Fourier transformed data on a uniform grid.
    """
    y = finufft.nufft2d1(
        k[0, :].cpu().numpy(),
        k[1, :].cpu().numpy(),
        x.cpu().numpy(),
        n_modes=n_modes,
        eps=eps,
        nthreads=nthreads,
    )
    return torch.tensor(y, dtype=x.dtype, device=x.device)


def nufft2d_type2(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Tuple[int, ...] = None,
    eps: float = 1e-6,
    nthreads: int = 1,
) -> torch.Tensor:
    """
    Compute the NUFFT (type 2) converting uniform grid data to non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (2, num_samples).
    x : torch.Tensor
        Fourier domain data on a uniform grid.
    n_modes : tuple of ints, optional
        Size of the input grid.
    eps : float, optional
        Accuracy threshold for the NUFFT (default: 1e-6).
    nthreads : int, optional
        Number of threads to use (default: 1).

    Returns
    -------
    torch.Tensor
        The inverse NUFFT result at non-uniform points.
    """
    y = finufft.nufft2d2(
        k[0, :].cpu().numpy(),
        k[1, :].cpu().numpy(),
        x.cpu().numpy(),
        eps=eps,
        nthreads=nthreads,
    )
    return torch.tensor(y, dtype=x.dtype, device=x.device)


def nonuniform_fourier_transform_forward(
    k: torch.Tensor,
    x: torch.Tensor,
    modeord: int = 0,
    isign: int = 1,
    device: Optional[torch.DeviceObjType] = None,
    **finufftkwargs,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) forward from uniform grid to
    non-uniform points.

    Parameters
    ----------
    k : torch.Tensor
        Non-uniform sampling points of shape (D, N) or (D, N, ...)
        with D dimensions (xyz) and N samples scaled between [-0.5, 0.5].
    x : torch.Tensor, Shape (C, R, P1, P2) or (C, R, P1, P2, ...)
        Input data on a uniform grid with C channels and R readout,
        P1 phase P2 partition samples.
    device : str, optional
        Troch device.
    **finufftkwargs : dict
        Additional arguments for the finufft function.
        See https://finufft.readthedocs.io/en/latest/opts.html#opts.

    Returns
    -------
    torch.Tensor
        Nonuniform Fourier transform of `x` at the sampling points `k`
        with shape (C, N, ...).
    """
    # Check if the input data is on right device
    if device is not None:
        k = k.to(device)
        x = x.to(device)
    else:
        device = k.device
        if x.device != device:
            warnings.warn(
                f"k and x are not on the same device. Copy x to {device}",
                stacklevel=1,
            )
            x = x.to(device)

    # Ensure that the input kspace locations have correct scaling
    # if torch.any(torch.abs(k) > 0.5):
    #     raise ValueError(
    #         "Non-uniform sampling points k must be scaled between -0.5 and 0.5."
    #     )

    # Ensure x has at least shape (C, R, P1, P2)
    if x.ndim < 4:
        raise ValueError(
            "Input data x must have at least three dimensions "
            "(channel, read, phase1, phase2)!"
        )

    # Ensure k has at least shape (D, N)
    if k.ndim < 2:
        raise ValueError(
            "Non-uniform sampling points k must have at least two dimensions "
            "(dim, cols)."
        )

    # Scales data
    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    # Move channel axis to the last axis
    x = torch.moveaxis(x, 0, -1)

    # k and x are shaped as (D, N, [shared_axes])
    # and (C, R, P1, P2, [shared_axes], [excl_axes]).
    # The shared axes are the same in k and x.
    # The exclusive axes are the additional axes in x.

    # Get dimensions
    D, N, *add_axes_k = k.shape
    R, P1, P2, *add_axes_x = x.shape

    norm = torch.sqrt(
        torch.prod(torch.tensor([R, P1, P2], dtype=torch.float32, device=device))
    )

    if len(add_axes_k) > len(add_axes_x):
        raise ValueError(
            "The number additional dims (D, N, ...) in k must be"
            " less than or equal to the number of additional dims (C, N, ...) in x."
        )

    shared_axes = add_axes_x[: len(add_axes_k)]
    excl_axes = add_axes_x[len(add_axes_k) :]

    if add_axes_k != shared_axes:
        raise ValueError(
            "The additional dims in k must be found in addtional dims in x"
            f"but are {add_axes_k} and {shared_axes}."
        )

    num_shared_axes = math.prod(shared_axes)
    num_excl_axes = math.prod(excl_axes)

    # Flatten the additional axes
    x = x.reshape(R, P1, P2, num_shared_axes, num_excl_axes)
    k = k.reshape(D, N, num_shared_axes)

    # Change to (shared, excl, N)
    x = x.permute(3, 4, 0, 1, 2)
    k = k.permute(2, 0, 1)

    # Flatten R, P1, P2 dim to match 1D/2D/3D finufft
    x = x.squeeze(dim=(-1, -2, -3))

    k = k.cpu().contiguous()
    x = x.cpu().contiguous()

    y = []

    for index in tqdm.tqdm(range(x.shape[0]), disable=True):
        res = finufft_type2(
            k[index, ...],
            x[index, ...],
            modeord=modeord,
            isign=isign,
            **finufftkwargs,
        )

        y.append(res)

    y = torch.stack(y)

    y = y.to(device)

    y /= norm

    # Reshape flattened additional axes
    y = y.reshape(*shared_axes, *excl_axes, N)

    # move channel and column axes to the front
    # to get (C, N, [shared_axes], [excl_axes])
    y = y.moveaxis((-2, -1), (0, 1))

    # Remove the added axis if there are no additional
    # axis in x except channel
    if add_axes_x[:-1] == [1]:
        y = y.squeeze(-1)

    return y


def nonuniform_fourier_transform_adjoint(
    k: torch.Tensor,
    x: torch.Tensor,
    n_modes: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    modeord: int = 0,
    isign: int = 1,
    device: Optional[torch.DeviceObjType] = None,
    **finufftkwargs,
) -> torch.Tensor:
    """
    Compute the non-uniform Fourier transform (NUFFT) adjoint
    from non-uniform points to a uniform grid.

    Parameters
    ----------
    k : torch.Tensor, Shape (D, N) or (D, N, ...)
        Non-uniform sampling points with D dimensions (xyz) and N samples.
    x : torch.Tensor, Shape (N, ), (C, N) or (C, N, ...)
        Signal data of the non-uniform sampling points `k`.
        with C channels and N samples.
    n_modes : tuple of ints
        Shape of the grid to reconstruct to (e.g., (nx,), (nx, ny), (nx, ny, nz)).
    device : torch.device, optional
        Device to run the computation.
        If None, the device of `k` will be used.
    **finufftkwargs : dict
        Additional arguments for the finufft function.
        See https://finufft.readthedocs.io/en/latest/opts.html#opts.
    Returns
    -------
    torch.Tensor, Shape (C, nx, ny, nz) or (C, nx, ny, nz, ...)
        Singal data on a uniform grid.
        If nx, ny or nz is not in n_modes, the corresponding output dimension will be 1.
    """
    if device is not None:
        k = k.to(device)
        x = x.to(device)
    else:
        device = k.device
        if x.device != device:
            warnings.warn(
                f"k and x are not on the same device. Copy x to {device}",
                stacklevel=1,
            )
            x = x.to(device)

    # if torch.max(torch.abs(k)) > 0.5:
    #     raise ValueError(
    #         "Non-uniform sampling points k must be scaled between -0.5 and 0.5."
    #     )

    if k.shape[0] != len(n_modes):
        raise ValueError(
            "The number of dimensions in n_modes "
            "must be equal to the number of dimensions in k "
            f"but are {k.shape[0]} and {len(n_modes)}."
        )

    norm = torch.sqrt(
        torch.prod(torch.tensor(n_modes, dtype=torch.float32, device=device))
    )

    k = 2 * torch.pi * k.to(torch.float32)
    x = x.to(torch.complex64)

    # pytorch finufft works best in batches.
    # There are batches where k and x have the same batch size ([shared_axes]).
    # x can have more batches than k where k does not change e.g. C ([excl_axes])
    # Reshape x to (prod[shared_axes], prod[excl_axes], N)
    # and k to (prod[shared_axes], D, N)
    # The torch output y will have
    # shape (prod[shared_axes], prod[excl_axes], [n_modes])
    # and is reshaped to (C, [n_modes], [shared_axes], [excl_axes/C])

    # Ensure x to have at least shape (C, N, 1)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add axis for channel dimension
    if x.ndim == 2:
        x = x.unsqueeze(2)  # Add axis for additional dimensions

    # Make channel axis the last axis and handle as additional axis too
    x = torch.moveaxis(x, 0, -1)

    # Ensure that the input data has the correct shape
    N_x, *add_axes_x = x.shape
    D, N_k, *add_axes_k = k.shape

    if N_x != N_k:
        raise ValueError(
            "The number of columns in x and k must be the same "
            f"but are {N_x} and {N_k}."
        )
    N = N_x

    if len(add_axes_k) > len(add_axes_x):
        raise ValueError(
            "The number additional dims (D, N, ...) in k must be"
            " less than or equal to the number of additional dims (C, N, ...) in x."
        )

    shared_axes = add_axes_x[: len(add_axes_k)]
    excl_axes = add_axes_x[len(add_axes_k) :]

    if add_axes_k != shared_axes:
        raise ValueError(
            "The additional dims in k must be found in addtional dims in x"
            f"but are {add_axes_k} and {shared_axes}."
        )

    num_shared_axes = math.prod(shared_axes)
    num_excl_axes = math.prod(excl_axes)

    # Flatten the additional axes
    x = x.reshape(N, num_shared_axes, num_excl_axes)
    k = k.reshape(D, N, num_shared_axes)

    # Adjust to (shared, excl, N)
    x = x.permute(1, 2, 0)
    k = k.permute(2, 0, 1)

    k = k.cpu().contiguous()
    x = x.cpu().contiguous()

    y = []

    for index in tqdm.tqdm(range(x.shape[0]), disable=True):
        res = finufft_type1(
            k[index, ...],
            x[index, ...],
            output_shape=n_modes,
            modeord=modeord,
            isign=isign,
            **finufftkwargs,
        )

        y.append(res)

    y = torch.stack(y)

    y = y.to(device)

    y /= norm

    # Ensure y has nx ny nz dim
    if len(n_modes) < 3:
        mode_diff = 3 - len(n_modes)
        for _ in range(mode_diff):
            y = y.unsqueeze(-1)

    # Reshape flattened additional axes
    y = y.reshape(*shared_axes, *excl_axes, *y.shape[-3:])

    # Permute back to original order, channel first
    y = y.moveaxis((len(add_axes_x) - 1, -3, -2, -1), (0, 1, 2, 3))

    # Remove the added axis if there are no additional
    # axis in x except channel
    if add_axes_x[:-1] == [1]:
        y = y.squeeze(-1)

    return y


def nonuniform_transfer_function(
    k: torch.Tensor,
    data_shape: Tuple[int, ...],
    oversampling: Tuple[int, ...] = 2,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the non-uniform transfer function using PyTorch.

    Parameters
    ----------
    k : torch.Tensor
        The k-space trajectory of shape (D, N, ...)
        with D dimensions (kx,ky,kz) and N columns.
    data_shape : tuple of ints
        The shape of the complex data tensor of format (1, R, P1, P2, ...).
    oversampling : int or tuple of ints, optional
        The oversampling factors along each axis
        (eg. (OS_R,), (OS_R, OS_P1), (OS_R, OS_P1, OS_P2)).
        If an integer is provided,
        it is used for all axes (R, P1, P2)>1 in `data_shape`.
        Default is 2.
    eps : float, optional
        The error tolerance for the NUFFT adjoint operation.

    Returns
    -------
    transfer_function : torch.Tensor
        The transfer function with
         oversampled data_shape (1, OS_R*R, OS_P1*P1, OS_P2*P2, ...).
    """

    _, nX, nY, nZ, *add_axes_x = data_shape
    nDim, nCol, *add_axes_k = k.shape

    excl_axes_x = add_axes_x[len(add_axes_k) :]
    n_modes = tuple([n for n in [nX, nY, nZ] if n > 1])

    if len(n_modes) != nDim:
        raise ValueError(
            "The number of spacial encoding dimensions (R, P1, P2)"
            " in data_shape must match the number of dimensions in k."
        )

    # Apply oversampling
    if isinstance(oversampling, int):
        oversampling = (oversampling,) * len(n_modes)
    elif len(oversampling) != len(n_modes):
        raise ValueError(
            "Oversampling must be an integer or a tuple with length equal to the "
            "number of spacial encoding dimensions (R, P1, P2) in data_shape."
        )
    n_modes = tuple([n * o for n, o in zip(n_modes, oversampling)])

    # Create normalized input array (PyTorch tensor)
    norm = 1 / torch.prod(torch.tensor(oversampling, dtype=torch.float32))
    x = torch.ones((1, nCol, *add_axes_k), dtype=torch.complex64, device=k.device)
    x = x / norm

    if weights is not None:
        x = x * weights

    # Compute the Point-Spread Function (PSF) using the non-uniform adjoint Fourier
    # transform for the dimensions where k changes
    PSF = nonuniform_fourier_transform_adjoint(k, x, n_modes)

    # Add the additioal dimensions of the output
    new_axes = PSF.shape + (1,) * len(excl_axes_x)
    new_shape = PSF.shape + tuple(excl_axes_x)
    PSF = PSF.reshape(new_axes).expand(new_shape)

    # Compute the transfer function using the forward Fourier transform
    fft_axes = tuple(range(1, len(n_modes) + 1))
    transfer_function = fourier_transform_forward(PSF, fft_axes)

    return transfer_function


def crop_tensor(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Crop a tensor to the specified target shape.

    The function crops the input tensor symmetrically around the center, based on
    the provided target shape.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be cropped.
    target_shape : Tuple[int, ...]
        The target shape for the cropped tensor. Must be smaller than or equal to
        the input tensor shape in all dimensions.

    Returns
    -------
    torch.Tensor
        The cropped tensor.

    Raises
    ------
    AssertionError
        If the input and target shapes do not have the same number of dimensions or
        if the target shape is larger than the input tensor shape in any dimension.
    """
    input_shape = tensor.shape

    # Ensure the input and target shapes have the same number of dimensions
    assert len(input_shape) == len(target_shape), (
        "Input and target shapes must have the same number of dimensions."
    )

    # Ensure that the target shape is less than or equal to the input shape in all
    # dimensions
    assert all(
        target_dim <= input_dim
        for target_dim, input_dim in zip(target_shape, input_shape)
    ), "Target shape must be less than or equal to input shape in all dimensions."

    # Calculate the slice indices for cropping
    indices = [
        slice(
            int(math.floor(input_dim / 2.0) + math.ceil(-target_dim / 2.0)),
            int(math.floor(input_dim / 2.0) + math.ceil(target_dim / 2.0)),
        )
        for input_dim, target_dim in zip(input_shape, target_shape)
    ]

    return tensor[tuple(indices)]


def pad_tensor(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Pad a tensor to the specified target shape.

    The function pads the input tensor symmetrically with zeros around the center,
    based on the provided target shape.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be padded.
    target_shape : Tuple[int, ...]
        The target shape for the padded tensor. Must be greater than or equal to
        the input tensor shape in all dimensions.

    Returns
    -------
    torch.Tensor
        The padded tensor.

    Raises
    ------
    AssertionError
        If the input and target shapes do not have the same number of dimensions or
        if the target shape is smaller than the input tensor shape in any dimension.
    """
    input_shape = tensor.shape

    # Ensure the input and target shapes have the same number of dimensions
    assert len(input_shape) == len(target_shape), (
        "Input and target shapes must have the same number of dimensions."
    )

    # Ensure that the target shape is greater than or equal to the input shape in all
    # dimensions
    assert all(
        target_dim >= input_dim
        for target_dim, input_dim in zip(target_shape, input_shape)
    ), "Target shape must be greater than or equal to input shape in all dimensions."

    # Create a new tensor with the target shape, initialized to zero
    padded_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)

    # Calculate the slice indices for padding
    indices = [
        slice(
            int(math.floor(target_dim / 2.0) + math.ceil(-input_dim / 2.0)),
            int(math.floor(target_dim / 2.0) + math.ceil(input_dim / 2.0)),
        )
        for target_dim, input_dim in zip(target_shape, input_shape)
    ]

    padded_tensor[tuple(indices)] = tensor
    return padded_tensor


def apply_transfer_function(
    x: torch.Tensor,
    transfer_function: torch.Tensor,
    axes: Tuple[int, ...],
) -> torch.Tensor:
    """
    Apply an oversampled transfer function in the Fourier domain using PyTorch.

    This function first pads the input tensor `x` to match the size of the transfer
    function `transfer_function` along the specified `axes`. It then applies the
    transfer function in the Fourier space, performs the inverse Fourier transform,
    and crops the result back to the original shape.

    Parameters:
    ----------
    x : torch.Tensor
        The input tensor to which the oversampled transfer function is applied.
    transfer_function : torch.Tensor
        The transfer function to apply in the Fourier domain.
    axes : tuple of ints
        The axes along which the Fourier transform is applied.

    Returns:
    -------
    torch.Tensor
        The tensor after applying the oversampled transfer function in the Fourier
        domain.
    """
    # Get the shape and pad it to match the transfer function's size along the specified
    # axes
    shape = x.shape
    pad_shape = list(shape)

    for ax in axes:
        pad_shape[ax] = transfer_function.shape[ax]  # Adjust the padding size

    # Pad the input tensor to match the size of transfer_function
    if pad_shape != shape:
        x = pad_tensor(x, pad_shape)

    # Perform forward Fourier transform
    x = fourier_transform_forward(x, axes)

    # Apply the transfer function
    x = transfer_function * x

    # Perform inverse Fourier transform
    x = fourier_transform_adjoint(x, axes)

    # Crop back to the original shape
    if pad_shape != shape:
        x = crop_tensor(x, shape)

    return x.clone()  # Clone to ensure memory continuity
