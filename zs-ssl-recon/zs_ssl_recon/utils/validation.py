import functools
import time

import torch.distributed as dist


def timing_layer(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.timing_level <= 0:
            return func(self, *args, **kwargs)
        else:
            base_string = (
                f"Rank {dist.get_rank()} - {self.__class__.__name__}.{func.__name__}"
            )
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            print(f"{base_string} took {end_time - start_time:.4f} seconds to run.")
            return result

    return wrapper


def validation_layer(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.validation_level <= 0:
            return func(self, *args, **kwargs)
        else:
            base_string = (
                f"Rank {dist.get_rank()} - {self.__class__.__name__}.{func.__name__}"
            )
            verbosity_layer(base_string, *args, **kwargs)
            assertion_layer(base_string, *args, **kwargs)
            return func(self, *args, **kwargs)

    return wrapper


def verbosity_layer(
    base_string,
    images=None,
    images_reconstructed=None,
    images_regridded=None,
    kspace_trajectory=None,
    kspace_data=None,
    kspace_mask=None,
    kspace_mask_source=None,
    kspace_mask_target=None,
    sensitivity_maps=None,
):
    if images is not None:
        print(f"{base_string} - images: {images.shape}, {images.dtype}")
    if images_reconstructed is not None:
        print(
            f"{base_string} - images_reconstructed: {images_reconstructed.shape}, {images_reconstructed.dtype}"
        )
    if images_regridded is not None:
        print(
            f"{base_string} - images_regridded: {images_regridded.shape}, {images_regridded.dtype}"
        )
    if kspace_trajectory is not None:
        print(
            f"{base_string} - kspace_trajectory: {kspace_trajectory.shape}, {kspace_trajectory.dtype}"
        )
    if kspace_data is not None:
        print(f"{base_string} - kspace_data: {kspace_data.shape}, {kspace_data.dtype}")
    if kspace_mask is not None:
        print(f"{base_string} - kspace_mask: {kspace_mask.shape}, {kspace_mask.dtype}")
    if kspace_mask_source is not None:
        print(
            f"{base_string} - kspace_mask_source: {kspace_mask_source.shape}, {kspace_mask_source.dtype}"
        )
    if kspace_mask_target is not None:
        print(
            f"{base_string} - kspace_mask_source: {kspace_mask_target.shape}, {kspace_mask_target.dtype}"
        )
    if sensitivity_maps is not None:
        print(
            f"{base_string} - sensitivity_maps: {sensitivity_maps.shape}, {sensitivity_maps.dtype}"
        )


def assertion_layer(
    base_string,
    images=None,
    images_reconstructed=None,
    images_regridded=None,
    kspace_trajectory=None,
    kspace_data=None,
    kspace_mask=None,
    kspace_mask_source=None,
    kspace_mask_target=None,
    sensitivity_maps=None,
):
    error_messages = list()

    if images is not None and images.dim() != 5:
        error_messages.append("images should have dimension 5")
    if images is not None and images.shape[0] != 1:
        error_messages.append("images should have batch dimension 1")

    if images_reconstructed is not None and images_reconstructed.dim() != 5:
        error_messages.append("images_reconstructed should have dimension 5")
    if images_reconstructed is not None and images_reconstructed.shape[0] != 1:
        error_messages.append("images_reconstructed should have batch dimension 1")

    if images_regridded is not None and images_regridded.dim() != 5:
        error_messages.append("images_regridded should have dimension 5")
    if images_regridded is not None and images_regridded.shape[0] != 1:
        error_messages.append("images_regridded should have batch dimension 1")

    if kspace_trajectory is not None and kspace_trajectory.dim() != 5:
        error_messages.append("kspace_trajectory should have dimension 5")
    if kspace_trajectory is not None and kspace_trajectory.shape[0] != 1:
        error_messages.append("kspace_trajectory should have batch dimension 1")

    if kspace_data is not None and kspace_data.dim() != 5:
        error_messages.append("kspace_data should have dimension 5")
    if kspace_data is not None and kspace_data.shape[0] != 1:
        error_messages.append("kspace_data should have batch dimension 1")

    if kspace_mask is not None and kspace_mask.dim() != 5:
        error_messages.append("kspace_mask should have dimension 5")
    if kspace_mask is not None and kspace_mask.shape[0] != 1:
        error_messages.append("kspace_mask should have batch dimension 1")

    if kspace_mask_source is not None and kspace_mask_source.dim() != 5:
        error_messages.append("kspace_mask_source should have dimension 5")
    if kspace_mask_source is not None and kspace_mask_source.shape[0] != 1:
        error_messages.append("kspace_mask_source should have batch dimension 1")

    if kspace_mask_target is not None and kspace_mask_target.dim() != 5:
        error_messages.append("kspace_mask_target should have dimension 5")
    if kspace_mask_target is not None and kspace_mask_target.shape[0] != 1:
        error_messages.append("kspace_mask_target should have batch dimension 1")

    if sensitivity_maps is not None and sensitivity_maps.dim() != 4:
        error_messages.append("sensitivity_maps should have dimension 4")
    if sensitivity_maps is not None and sensitivity_maps.shape[0] != 1:
        error_messages.append("sensitivity_maps should have batch dimension 1")

    if error_messages:
        error_report = (
            f"{base_string} - Input validation failed with the following errors:\n"
            + "\n".join(error_messages)
        )
        raise AssertionError(error_report)
