import pytest
import torch

from juart.utils import resize


def test_resize_centric_pad():
    x = torch.ones(3, 3)
    y = resize(x, size=(5, 5), dims=(0, 1), mode="centric")
    assert y.shape == (5, 5)
    assert torch.all(y[1:4, 1:4] == 1)
    assert torch.all(y[0] == 0)
    assert torch.all(y[:, 0] == 0)
    assert torch.all(y[-1] == 0)
    assert torch.all(y[:, -1] == 0)


def test_resize_left_pad():
    x = torch.ones(2, 2)
    y = resize(x, size=(4, 4), dims=(0, 1), mode="left")
    assert y.shape == (4, 4)
    assert torch.all(y[:2, :2] == 0)
    assert torch.all(y[2:, 2:] == 1)
    assert torch.all(y[2:, 2:] == 1)


def test_resize_right_pad():
    x = torch.ones(2, 2)
    y = resize(x, size=(4, 4), dims=(0, 1), mode="rigth")
    assert y.shape == (4, 4)
    assert torch.all(y[-2:, -2:] == 0)
    assert torch.all(y[:2, :2] == 1)
    assert torch.all(y[:2, :2] == 1)


def test_resize_centric_crop():
    x = torch.arange(25).reshape(5, 5)
    y = resize(x, size=(3, 3), dims=(0, 1), mode="centric")
    assert y.shape == (3, 3)
    assert torch.equal(y, x[1:4, 1:4])


def test_resize_left_crop():
    x = torch.ones(4, 4)
    y = resize(x, size=(2, 2), dims=(0, 1), mode="left")
    assert y.shape == (2, 2)
    assert torch.equal(y, x[:2, :2])


def test_resize_right_crop():
    x = torch.ones(4, 4)
    y = resize(x, size=(2, 2), dims=(0, 1), mode="rigth")
    assert y.shape == (2, 2)
    assert torch.equal(y, x[-2:, -2:])


def test_resize_invalid_mode():
    x = torch.ones(2, 2)
    with pytest.raises(ValueError) as e:
        resize(x, size=(3,), dims=(0,), mode="invalid")
    assert "Unknown mode" in str(e.value)


def test_resize_partial_dims_pad():
    x = torch.ones(3, 3)
    y = resize(x, size=5, dims=0, mode="centric")
    assert y.shape == (5, 3)
    assert torch.all(y[1:4, :] == 1)
    assert torch.all(y[0, :] == 0)
    assert torch.all(y[-1, :] == 0)


def test_resize_partial_dims_crop():
    x = torch.arange(25).reshape(5, 5)
    y = resize(x, size=3, dims=1, mode="left")
    expected = x[:, 2:]
    assert y.shape == (5, 3)
    assert torch.equal(y, expected)


if __name__ == "__main__":
    pytest.main([__file__])
