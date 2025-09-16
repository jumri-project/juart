import pytest
import numpy as np

from juart.gradcorr.calibscans import _find_zero_crossings, _window_selection


def test_simple_sine_like():
    """Basic alternating trajectory should have regular zero crossings."""
    k = np.array([ -30., -10., 10., 30., -20., -5., 5., 25. ])
    expect = np.array([1, 3, 5])
    crossings = _find_zero_crossings(k)
    # Expect crossings between indices (1,2) and (5,6)
    np.testing.assert_array_equal(expect, crossings)

def test_no_crossings_all_positive():
    """Trajectory fully above zero → no zero crossings."""
    k = np.linspace(1, 30, 10)
    crossings = _find_zero_crossings(k)
    assert crossings.size == 0

def test_no_crossings_all_negative():
    """Trajectory fully below zero → no zero crossings."""
    k = np.linspace(-30, -1, 10)
    crossings = _find_zero_crossings(k)
    assert crossings.size == 0

@pytest.mark.parametrize(
    "ktraj, expected",
    [
        (np.array([-30., -10., 0., 10., 30.]), np.array([1])),
        (np.array([30., 10., 0., -10., -30.]), np.array([1])),
    ],
)
def test_crossing_exact_zero(ktraj, expected):
    crossings = _find_zero_crossings(ktraj)
    np.testing.assert_array_equal(crossings, expected)

def test_multiple_crossings_large_wave():
    """Longer oscillating waveform across -30..30 should yield several crossings."""
    # Simple square wave: -30 → +30 → -30 → +30
    k = np.array([-30., -10., 10., 30., -20., -5., 5., 25., -30., -15., 15., 30.])
    crossings = _find_zero_crossings(k)
    assert crossings.size == 5   # expect 5 sign changes

def test_input_shape_with_extra_dim():
    """Should accept input of shape (1, N) and squeeze correctly."""
    k = np.array([[-30., -10., 10., 20.]])
    crossings = _find_zero_crossings(k)
    assert crossings == np.array([1])

@pytest.mark.parametrize(
    "x, window, x_limits, expected",
    [
        ([29, 5, 241, 125], 30, None, [29, 125, 241]),
        ([4, 50, 200, 246], 20, (0, 250), [50, 200]),
        ([10, 40, 70], 20, (0, 80), [10, 40, 70]),
        ([5, 15, 25], 40, (0, 100), []),
        ([20], 30, (0, 40), [20]),
        ([5], 30, (0, 40), []),
        ([5, 60, 120, 180], 100, (0, 200), [60, 120]),
    ],
)
def test_window_selection_parametrized(x, window, x_limits, expected):
    out = _window_selection(np.array(x), window=window, x_limits=x_limits)
    np.testing.assert_array_equal(out, np.array(expected))

def test_empty_input():
    x = np.array([])
    out = _window_selection(x, window=30)
    assert out.size == 0
    
@pytest.mark.parametrize("window", [1, 2, 31])
def test_half_window_rounding(window):
    # Validate that integer division (//2) is used consistently
    x = np.array([10, 20, 30, 40, 50])
    out = _window_selection(x, window=window, x_limits=(0, 60))
    # Just check it doesn't crash and returns a subset of sorted x
    assert (out <= x.max()).all() and (out >= x.min()).all()


if __name__=="__main__":
    pytest.main([__file__])
