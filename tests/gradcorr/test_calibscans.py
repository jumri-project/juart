import pytest
import numpy as np

from juart.gradcorr.calibscans import _find_zero_crossings, _window_selection, _interpolate_signal


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

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline
from typing import Tuple

# Assuming your function is imported or defined here
def interpolate_signal(
    ktraj: np.ndarray, signal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Your interpolation function (included for completeness)"""
    num_cha, _, num_echo = signal.shape
    
    # Find max and min trajectory
    kmin, kmax = ktraj.min(), ktraj.max()
    
    # Generate new, equidistant trajectory samples
    ktraj_interpolate = np.arange(kmin, kmax + 0.5)
    
    # Interpolate
    shape_signal_interp = (num_cha, ktraj_interpolate.size, num_echo)
    signal_interpolate = np.zeros(shape_signal_interp, dtype=np.float32)
    
    for n_echo in range(num_echo):
        finterp = make_interp_spline(
            ktraj[0, :, n_echo], signal[..., n_echo], k=1, axis=1
        )
        signal_interpolate[..., n_echo] = finterp(ktraj_interpolate)
    
    # Return interpolated trajectory in same dimensions as input traj
    ktraj_interpolate = np.broadcast_to(
        ktraj_interpolate[None, :, None],
        (1, ktraj_interpolate.size, num_echo)
    )
    
    return ktraj_interpolate, signal_interpolate


class TestInterpolateSignal:
    """Essential test suite for the interpolate_signal function."""
    
    def setup_method(self):
        """Set up basic test data."""
        # Simple test case
        self.n_samples = 8
        self.n_channels = 2
        self.n_echoes = 2
        
        # Create simple trajectory
        self.ktraj = np.zeros((1, self.n_samples, self.n_echoes))
        self.ktraj[0, :, 0] = np.linspace(-3, 4, self.n_samples)
        self.ktraj[0, :, 1] = np.linspace(-2, 5, self.n_samples)
        
        # Create test signal
        self.signal = np.zeros((self.n_channels, self.n_samples, self.n_echoes), dtype=np.float32)
        for ch in range(self.n_channels):
            for echo in range(self.n_echoes):
                self.signal[ch, :, echo] = np.exp(0.1 * self.ktraj[0, :, echo])
    
    def test_basic_functionality(self):
        """Test that function runs and returns correct shapes."""
        ktraj_interp, signal_interp = interpolate_signal(self.ktraj, self.signal)
        
        # Check shapes
        assert ktraj_interp.shape[0] == 1
        assert ktraj_interp.shape[2] == self.n_echoes
        assert signal_interp.shape[0] == self.n_channels
        assert signal_interp.shape[2] == self.n_echoes
        assert signal_interp.shape[1] == ktraj_interp.shape[1]
        assert signal_interp.dtype == np.float32
    
    def test_trajectory_values(self):
        """Test that output trajectory contains expected integer sequence."""
        ktraj_interp, _ = interpolate_signal(self.ktraj, self.signal)
        
        for echo in range(self.n_echoes):
            traj = ktraj_interp[0, :, echo]
            # Should be consecutive integers
            assert np.all(np.diff(traj) == 1), f"Echo {echo} should have consecutive integer trajectory"
    
    def test_linear_signal_accuracy(self):
        """Test interpolation accuracy with a linear signal."""
        # Simple case with known result
        ktraj = np.zeros((1, 3, 1))
        ktraj[0, :, 0] = [0, 2, 4]  # Non-uniform spacing
        
        signal = np.zeros((1, 3, 1), dtype=np.float32)
        signal[0, :, 0] = ktraj[0, :, 0] * 2  # Linear: y = 2x
        
        ktraj_interp, signal_interp = interpolate_signal(ktraj, signal)
        
        # Should interpolate to [0, 1, 2, 3, 4] with values [0, 2, 4, 6, 8]
        expected = ktraj_interp[0, :, 0] * 2
        np.testing.assert_array_almost_equal(signal_interp[0, :, 0].real, expected, decimal=5)
    
    def test_single_channel_echo(self):
        """Test minimal case: 1 channel, 1 echo."""
        ktraj = np.zeros((1, 4, 1))
        ktraj[0, :, 0] = [-1, 0, 1, 2]
        
        signal = np.ones((1, 4, 1), dtype=np.float32)
        
        ktraj_interp, signal_interp = interpolate_signal(ktraj, signal)
        
        assert ktraj_interp.shape == (1, 4, 1)
        assert signal_interp.shape == (1, 4, 1)
    
    def test_multiple_channels(self):
        """Test that multiple channels work independently."""
        ktraj = np.zeros((1, 3, 1))
        ktraj[0, :, 0] = [0, 1, 2]
        
        signal = np.zeros((2, 3, 1), dtype=np.float32)
        signal[0, :, 0] = 1.0  # Constant
        signal[1, :, 0] = ktraj[0, :, 0]  # Linear
        
        _, signal_interp = interpolate_signal(ktraj, signal)
        
        # Channel 0 should remain constant
        np.testing.assert_array_almost_equal(signal_interp[0, :, 0].real, 1.0, decimal=5)
        # Channel 1 should be linear
        expected_linear = np.array([0, 1, 2], dtype=float)
        np.testing.assert_array_almost_equal(signal_interp[1, :, 0].real, expected_linear, decimal=5)


if __name__=="__main__":
    pytest.main([__file__])
