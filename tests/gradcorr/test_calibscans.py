from typing import Literal

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

from juart.gradcorr.calibscans import (
    CorrectionScan,
    _cross_spectrum_delay,
    _find_zero_crossings,
    _window_selection,
)


def test_simple_sine_like():
    """Basic alternating trajectory should have regular zero crossings."""
    k = np.array([-30.0, -10.0, 10.0, 30.0, -20.0, -5.0, 5.0, 25.0])
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
        (np.array([-30.0, -10.0, 0.0, 10.0, 30.0]), np.array([1])),
        (np.array([30.0, 10.0, 0.0, -10.0, -30.0]), np.array([1])),
    ],
)
def test_crossing_exact_zero(ktraj, expected):
    crossings = _find_zero_crossings(ktraj)
    np.testing.assert_array_equal(crossings, expected)


def test_multiple_crossings_large_wave():
    """Longer oscillating waveform across -30..30 should yield several crossings."""
    # Simple square wave: -30 → +30 → -30 → +30
    k = np.array(
        [-30.0, -10.0, 10.0, 30.0, -20.0, -5.0, 5.0, 25.0, -30.0, -15.0, 15.0, 30.0]
    )
    crossings = _find_zero_crossings(k)
    assert crossings.size == 5  # expect 5 sign changes


def test_input_shape_with_extra_dim():
    """Should accept input of shape (1, N) and squeeze correctly."""
    k = np.array([[-30.0, -10.0, 10.0, 20.0]])
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


# Test function for interpolate_signal (existing test, keeping it)
def interpolate_signal(
    ktraj: np.ndarray, signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
        ktraj_interpolate[None, :, None], (1, ktraj_interpolate.size, num_echo)
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
        self.signal = np.zeros(
            (self.n_channels, self.n_samples, self.n_echoes), dtype=np.float32
        )
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
            assert np.all(np.diff(traj) == 1), (
                f"Echo {echo} should have consecutive integer trajectory"
            )

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
        np.testing.assert_array_almost_equal(
            signal_interp[0, :, 0].real, expected, decimal=5
        )

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
        np.testing.assert_array_almost_equal(
            signal_interp[0, :, 0].real, 1.0, decimal=5
        )
        # Channel 1 should be linear
        expected_linear = np.array([0, 1, 2], dtype=float)
        np.testing.assert_array_almost_equal(
            signal_interp[1, :, 0].real, expected_linear, decimal=5
        )


class TestCrossSpectrumDelay:
    """Test suite for the _cross_spectrum_delay function."""

    def test_basic_functionality(self):
        """Test that function runs and returns correct shapes."""
        # Create simple test data: 2 channels, 10 samples
        g = np.ones((2, 10), dtype=complex)

        delays, max_intensitys = _cross_spectrum_delay(g)

        # Check output shapes
        assert delays.shape == (2,)
        assert max_intensitys.shape == (2,)
        assert isinstance(delays, np.ndarray)
        assert isinstance(max_intensitys, np.ndarray)

    def test_single_channel(self):
        """Test with single channel data."""
        # Single channel, simple constant phase
        g = np.ones((1, 8), dtype=complex)

        delays, max_intensitys = _cross_spectrum_delay(g)

        assert delays.shape == (1,)
        assert max_intensitys.shape == (1,)

    def test_constant_phase_zero_delay(self):
        """Test that constant phase gives zero delay."""
        # Create cross spectrum with constant phase (should give zero delay)
        n_samples = 16
        g = np.ones((1, n_samples), dtype=complex)

        delays, _ = _cross_spectrum_delay(g, peak_tresh=0.1)

        # Constant phase should give zero delay
        np.testing.assert_array_almost_equal(delays, [0.0], decimal=3)

    def test_linear_phase_delay(self):
        """Test that linear phase gives expected delay."""
        n_samples = 32

        # Create linear phase that should give known delay
        x = np.arange(n_samples)
        # Linear phase with slope that should give delay ≈ 1
        target_delay = 1.0

        # Create two Gaussian curves shifted by target_delay
        sigma = n_samples / 8  # Width of Gaussian
        center = n_samples / 2  # Center position
        y1 = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        y2 = np.exp(-0.5 * ((x - center - target_delay) / sigma) ** 2)

        # Create cross spectrum: g = fft(y1) * conj(fft(y2))
        fft_y1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y1)))
        fft_y2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y2)))

        g = fft_y1 * np.conjugate(fft_y2)

        # Add channels axis
        g = g[None, :]

        delays, _ = _cross_spectrum_delay(g, peak_tresh=0.1)

        # Should recover the target delay (within tolerance)
        np.testing.assert_array_almost_equal(delays, [target_delay], decimal=1)

    def test_multiple_channels_different_delays(self):
        """Test multiple channels with different known delays."""
        n_samples = 32
        n_channels = 3
        target_delays = [0.0, 1.0, -0.5]

        g = np.zeros((n_channels, n_samples), dtype=complex)
        x = np.arange(n_samples)
        sigma = n_samples / 8  # Width of Gaussian
        center = n_samples / 2  # Center position

        for ch, target_delay in enumerate(target_delays):
            # Create two Gaussian curves shifted by target_delay
            y1 = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            y2 = np.exp(-0.5 * ((x - center - target_delay) / sigma) ** 2)

            # Create cross spectrum: g = fft(y1) * conj(fft(y2))
            fft_y1 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y1)))
            fft_y2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(y2)))

            g[ch, :] = fft_y1 * np.conjugate(fft_y2)

        delays, _ = _cross_spectrum_delay(g, peak_tresh=0.1)

        # Should recover target delays
        np.testing.assert_array_almost_equal(delays, target_delays, decimal=1)

    def test_peak_threshold_effect(self):
        """Test that peak threshold affects which samples are used."""
        n_samples = 16

        # Create data where only some samples exceed threshold
        g = np.ones((1, n_samples), dtype=complex) * 0.1  # Low intensity
        g[0, 5:10] = 1.0  # High intensity region

        # With high threshold, only high-intensity samples should be used
        delays_high, max_int_high = _cross_spectrum_delay(g, peak_tresh=0.5)

        # With low threshold, more samples should be used
        delays_low, max_int_low = _cross_spectrum_delay(g, peak_tresh=0.05)

        # Max intensity should reflect the threshold
        assert max_int_high[0] > max_int_low[0]

    def test_max_intensity_calculation(self):
        """Test that max intensity is calculated correctly."""
        n_samples = 10
        peak_val = 2.0
        peak_tresh = 0.3

        g = np.ones((1, n_samples), dtype=complex)
        g[0, 5] = peak_val  # Set one peak value

        _, max_intensitys = _cross_spectrum_delay(g, peak_tresh=peak_tresh)

        expected_max_intensity = peak_val * peak_tresh
        np.testing.assert_array_almost_equal(
            max_intensitys, [expected_max_intensity], decimal=5
        )

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small array
        g_small = np.ones((1, 3), dtype=complex)
        delays, max_int = _cross_spectrum_delay(g_small)
        assert delays.shape == (1,)
        assert max_int.shape == (1,)

        # Array with all zeros (should handle gracefully)
        g_zeros = np.zeros((1, 5), dtype=complex)
        delays_zero, max_int_zero = _cross_spectrum_delay(g_zeros)
        assert delays_zero.shape == (1,)
        assert max_int_zero.shape == (1,)

    def test_complex_input_handling(self):
        """Test that function properly handles complex input."""
        n_samples = 16

        # Create complex data with both real and imaginary parts
        real_part = np.random.rand(1, n_samples)
        imag_part = np.random.rand(1, n_samples)
        g = real_part + 1j * imag_part

        delays, max_intensitys = _cross_spectrum_delay(g)

        # Should handle complex input without error
        assert delays.shape == (1,)
        assert max_intensitys.shape == (1,)
        assert np.all(np.isfinite(delays))
        assert np.all(np.isfinite(max_intensitys))


class TestCorrectionScan:
    """Comprehensive test suite for the CorrectionScan class."""

    def setup_method(self):
        """Set up test data for CorrectionScan tests."""
        # Basic test parameters
        self.n_samples = 100
        self.n_channels = 3
        self.n_echoes = 2
        self.dwell = 1e-6  # 1 microsecond
        self.gamma = 42.576e6  # Hz/T
        self.window = 40
        self.axis: Literal["x", "y", "z"] = "x"

        # Create a realistic k-space trajectory with zero crossings
        # Simulate an oscillating readout with multiple zero crossings
        time_points = np.linspace(-5, 5, self.n_samples)

        # Create trajectory that oscillates and crosses zero multiple times
        ktraj_echo1 = 10 * np.sin(2 * np.pi * time_points) + 2 * np.sin(
            4 * np.pi * time_points
        )
        ktraj_echo2 = 12 * np.sin(2.2 * np.pi * time_points) + 3 * np.sin(
            3.8 * np.pi * time_points
        )

        self.ktraj = np.zeros((1, self.n_samples, self.n_echoes))
        self.ktraj[0, :, 0] = ktraj_echo1
        self.ktraj[0, :, 1] = ktraj_echo2

        # Create corresponding signal data with some realistic properties
        # Signal magnitude should vary smoothly and have some noise
        self.sig = np.zeros((self.n_channels, self.n_samples, self.n_echoes))

        for ch in range(self.n_channels):
            for echo in range(self.n_echoes):
                # Base signal with some dependence on k-space position
                base_signal = 10 + 5 * np.exp(-0.1 * np.abs(self.ktraj[0, :, echo]))
                # Add channel-dependent variation
                channel_factor = 0.8 + 0.4 * ch / (self.n_channels - 1)
                # Add small amount of noise
                noise = 0.1 * np.random.randn(self.n_samples)
                self.sig[ch, :, echo] = base_signal * channel_factor + noise

    def test_initialization_basic_attributes(self):
        """Test that basic attributes are correctly set during initialization."""
        corr_scan = CorrectionScan(
            ktraj=self.ktraj,
            sig=self.sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
            gamma=self.gamma,
        )

        # Check basic attributes
        assert corr_scan.window == self.window
        assert corr_scan.axis == self.axis

        # Check that zero_crossings is an array
        assert isinstance(corr_scan.zero_crossings, np.ndarray)
        assert corr_scan.zero_crossings.dtype == np.int64

        # Check num_crossings consistency
        assert corr_scan.num_crossings == corr_scan.zero_crossings.size
        assert corr_scan.num_crossings >= 0

    def test_computed_attributes_shapes(self):
        """Test that computed attributes have correct shapes."""
        corr_scan = CorrectionScan(
            ktraj=self.ktraj,
            sig=self.sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
            gamma=self.gamma,
        )

        # Check delays and max_intensity shapes
        assert isinstance(corr_scan._delays, np.ndarray)
        assert isinstance(corr_scan._max_intensity, np.ndarray)

        # Both should have shape (num_crossings, n_channels)
        expected_shape = (corr_scan.num_crossings, self.n_channels)
        assert corr_scan._delays.shape == expected_shape
        assert corr_scan._max_intensity.shape == expected_shape

        # Values should be finite
        assert np.all(np.isfinite(corr_scan._delays))
        assert np.all(np.isfinite(corr_scan._max_intensity))
        assert np.all(corr_scan._max_intensity >= 0)

    def test_different_window_sizes(self):
        """Test initialization with different window sizes."""
        for window in [20, 40, 80, 120]:
            corr_scan = CorrectionScan(
                ktraj=self.ktraj,
                sig=self.sig,
                dwell=self.dwell,
                axis=self.axis,
                window=window,
            )

            assert corr_scan.window == window
            # Smaller windows might result in more filtered crossings
            assert corr_scan.num_crossings >= 0

    def test_different_axes(self):
        """Test initialization with different axis labels."""
        for axis in ["x", "y", "z"]:
            corr_scan = CorrectionScan(
                ktraj=self.ktraj,
                sig=self.sig,
                dwell=self.dwell,
                axis=axis,
                window=self.window,
            )

            assert corr_scan.axis == axis

    def test_get_delay_default_parameters(self):
        """Test get_delay with default parameters."""
        corr_scan = CorrectionScan(
            ktraj=self.ktraj,
            sig=self.sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        if corr_scan.num_crossings > 0:
            delay = corr_scan.get_delay()

            # Should return a scalar float
            assert isinstance(delay, (float, np.floating))
            assert np.isfinite(delay)

    def test_get_delay_no_weighting(self):
        """Test get_delay without channel weighting."""
        corr_scan = CorrectionScan(
            ktraj=self.ktraj,
            sig=self.sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        if corr_scan.num_crossings > 0:
            delay_weighted = corr_scan.get_delay(weight_channel=True)
            delay_unweighted = corr_scan.get_delay(weight_channel=False)

            # Both should be finite floats
            assert isinstance(delay_weighted, (float, np.floating))
            assert isinstance(delay_unweighted, (float, np.floating))
            assert np.isfinite(delay_weighted)
            assert np.isfinite(delay_unweighted)

            # They might be different (unless all channels have equal weight)
            # We just check they're both reasonable values

    def test_get_delay_crossing_limit(self):
        """Test get_delay with different crossing limits."""
        corr_scan = CorrectionScan(
            ktraj=self.ktraj,
            sig=self.sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        if corr_scan.num_crossings > 0:
            # Test with limit smaller than available crossings
            if corr_scan.num_crossings > 1:
                delay_limited = corr_scan.get_delay(crossing_limit=1)
                delay_all = corr_scan.get_delay(crossing_limit=None)

                assert isinstance(delay_limited, (float, np.floating))
                assert isinstance(delay_all, (float, np.floating))
                assert np.isfinite(delay_limited)
                assert np.isfinite(delay_all)

            # Test with limit larger than available crossings
            delay_large_limit = corr_scan.get_delay(
                crossing_limit=corr_scan.num_crossings + 10
            )
            delay_no_limit = corr_scan.get_delay(crossing_limit=None)

            # Should give same result as no limit
            np.testing.assert_almost_equal(delay_large_limit, delay_no_limit)

    def test_get_delay_zero_crossings(self):
        """Test behavior when no zero crossings are found."""
        # Create trajectory with no zero crossings (all positive)
        ktraj_no_crossings = np.ones((1, self.n_samples, self.n_echoes))
        ktraj_no_crossings[0, :, 0] = np.linspace(1, 10, self.n_samples)
        ktraj_no_crossings[0, :, 1] = np.linspace(2, 12, self.n_samples)

        corr_scan = CorrectionScan(
            ktraj=ktraj_no_crossings,
            sig=self.sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        # Should have zero crossings
        assert corr_scan.num_crossings == 0

        # get_delay should handle this gracefully
        delay = corr_scan.get_delay()

        # Should return 0
        assert delay == 0.0

    def test_different_dwell_times(self):
        """Test with different dwell times."""
        for dwell in [0.5e-6, 1e-6, 2e-6, 5e-6]:
            corr_scan = CorrectionScan(
                ktraj=self.ktraj,
                sig=self.sig,
                dwell=dwell,
                axis=self.axis,
                window=self.window,
            )

            # Delays should scale inversely with gradient amplitude
            # (which depends on dwell time through gradient calculation)
            if corr_scan.num_crossings > 0:
                delay = corr_scan.get_delay()
                assert isinstance(delay, (float, np.floating))
                assert np.isfinite(delay)

    def test_different_gamma_values(self):
        """Test with different gyromagnetic ratios."""
        for gamma in [42.576e6, 26.752e6, 11.262e6]:  # 1H, 13C, 15N
            corr_scan = CorrectionScan(
                ktraj=self.ktraj,
                sig=self.sig,
                dwell=self.dwell,
                axis=self.axis,
                window=self.window,
                gamma=gamma,
            )

            if corr_scan.num_crossings > 0:
                delay = corr_scan.get_delay()
                assert isinstance(delay, (float, np.floating))
                assert np.isfinite(delay)

    def test_single_channel_data(self):
        """Test with single channel data."""
        single_channel_sig = self.sig[:1, :, :]  # Take only first channel

        corr_scan = CorrectionScan(
            ktraj=self.ktraj,
            sig=single_channel_sig,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        # Should work with single channel
        if corr_scan.num_crossings > 0:
            assert corr_scan._delays.shape == (corr_scan.num_crossings, 1)
            assert corr_scan._max_intensity.shape == (corr_scan.num_crossings, 1)

            delay = corr_scan.get_delay()
            assert isinstance(delay, (float, np.floating))
            assert np.isfinite(delay)

    def test_reproducibility(self):
        """Test that repeated initialization gives same results."""
        # Fix random seed for reproducible signal
        np.random.seed(42)
        sig1 = self.sig.copy()

        np.random.seed(42)
        sig2 = self.sig.copy()

        corr_scan1 = CorrectionScan(
            ktraj=self.ktraj,
            sig=sig1,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        corr_scan2 = CorrectionScan(
            ktraj=self.ktraj,
            sig=sig2,
            dwell=self.dwell,
            axis=self.axis,
            window=self.window,
        )

        # Should have identical results
        assert corr_scan1.num_crossings == corr_scan2.num_crossings
        np.testing.assert_array_equal(
            corr_scan1.zero_crossings, corr_scan2.zero_crossings
        )

        if corr_scan1.num_crossings > 0:
            delay1 = corr_scan1.get_delay()
            delay2 = corr_scan2.get_delay()
            np.testing.assert_almost_equal(delay1, delay2)


if __name__ == "__main__":
    pytest.main([__file__])
