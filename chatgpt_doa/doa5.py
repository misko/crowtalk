import pyaudio
import numpy as np
import time
import math

from scipy.optimize import least_squares

#####################################
# 0) Constants and Configuration
#####################################

# Speed of sound (m/s)
SPEED_OF_SOUND = 343.0

# Side length of the square array in meters
d = 0.0457  # 4.57 cm

# Microphone positions, centered at (0,0)
# mic0 reference
mic_positions = np.array([
    [ d/2,  d/2],  # mic0
    [-d/2,  d/2],  # mic1
    [-d/2, -d/2],  # mic2
    [ d/2, -d/2],  # mic3
])

# PyAudio settings
FORMAT = pyaudio.paInt16
CHANNELS = 5
RATE = 16000     # sample rate (Hz)
CHUNK = 1024     # frames per buffer
MAX_TAU = 0.0004 # ~0.4 ms search window, a bit more than needed for 4.57 cm


#####################################
# 1) GCC-PHAT Function
#####################################
def gcc_phat(sig, refsig, fs=RATE, max_tau=None, interp=16):
    """
    Estimate time delay between `sig` and `refsig` using GCC-PHAT.
    Returns the delay (in seconds).
    """
    n = sig.shape[0] + refsig.shape[0]
    # Next power of 2
    nfft = 1 << (n - 1).bit_length()

    # FFT
    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)

    # Cross-spectral density
    R = SIG * np.conj(REFSIG)

    # PHAT weighting (normalize magnitude)
    denom = np.abs(R)
    denom[denom < 1e-15] = 1e-15
    R /= denom

    # Inverse FFT
    cc = np.fft.irfft(R, n=nfft * interp)

    max_shift = nfft * interp // 2
    if max_tau:
        # The maximum sample shift we care about
        max_shift = min(int(fs * max_tau * interp), max_shift)

    # only keep the useful range around zero-lag
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find the index of max
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    return tau


#####################################
# 2) TDOA => (x,y) via Nonlinear Solver
#####################################
def predicted_tdoas(xy, mic_positions, speed=SPEED_OF_SOUND):
    """
    For a candidate source location xy = (x,y),
    compute T_{1,0}, T_{2,0}, T_{3,0} as predicted by geometry.
    mic0 is positions[0], mic1 => positions[1], etc.

    Returns array [ T_{1,0}, T_{2,0}, T_{3,0} ].
    """
    x0, y0 = mic_positions[0]  # mic0
    d0 = np.sqrt((xy[0]-x0)**2 + (xy[1]-y0)**2)  # distance to mic0

    tdoas = []
    for i in [1,2,3]:
        xi, yi = mic_positions[i]
        di = np.sqrt((xy[0]-xi)**2 + (xy[1]-yi)**2)
        # T_{i,0} = (distance_i - distance_0)/speed
        tdoas.append( (di - d0)/speed )
    return np.array(tdoas)

def residuals_tdoa(xy, measured, mic_positions, speed=SPEED_OF_SOUND):
    """
    The difference between measured TDOAs and predicted TDOAs for location (x,y).
    measured = [t_{1,0}, t_{2,0}, t_{3,0}]
    """
    pred = predicted_tdoas(xy, mic_positions, speed)
    return measured - pred

def solve_for_location(measured_tdoas, mic_positions, speed=SPEED_OF_SOUND):
    """
    Solve for (x,y) that best matches the measured T_{1,0}, T_{2,0}, T_{3,0}.
    We'll do a least-squares approach using scipy.
    """
    from scipy.optimize import least_squares
    x0 = np.array([0.1, 0.0])  # initial guess
    res = least_squares(
        residuals_tdoa, x0,
        args=(measured_tdoas, mic_positions, speed),
        method='lm'
    )
    return res.x  # (x*, y*)

def compute_azimuth(xy):
    """
    Convert (x,y) to an azimuth in degrees [0..360).
    """
    angle = math.degrees(math.atan2(xy[1], xy[0]))
    return angle % 360


#####################################
# 3) Main Real-Time Capture + Solve
#####################################
def main():
    import pyaudio

    # Setup PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Listening... Press Ctrl+C to stop.")

    try:
        while True:
            # Read audio block
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert to np array shape = (CHUNK, CHANNELS)
            samples = np.frombuffer(data, dtype=np.int16).reshape(CHUNK, CHANNELS)

            # Convert each channel to float32
            ch0 = samples[:, 0+1].astype(np.float32)
            ch1 = samples[:, 1+1].astype(np.float32)
            ch2 = samples[:, 2+1].astype(np.float32)
            ch3 = samples[:, 3+1].astype(np.float32)

            # Estimate TDOAs: T_{1,0}, T_{2,0}, T_{3,0}
            t10 = gcc_phat(ch1, ch0, fs=RATE, max_tau=MAX_TAU)
            t20 = gcc_phat(ch2, ch0, fs=RATE, max_tau=MAX_TAU)
            t30 = gcc_phat(ch3, ch0, fs=RATE, max_tau=MAX_TAU)

            measured_tdoas = np.array([t10, t20, t30], dtype=float)

            # Solve for (x,y)
            xy_est = solve_for_location(measured_tdoas, mic_positions, SPEED_OF_SOUND)
            # Convert to azimuth
            az_deg = compute_azimuth(xy_est)

            print(f"DoA: {az_deg:6.2f} deg  (x={xy_est[0]:.3f}, y={xy_est[1]:.3f})")

            time.sleep(0.1)  # small pause to reduce CPU usage

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

