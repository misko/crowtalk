import pyaudio
import numpy as np
import time
import math

###############################
# 0) Configuration
###############################

RATE = 16000       # sampling rate (Hz)
CHUNK = 1024       # frames per buffer
CHANNELS = 5       # 4 mics
FORMAT = pyaudio.paInt16

# Speed of sound (m/s)
C = 343.0
# Side length of the square array in meters
d = 0.0457  # 4.57 cm

# For GCC-PHAT search
MAX_TAU = 0.0004   # about 0.4 ms search window

###############################
# 1) GCC-PHAT Implementation
###############################
def gcc_phat(sig, refsig, fs=RATE, max_tau=None, interp=16):
    """
    Estimate time delay between `sig` and `refsig` using GCC-PHAT.
    Returns the delay (in seconds).
    """
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()

    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)

    # PHAT weighting
    denom = np.abs(R)
    denom[denom < 1e-15] = 1e-15
    R /= denom

    cc = np.fft.irfft(R, n=nfft*interp)

    max_shift = nfft * interp // 2
    if max_tau:
        max_shift = min(int(fs * max_tau * interp), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    return tau

###############################
# 2) Compute Angle Directly
###############################
def direct_azimuth(tau_x, tau_y, d, c=C):
    """
    Given two TDOAs:
        tau_x = TDOA(mic1 vs mic0), x-direction
        tau_y = TDOA(mic3 vs mic0), y-direction
    directly compute a single 2D angle (azimuth).

    Step:
      sinThetaX = c * tau_x / d
      sinThetaY = c * tau_y / d
      theta_x = arcsin(sinThetaX)
      theta_y = arcsin(sinThetaY)
      az = atan2(theta_y, theta_x) in degrees [0..360)
    """
    # clamp to [-1..1]
    stx = np.clip((c * tau_x) / d, -1.0, 1.0)
    sty = np.clip((c * tau_y) / d, -1.0, 1.0)

    # arcsin
    thx = math.degrees(math.asin(stx))
    thy = math.degrees(math.asin(sty))

    # combine
    az = math.degrees(math.atan2(thy, thx)) % 360
    return az

###############################
# 3) Main Real-Time Loop
###############################
def main():
    import pyaudio

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).reshape(CHUNK, CHANNELS)

            # Extract channels (float32 for GCC-PHAT)
            ch0 = samples[:, 0+1].astype(np.float32)
            ch1 = samples[:, 1+1].astype(np.float32)
            ch2 = samples[:, 2+1].astype(np.float32)
            ch3 = samples[:, 3+1].astype(np.float32)

            # TDOA for "x-axis": mic1 vs mic0
            tau_x = gcc_phat(ch1, ch0, fs=RATE, max_tau=MAX_TAU)
            # TDOA for "y-axis": mic3 vs mic0
            tau_y = gcc_phat(ch3, ch0, fs=RATE, max_tau=MAX_TAU)

            # Convert to direct angle
            az_deg = direct_azimuth(tau_x, tau_y, d, C)

            print(f"Azimuth: {az_deg:6.2f} deg    (tau_x={tau_x:.5f}s, tau_y={tau_y:.5f}s)")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

