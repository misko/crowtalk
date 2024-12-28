import pyaudio
import numpy as np
import time

# Parameters
RATE = 16000        # sample rate in Hz
CHUNK = 1024        # frames per buffer
CHANNELS = 5        # we have 4 mics
FORMAT = pyaudio.paInt16
D = 0.0457          # 4.57 cm spacing
C = 343.0           # speed of sound m/s
MAX_TAU = 0.0003    # ~0.3 ms, slightly bigger than (D/C) ~ 0.13 ms

def gcc_phat(sig, refsig, fs=RATE, max_tau=None, interp=16):
    """
    GCC-PHAT function as above.
    """
    n = sig.shape[0] + refsig.shape[0]
    n = 1 << (n - 1).bit_length()
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= (np.abs(R) + 1e-15)
    cc = np.fft.irfft(R, n=n*interp)
    max_shift = n * interp // 2
    if max_tau:
        max_shift = min(int(fs * max_tau * interp), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def estimate_azimuth(tau_x, tau_y, d, c=343.0):
    """
    Map TDOA on x & y axes to azimuth angle [0..360).
    """
    stx = np.clip((c * tau_x) / d, -1.0, 1.0)
    sty = np.clip((c * tau_y) / d, -1.0, 1.0)
    thx = np.arcsin(stx)  # radians
    thy = np.arcsin(sty)  # radians
    thx_deg = np.degrees(thx)
    thy_deg = np.degrees(thy)
    az = np.degrees(np.arctan2(thy_deg, thx_deg)) % 360
    return az

def main():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Capturing audio... Press Ctrl+C to stop.")

    try:
        while True:
            # Read audio from device
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Convert to np array shape (CHUNK, CHANNELS)
            samples = np.frombuffer(data, dtype=np.int16).reshape(CHUNK, CHANNELS)

            # Extract channels (0-indexed):
            ch0 = samples[:, 1].astype(np.float32)
            ch1 = samples[:, 2].astype(np.float32)
            ch2 = samples[:, 3].astype(np.float32)
            ch3 = samples[:, 4].astype(np.float32)

            # We want TDOA on x-axis (ch1 vs ch0) and y-axis (ch2 vs ch0)
            tau_x = gcc_phat(ch1, ch0, fs=RATE, max_tau=MAX_TAU)
            tau_y = gcc_phat(ch2, ch0, fs=RATE, max_tau=MAX_TAU)
            print(tau_x,tau_y)

            # Compute azimuth
            az = estimate_azimuth(tau_x, tau_y, D, C)

            print(f"Azimuth: {az:.2f} deg")

            time.sleep(0.1)  # short delay so we don't spam prints
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

