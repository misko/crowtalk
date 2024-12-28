import pyaudio
import numpy as np
import time
import math

############################
# Configuration
############################
RATE = 16000
CHUNK = 1024
CHANNELS = 5
FORMAT = pyaudio.paInt16

SPEED_OF_SOUND = 343.0
d = 0.0457  # 4.57 cm

# Microphone positions (square, centered at (0,0)):
mic_positions = np.array([
    [ +d/2, +d/2 ],  # mic0 (reference)
    [ -d/2, +d/2 ],  # mic1
    [ -d/2, -d/2 ],  # mic2
    [ +d/2, -d/2 ],  # mic3
], dtype=float)

MAX_TAU = 0.0005  # ~0.5ms search window in GCC-PHAT

############################
# GCC-PHAT for TDOA
############################
def gcc_phat(sig, refsig, fs=RATE, max_tau=None, interp=16):
    """
    Returns estimated time delay (in seconds) between sig and refsig via GCC-PHAT.
    """
    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()

    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)

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

############################
# 1D angle search
############################
def predict_tdoa(angle_deg, i, mic_positions, speed=SPEED_OF_SOUND):
    """
    Predict TDOA of mic i vs mic0 if the plane wave is from angle_deg.
    angle_deg: in degrees
    i: mic index 1..3
    """
    angle_rad = math.radians(angle_deg)
    # direction vector
    dirx = math.cos(angle_rad)
    diry = math.sin(angle_rad)

    # dot positions with the direction
    pos0 = mic_positions[0]
    posi = mic_positions[i]

    # path difference = (pos_i dot dir) - (pos_0 dot dir)
    pdiff = (posi[0]*dirx + posi[1]*diry) - (pos0[0]*dirx + pos0[1]*diry)

    # TDOA = pathDiff / speed
    return pdiff / speed

def angle_search_3tdoas(t10, t20, t30, mic_positions):
    """
    We have measured TDOAs:
      t10 = T_{1,0}, t20 = T_{2,0}, t30 = T_{3,0}.
    We'll search angles 0..359 deg to find which best matches these TDOAs in a least-squares sense.
    Returns the best angle in [0..360), plus the minimal error.
    """
    best_angle = 0.0
    best_error = 1e9

    # Step size for angle search (in degrees)
    step = 1.0

    for deg in np.arange(0, 360, step):
        # predicted T_{1,0}, T_{2,0}, T_{3,0} at angle deg
        p10 = predict_tdoa(deg, 1, mic_positions)
        p20 = predict_tdoa(deg, 2, mic_positions)
        p30 = predict_tdoa(deg, 3, mic_positions)

        # sum of squared errors
        err = ((t10 - p10)**2 + (t20 - p20)**2 + (t30 - p30)**2)
        if err < best_error:
            best_error = err
            best_angle = deg

    return best_angle, best_error

############################
# Main Real-Time Loop
############################
def main():
    import pyaudio

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Capturing audio... Press Ctrl+C to stop.")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).reshape(CHUNK, CHANNELS)

            # Convert each channel to float32
            ch0 = samples[:,0+1].astype(np.float32)
            ch1 = samples[:,1+1].astype(np.float32)
            ch2 = samples[:,2+1].astype(np.float32)
            ch3 = samples[:,3+1].astype(np.float32)

            # TDOAs: mic1->mic0, mic2->mic0, mic3->mic0
            t10 = gcc_phat(ch1, ch0, fs=RATE, max_tau=MAX_TAU)
            t20 = gcc_phat(ch2, ch0, fs=RATE, max_tau=MAX_TAU)
            t30 = gcc_phat(ch3, ch0, fs=RATE, max_tau=MAX_TAU)

            # search angles 0..359 to find best
            angle_deg, err = angle_search_3tdoas(t10, t20, t30, mic_positions)
            print(f"Angle: {angle_deg:6.2f} deg (err={err:.6g})  t10={t10:.5f}, t20={t20:.5f}, t30={t30:.5f}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

