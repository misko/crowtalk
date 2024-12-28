import pyaudio
import numpy as np
import math
import time

from scipy.optimize import minimize_scalar

###############################
# 0) Configuration
###############################
RATE = 16000
CHUNK = 1024
CHANNELS = 5
FORMAT = pyaudio.paInt16

SPEED_OF_SOUND = 343.0
d = 0.0457  # 4.57 cm side of the square

# Square array, centered at (0,0):
# mic0, mic1, mic2, mic3
mic_positions = np.array([
    [ +d/2, +d/2 ],  # mic0 (reference)
    [ -d/2, +d/2 ],  # mic1
    [ -d/2, -d/2 ],  # mic2
    [ +d/2, -d/2 ],  # mic3
], dtype=float)

MAX_TAU = 0.0005  # ~0.5ms search window in GCC-PHAT

###############################
# 1) GCC-PHAT
###############################
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

###############################
# 2) Predict TDOA
###############################
def predict_tdoa(angle_deg, i, mic_positions, speed=SPEED_OF_SOUND):
    """
    Predict T_{i,0} if plane wave arrives at angle_deg (in degrees).
    mic0 is mic_positions[0].
    i in {1,2,3}.
    """
    angle_rad = math.radians(angle_deg)
    dirx = math.cos(angle_rad)
    diry = math.sin(angle_rad)

    pos0 = mic_positions[0]
    posi = mic_positions[i]

    # path difference = (pos_i dot dir) - (pos0 dot dir)
    pdiff = (posi[0]*dirx + posi[1]*diry) - (pos0[0]*dirx + pos0[1]*diry)
    return pdiff / speed

###############################
# 3) Cost Function
###############################
def cost_angle(angle_deg, t10, t20, t30, mic_positions):
    """
    Sum of squared errors between measured T_{1,0}, T_{2,0}, T_{3,0} 
    and predicted TDOAs at angle_deg.
    """
    p10 = predict_tdoa(angle_deg, 1, mic_positions)
    p20 = predict_tdoa(angle_deg, 2, mic_positions)
    p30 = predict_tdoa(angle_deg, 3, mic_positions)

    err = ((t10 - p10)**2 + 
           (t20 - p20)**2 + 
           (t30 - p30)**2)
    return err

def find_best_angle(t10, t20, t30, mic_positions):
    """
    Minimizes cost_angle over angle in [0..360].
    We'll use minimize_scalar with method='bounded'.
    """
    from scipy.optimize import minimize_scalar

    def obj_fn(angle):
        return cost_angle(angle, t10, t20, t30, mic_positions)

    # We'll do a bounded search from 0..360 degrees
    result = minimize_scalar(obj_fn, 
                             bounds=(0.0, 360.0), 
                             method='bounded')
    
    best_angle_deg = result.x  # might not be integer
    best_error = result.fun
    # ensure angle in [0..360)
    best_angle_deg = best_angle_deg % 360

    return best_angle_deg, best_error

###############################
# 4) Main Real-Time Loop
###############################
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

            best_angle, err = find_best_angle(t10, t20, t30, mic_positions)
            print(f"Angle: {best_angle:6.2f} deg  SSE={err:.6g}   (t10={t10:.5f}, t20={t20:.5f}, t30={t30:.5f})")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

