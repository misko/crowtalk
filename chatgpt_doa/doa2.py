import numpy as np
import pyaudio
from scipy.optimize import least_squares

# Speed of sound
SPEED_OF_SOUND = 343.0  # m/s

# Square geometry: (adjust for your real positions)
# M0 = (0, 0), M1 = (d, 0), M2 = (0, d), M3 = (d, d)
d=0.0457
MIC_POS = np.array([
    [d/2, d/2],  # M0
    [-d/2, d/2], # M1 (example 6cm spacing)
    [-d/2, -d/2], # M2
    [d/2, -d/2] # M3
])

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    """
    Simplified GCC-PHAT for TDOA estimation. 
    """
    n = sig.shape[0] + refsig.shape[0]
    n = 1 << (n - 1).bit_length()
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-15
    cc = np.fft.irfft(R, n=n * interp)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def predicted_tdoa(xy, mic_pos, speed):
    """
    Given a source at (x, y), compute TDOAs to M0 for each of M1, M2, M3.
    Returns array of shape (3,).
    """
    (x_s, y_s) = xy
    d0 = np.sqrt((x_s - mic_pos[0,0])**2 + (y_s - mic_pos[0,1])**2)
    t0 = d0 / speed

    tdoas = []
    for i in [1, 2, 3]:
        di = np.sqrt((x_s - mic_pos[i,0])**2 + (y_s - mic_pos[i,1])**2)
        ti = di / speed
        tdoas.append(ti - t0)  # T_{i,0}
    return np.array(tdoas)

def residuals_tdoa(xy, measured_tdoas, mic_pos, speed):
    """
    Return difference between measured TDOAs and predicted TDOAs 
    for a candidate source location xy = (x, y).
    """
    return measured_tdoas - predicted_tdoa(xy, mic_pos, speed)

def find_source_location(tdoas, mic_pos, speed_of_sound):
    """
    Solve for (x,y) using least squares with measured TDOAs for M1, M2, M3 (ref=M0).
    tdoas = [Δt_{1,0}, Δt_{2,0}, Δt_{3,0}]
    """
    # Initial guess (just guess the source is 1 meter in front of M0)
    x0 = [0.5, 0.5]  
    
    # Run least squares
    res = least_squares(
        fun=residuals_tdoa,
        x0=x0,
        args=(tdoas, mic_pos, speed_of_sound),
        method='lm',  # Levenberg-Marquardt, for instance
    )
    return res.x  # (x_solved, y_solved)

def main():
    # Setup PyAudio (example settings)
    p = pyaudio.PyAudio()
    fs=16000
    read_size=1024*4
    print("OPEN STREAM")
    stream = p.open(format=pyaudio.paInt16,
                    channels=5,
                    rate=fs,
                    input=True,
                    frames_per_buffer=read_size,
                    input_device_index=0)  # adjust device index
     
    print("Start capturing. Press Ctrl+C to stop.")
    try:
        while True:
            # Read one chunk
            data = stream.read(read_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data = np.reshape(audio_data, (read_size, 5))

            # Extract channels
            #channel 0 is ASR
            ch0 = audio_data[:, 0+1].astype(np.float32)
            ch1 = audio_data[:, 1+1].astype(np.float32)
            ch2 = audio_data[:, 2+1].astype(np.float32)
            ch3 = audio_data[:, 3+1].astype(np.float32)

            # Compute TDOAs relative to ch0
            dt10 = gcc_phat(ch1, ch0, fs=fs)
            dt20 = gcc_phat(ch2, ch0, fs=fs)
            dt30 = gcc_phat(ch3, ch0, fs=fs)
            print(dt10,dt20,dt30)

            measured_tdoas = np.array([dt10, dt20, dt30])

            # Solve for (x, y)
            xy_est = find_source_location(measured_tdoas, MIC_POS, SPEED_OF_SOUND)
            x_est, y_est = xy_est

            # Convert to angle in degrees, relative to mic M0
            angle_rad = np.arctan2(y_est, x_est)
            angle_deg = np.degrees(angle_rad)

            print(f"Estimated source: (x={x_est:.2f} m, y={y_est:.2f} m), angle={angle_deg:.2f}°")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

