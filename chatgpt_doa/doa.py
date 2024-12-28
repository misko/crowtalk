import pyaudio
import numpy as np
import math
import sys

########################################################################
# Constants / Configuration
########################################################################
CHUNK = 1024               # Frames per buffer
RATE = 16000               # Sampling rate
CHANNELS = 4               # Four microphones
DEVICE_INDEX = 1           # Replace with your mic array's device index
MIC_DISTANCE = 0.057       # Distance between mic 0 and mic 1 in meters (approx for ReSpeaker 4-Mic USB)
SPEED_OF_SOUND = 343.0     # m/s (approx at 20C)
MAX_TDOA = MIC_DISTANCE / SPEED_OF_SOUND  # Maximum possible time delay for two mics
MAX_SAMPLES_LAG = int(MAX_TDOA * RATE)    # Max sample lag

########################################################################
# Helper function: GCC-PHAT
########################################################################
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    """
    Generalized Cross Correlation - Phase Transform (GCC-PHAT)
    Returns the delay (in seconds) that maximizes the cross-correlation.

    :param sig: signal
    :param refsig: reference signal
    :param fs: sampling frequency
    :param max_tau: maximum correlation shift
    :param interp: interpolation factor
    :return: estimated delay
    """
    n = sig.shape[0] + refsig.shape[0]
    # FFT length
    n = 1 << (n-1).bit_length()

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    # GCC-PHAT
    R = R / (np.abs(R) + 1e-15)

    cc = np.fft.irfft(R, n=n * interp)
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    return tau

########################################################################
# Helper function: Compute angle from time delay
########################################################################
def estimate_angle(tdoa, mic_distance, speed_of_sound):
    """
    Estimate angle (in radians) from time difference of arrival between two mics.
    We assume the mics are placed horizontally in 2D.
    
    :param tdoa: time difference of arrival (seconds)
    :param mic_distance: distance between mics (meters)
    :param speed_of_sound: speed of sound in m/s
    :return: angle in degrees (0 degrees = broadside, 90 deg = endfire)
    """
    # Clamp tdoa to max feasible
    max_tdoa = mic_distance / speed_of_sound
    if abs(tdoa) > max_tdoa:
        # The computed TDOA is out-of-bounds; ignore or clamp
        tdoa = np.sign(tdoa) * max_tdoa

    # arcsin( c * tdoa / d )
    # This is the angle from perpendicular to the line between the two mics.
    # Convert from radians to degrees:
    angle_radians = math.asin((speed_of_sound * tdoa) / mic_distance)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

########################################################################
# Main: Real-Time Loop
########################################################################
def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=DEVICE_INDEX)

    print("Starting real-time DOA estimation. Press Ctrl+C to stop.")
    try:
        while True:
            # Read raw bytes from stream
            data = stream.read(CHUNK, exception_on_overflow=False)

            # Convert to int16 numpy array, shape = (CHUNK * CHANNELS,)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Reshape to separate channels: shape = (CHUNK, CHANNELS)
            audio_data = np.reshape(audio_data, (CHUNK, CHANNELS))

            # Extract two channels (for example, ch0 and ch1)
            ch0 = audio_data[:, 0].astype(np.float32)
            ch1 = audio_data[:, 1].astype(np.float32)

            # Compute TDOA via GCC-PHAT
            tau = gcc_phat(ch0, ch1, fs=RATE, max_tau=MAX_TDOA, interp=1)

            # Convert TDOA (seconds) to angle (degrees)
            angle = estimate_angle(tau, MIC_DISTANCE, SPEED_OF_SOUND)

            # Print or log the angle
            print(f"TDOA: {tau:.6f} s, Angle: {angle:.2f} degrees")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()

