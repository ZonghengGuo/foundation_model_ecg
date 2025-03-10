import wfdb
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg


def peak_detection(sig, fs):
    r = ecg.ecg(signal=sig, sampling_rate=fs, show=False)
    return r['rpeaks']


if __name__ == '__main__':
    database_name = 'mimic3wdb'
    rel_segment_name = '3047608_0002'
    rel_segment_dir = 'mimic3wdb/30/3047608'
    segment_data = wfdb.rdrecord(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    ecg_signal = segment_data.p_signal[:, 0][8000:12000]
    sampling_rate = segment_data.fs

    r_peaks = peak_detection(ecg_signal, sampling_rate)

    print(f"detect {len(r_peaks)} R peaks")
    print(f"R peaks index: {r_peaks}")

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.2)

    time = np.arange(len(ecg_signal))/sampling_rate
    line, = ax.plot(time, ecg_signal, label='ECG Signal')

    peaks_scatter = ax.scatter(r_peaks/sampling_rate, ecg_signal[r_peaks], color='red', s=50, label='R Peaks')

    ax.set_title('ECG Signal with Detected R Peaks')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    plt.show()