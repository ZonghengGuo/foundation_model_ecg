import matlab.engine
import os
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
import numpy as np


def call_rrSQI(ecg_signal, eng):
    """
    Parameters:
        ecg_signal (numpy.ndarray): The ECG signal as a 1D NumPy array.
        qrs_indices (numpy.ndarray): The detected QRS peak indices as a 1D NumPy array.

    Returns:
        BeatQ (numpy.ndarray): Quality assessment of each RR interval.
        BeatN (numpy.ndarray): Quality of ECG corresponding to each beat.
        r (float): Fraction of good beats in RR.
    """

    # resampling ECG signals（40 Hz -> 200 Hz）
    ecg_resampled = signal.resample_poly(ecg_signal, up=5, down=1)

    # transfer ecg to matlab format
    ecg_matlab = matlab.double(ecg_resampled.tolist())

    # us matlab QRS function
    r_peaks = eng.QRS(ecg_matlab, float(200))

    # Call the MATLAB function
    r = eng.rrSQI(ecg_matlab, r_peaks, 200.0)

    # Convert MATLAB outputs to NumPy arrays
    # BeatQ_np = np.array(BeatQ)
    # BeatN_np = np.array(BeatN)
    r_val = float(r)

    return r_val

if __name__ == "__main__":
    input_ecg_segment_path = r"D:\code_zongheng\foundation_model\saved_data\ecg_segments"
    save_path = r"D:\code_zongheng\foundation_model\saved_data\quality_labels"

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    eng.addpath(r'D:\code_zongheng\foundation_model\dataset', nargout=0)

    for ecg_segments_path_name in tqdm(os.listdir(input_ecg_segment_path)):
        ecg_segments_path = os.path.join(input_ecg_segment_path, ecg_segments_path_name)
        ecg_segments = np.load(ecg_segments_path)

        ecg_segments_file_name = ecg_segments_path_name.replace('.npy', '')

        qua_labels = []
        for i, ecg_segment in enumerate(ecg_segments):
            qua = call_rrSQI(ecg_segment, eng)
            qua_labels.append(qua)
            print(f"The quality of {i}-th ECG segment in {ecg_segments_path_name} is: {(qua*100):.2f}%")
        np.save((save_path + '/' + ecg_segments_file_name), qua_labels)

    # Close the MATLAB engine
    eng.quit()



