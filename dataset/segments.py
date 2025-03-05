from pathlib import PurePosixPath
import wfdb
from tqdm import tqdm
import random
import posixpath
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction


# Preprocessing function
def is_nan_ratio_exceed(sig, threshold):
    nan_ratio = np.isnan(sig).sum() / 3750  # count nan value
    return nan_ratio > threshold # tell if reach the limit

# Downsample signals
def downsample(signal_data, fs_orig, fs_new):
    ratio = Fraction(fs_new, fs_orig).limit_denominator()
    up = ratio.numerator
    down = ratio.denominator
    return resample_poly(signal_data, up, down)

# Min-max normalization
def min_max_normalize(signal, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    signal = np.asarray(signal)

    min_signal = np.min(signal)
    max_signal = np.max(signal)

    if max_signal == min_signal:
        return np.zeros_like(signal) if min_val == 0 else np.full_like(signal, min_val)

    normalized_signal = (signal - min_signal) / (max_signal - min_signal)
    normalized_signal = normalized_signal * (max_val - min_val) + min_val

    return normalized_signal

def set_nan_to_zero(sig):
    zero_segment = np.nan_to_num(sig, nan=0.0)
    return zero_segment


# Setting parameters
database_name = 'mimic3wdb'
required_sigs = ['II'] # we select the longest lead: 'II'
shortest_minutes = 5
req_seg_duration = shortest_minutes*60
save_path = "D:/code_zongheng/foundation_model/saved_data/ecg_segments" # Todo
slide_segment_time = 30 # 30 seconds window size
nan_limit = 0.2
random.seed(42)
max_records_to_load = 1
original_fs = 125 # mimic3wfdb ecg fs
target_fs = 40 # downsample fs

# Get the database and records
subjects = wfdb.get_record_list(database_name)
print(f"The '{database_name}' database contains data from {len(subjects)} subjects")
all_records = wfdb.get_record_list(database_name)
random_records = random.sample(all_records, max_records_to_load)
records = [PurePosixPath(record) for record in random_records]
print(f"Loaded {len(records)} random records from the '{database_name}' database.")
print(records[0])

# Select the suitable segments with 'II' lead and time length > f{"shortest_minutes"}
for record in tqdm(records):
    # index record data path
    record_name = record.name
    record_path = posixpath.join(database_name, record.parent, record_name)

    # Skip the empty record
    try:
        record_data = wfdb.rdheader(record_name, pn_dir=record_path, rd_segments=True)
    except FileNotFoundError:
        print(f"Record {record_name} not found, skipping...")
        continue

    # index segments according to the record data path
    segments = record_data.seg_name
    for segment in segments:
        if segment == "~":
              continue
        segment_metadata = wfdb.rdheader(record_name=segment, pn_dir=record_path)

        # Check if the segments include required lead
        sigs_leads = segment_metadata.sig_name
        if not all(x in sigs_leads for x in required_sigs):
            print(f'{sigs_leads} is missing signal of {required_sigs[0]}')
            continue

        # check if the segments is longer than f{shortest_minutes}
        seg_length = segment_metadata.sig_len/(segment_metadata.fs)
        if seg_length < req_seg_duration:
            print(f' (too short at {seg_length/60:.1f} mins)')
            continue

        matching_seg = posixpath.join(record_path, segment) # "mimic3wdb/32/3213671/3213671_0002"
        print(' (met requirements)')

        # segment every signal into 30s slides
        seg_sig = wfdb.rdrecord(segment, pn_dir=record_path)
        sig_no = seg_sig.sig_name.index('II')
        ecg_seg_sig = seg_sig.p_signal[:, sig_no]

        # setting
        fs = seg_sig.fs
        slide_segment_length = slide_segment_time * fs
        slide_segments = []

        # divide into 30 sec, and discard the last slide (<30s)
        for start in range(0, len(ecg_seg_sig) - slide_segment_length + 1, slide_segment_length):
            end = start + slide_segment_length
            slide_segment = ecg_seg_sig[start:end]

            # check if too much nan value
            if is_nan_ratio_exceed(slide_segment, nan_limit):
                print(f"too much missing value, nan ratio is {((np.isnan(slide_segment).sum() / 3750) * 100):.2f}%")
                continue

            # downsample, set nan as zero then normalize
            slide_segment = downsample(slide_segment, original_fs, target_fs)
            slide_segment = set_nan_to_zero(slide_segment)
            slide_segment = min_max_normalize(slide_segment)

            slide_segments.append(slide_segment)

        # save the segments list
        np.save((save_path + '/' + segment), slide_segments)