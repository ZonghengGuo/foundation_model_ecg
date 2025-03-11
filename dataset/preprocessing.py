from pathlib import PurePosixPath
import wfdb
from tqdm import tqdm
import posixpath
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
from rrSQI import rrSQI
from QRS import peak_detection
import requests

# Setting
with open("dataset/preprocessing_setting.json", "r") as f:
    setting = json.load(f)

# Setting parameters
database_name = setting["database"]
required_sigs = setting["required_sigs"]  # we select the longest lead: 'II'
shortest_minutes = setting["shortest_minutes"]
req_seg_duration = shortest_minutes * 60
seg_save_path = setting["segment_save_path"]
qua_save_path = setting["qua_save_path"]
slide_segment_time = setting["segment_second"]  # ~ seconds window size
nan_limit = setting["nan_limit"]
original_fs = setting["original_fs"]  # mimic3wfdb ecg fs
target_fs = setting["target_fs"]  # downsample fs


# Preprocessing function
def is_nan_ratio_exceed(sig, threshold):
    nan_ratio = np.isnan(sig).sum() / 3750  # count nan value
    return nan_ratio > threshold  # tell if reach the limit


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


def is_constant_signal(slide_segment):
    return np.all(slide_segment == slide_segment[0])


# Select the suitable segments with 'II' lead and time length > f{"shortest_minutes"}
# index record data path
def process_record(record):
    record_name = record.name
    record_path = posixpath.join(database_name, record.parent, record_name)

    # Skip the empty record or skip network disconnection or anyother read error
    try:
        record_data = wfdb.rdheader(record_name, pn_dir=record_path, rd_segments=True)
    except FileNotFoundError:
        print(f"Record {record_name} not found, skipping...")
        return
    except requests.exceptions.RequestException as e:
        print(f"Network error while accessing {record_name}, skipping... Error: {e}")
        return
    except Exception as e:
        print(f"Error processing {record_name}, skipping... Error: {e}")
        return

    # index segments according to the record data path
    segments = record_data.seg_name
    for segment in segments:
        if segment == "~":
            continue
        segment_metadata = wfdb.rdheader(record_name=segment, pn_dir=record_path)

        print('-------------------')
        print(f"Start preprocessing {record_path}/{segment}")

        # Check if the segments include required lead
        sigs_leads = segment_metadata.sig_name
        if not all(x in sigs_leads for x in required_sigs):
            print(f'{sigs_leads} is missing signal of {required_sigs[0]}')
            continue

        # check if the segments is longer than f{shortest_minutes}
        seg_length = segment_metadata.sig_len / (segment_metadata.fs)
        if seg_length < req_seg_duration:
            print(f' (too short at {seg_length / 60:.1f} mins)')
            continue

        matching_seg = posixpath.join(record_path, segment)  # "mimic3wdb/32/3213671/3213671_0002"
        print(' (met requirements)')

        # segment every signal into 30s slides
        seg_sig = wfdb.rdrecord(segment, pn_dir=record_path)
        sig_no = seg_sig.sig_name.index('II')
        ecg_seg_sig = seg_sig.p_signal[:, sig_no]

        # setting
        slide_segment_length = slide_segment_time * original_fs
        slide_segments = []
        qua_labels = []

        # divide into 30 sec, and discard the last slide (<30s)
        for i, start in enumerate(range(0, len(ecg_seg_sig) - slide_segment_length + 1, slide_segment_length)):
            end = start + slide_segment_length
            slide_segment = ecg_seg_sig[start:end]

            # check if too much nan value
            if is_nan_ratio_exceed(slide_segment, nan_limit):
                print(f"too much missing value, nan ratio is {((np.isnan(slide_segment).sum() / 3750) * 100):.2f}%")
                continue

            # check if the signal is stable
            if is_constant_signal(slide_segment):
                print(f"the sequence is stable, not a signal")
                continue

            # set nan as zero then normalize
            slide_segment = set_nan_to_zero(slide_segment)
            slide_segment = min_max_normalize(slide_segment)
            print("set nan value to zero and normalize signal")

            # quality assessment
            try:
                peaks = peak_detection(slide_segment, original_fs)
                print("find peaks")
            except ValueError as e:
                print(f"Warning: {e}, skipping this segment.")
                continue

            _, _, qua = rrSQI(slide_segment, peaks, original_fs)

            qua_labels.append(qua)
            print(f"The quality of ECG segment in {str(record.name)}.npy_{i} is: {qua}")

            # downsample to 40Hz
            slide_segment = downsample(slide_segment, original_fs, target_fs)

            slide_segments.append(slide_segment)

        # save the segments and qualities list
        segment_save_path = seg_save_path + '/' + str(record.parent) + '/' + str(record.name) + '/' + segment
        os.makedirs(os.path.dirname(segment_save_path), exist_ok=True)

        quality_save_path = qua_save_path + '/' + str(record.parent) + '/' + str(record.name) + '/' + segment
        os.makedirs(os.path.dirname(quality_save_path), exist_ok=True)

        np.save(segment_save_path, slide_segments)
        try:
            np.save(quality_save_path, qua_labels)
        except ValueError as e:
            print(f"Skip wrong dimension of qua_labels in {record_path}/{segment}")
            continue

        print(f"save segments into: {segment_save_path}.npy and qualities into {quality_save_path}.npy")


async def async_process_records(records):
    print(f"Using {os.cpu_count()} cpu cores for synchronous programming and multi-thread pool processing")
    with ProcessPoolExecutor(max_workers=os.cpu_count() * 5) as pool:
        loop = asyncio.get_event_loop()
        tasks = []
        with tqdm(total=len(records), desc="Processing records") as pbar:
            for record in records:
                task = loop.run_in_executor(pool, process_record, record)
                task.add_done_callback(lambda _: pbar.update())
                tasks.append(task)

            await asyncio.gather(*tasks)


async def main(records):
    await async_process_records(records)


if __name__ == '__main__':
    min_records_to_load = 0
    max_records_to_load = 100
    # total: 67830

    # Get the database and records
    subjects = wfdb.get_record_list(database_name)
    print(f"The '{database_name}' database contains data from {len(subjects)} subjects")
    all_records = wfdb.get_record_list(database_name)
    sequent_records = all_records[min_records_to_load: max_records_to_load]
    records = [PurePosixPath(record) for record in sequent_records]
    print(f"Loaded {len(records)} records from the '{database_name}' database.")

    # Synchronous programming and multi-thread pool processing
    asyncio.run(main(records))
