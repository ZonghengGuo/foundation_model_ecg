import numpy as np
import h5py
import os
import json
from tqdm import tqdm


with open("dataset/preprocessing_setting.json", "r") as f:
    preprocessing_setting = json.load(f)

with open("pre_train/pre_train_setting.json", "r") as f:
    pre_training_setting = json.load(f)

# saving setting
ecg_segments_path = preprocessing_setting["segment_save_path"]
qualities_path = preprocessing_setting["qua_save_path"]

h5_save_path = pre_training_setting["h5_save_path"]
write_batch_size = pre_training_setting["write_batch_size"]  # write data numbers every time
chunk_size = pre_training_setting["chunk_size"]
downsample_fs = preprocessing_setting["target_fs"]
signal_length = downsample_fs * preprocessing_setting["segment_second"]
h5_file = os.path.join(h5_save_path, 'datasets.h5')

# create base for all data and set appropriate chunk size
with h5py.File(h5_file, 'w') as f:
    # create dataset to store all pairs
    dset_seg1 = f.create_dataset('seg1', shape=(0, signal_length), maxshape=(None, signal_length),
                                 chunks=(chunk_size, signal_length), compression="gzip", dtype='float64')
    dset_seg2 = f.create_dataset('seg2', shape=(0, signal_length), maxshape=(None, signal_length),
                                 chunks=(chunk_size, signal_length), compression="gzip", dtype='float64')
    dset_diff = f.create_dataset('diff', shape=(0,), maxshape=(None,),
                                 chunks=(chunk_size,), compression="gzip", dtype='float64')

    # select segments within 5 min(6 segments)
    for ecg_subject_title in os.listdir(ecg_segments_path):
        ecg_subject_title_path = os.path.join(ecg_segments_path, ecg_subject_title)
        for ecg_subject_name in os.listdir(ecg_subject_title_path):
            ecg_subject_path = os.path.join(ecg_subject_title_path, ecg_subject_name)

            # display process bar
            ecg_segments_list = os.listdir(ecg_subject_path)
            for ecg_segments_name in tqdm(ecg_segments_list, desc=f"Processing {ecg_subject_name}"):

                # read segments.npy
                ecg_segments_path = os.path.join(ecg_subject_path, ecg_segments_name)
                segments = np.load(ecg_segments_path)

                # read corresponding quality label
                quality_path = os.path.join(qualities_path, ecg_subject_title, ecg_subject_name, ecg_segments_name)
                qualities = np.load(quality_path)

                # -----------------------------------------
                # Step 1: Dynamically write data into h5py
                # -----------------------------------------

                # cache to store the interim paris
                buffer_seg1, buffer_seg2, buffer_diff = [], [], []

                # Get pairs within 5 min, one segments last 0.5 min, so get
                n = len(segments)
                for i in range(n):
                    for j in range(i + 1, min(i + 10, n)):
                        # if two samples qualities are similar, skip this pair
                        if round(qualities[i], 2) == round(qualities[j], 2):
                            continue

                        # add the 2 sigs and the quality difference to interim lists
                        buffer_seg1.append(segments[i])
                        buffer_seg2.append(segments[j])
                        buffer_diff.append(abs(qualities[i] - qualities[j]))

                        # if reach the limitation of interim lists, add the data to dynamic dataset then clear interim lists
                        if len(buffer_seg1) >= write_batch_size:
                            # adjust the length of dynamic lists
                            current_size = dset_seg1.shape[0]
                            new_size = current_size + len(buffer_seg1)
                            dset_seg1.resize((new_size, signal_length))
                            dset_seg2.resize((new_size, signal_length))
                            dset_diff.resize((new_size,))

                            # add data to dynamic lists from interim lists
                            dset_seg1[current_size: new_size] = np.array(buffer_seg1)
                            dset_seg2[current_size: new_size] = np.array(buffer_seg2)
                            dset_diff[current_size: new_size] = np.array(buffer_diff)

                            # clear cache(interim lists)
                            buffer_seg1, buffer_seg2, buffer_diff = [], [], []
                            print(f"Reach the interim limitation of {write_batch_size}. Save interim data into dataset, with size {new_size} currently")

                # left some data finally and it didn't reach the write_batch_size, add them to dynamic dataset
                if buffer_seg1:
                    current_size = dset_seg1.shape[0]
                    new_size = current_size + len(buffer_seg1)

                    dset_seg1.resize((new_size, signal_length))
                    dset_seg2.resize((new_size, signal_length))
                    dset_diff.resize((new_size,))

                    dset_seg1[current_size:new_size] = np.array(buffer_seg1)
                    dset_seg2[current_size:new_size] = np.array(buffer_seg2)
                    dset_diff[current_size:new_size] = np.array(buffer_diff)

                    print(f"Add the left data into dataset, with size {new_size} currently")

    # -------------------------------
    # Step 2: build an index of diff
    # -------------------------------
    diff_data = dset_diff[:]
    sorted_indices = np.argsort(diff_data)  # Ascending order
    f.create_dataset('sorted_indices', data=sorted_indices, compression="gzip")
    print("Finish writing dataset into h5 file")

f.close()

