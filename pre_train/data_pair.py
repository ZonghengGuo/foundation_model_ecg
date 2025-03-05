import numpy as np
import h5py
import os



# saving setting
ecg_segments_path = r"D:\code_zongheng\foundation_model\saved_data\ecg_segments"
qualities_path = r"D:\code_zongheng\foundation_model\saved_data\quality_labels"
save_path = r"D:\code_zongheng\foundation_model\saved_data\pair_segments"
write_batch_size = 1000  # write data numbers every time
chunk_size = 1024
downsample_fs = 40
signal_length = downsample_fs * 30
h5_file = os.path.join(save_path, 'datasets.h5')


# select segments within 5 min(6 segments)
for ecg_segment_name in os.listdir(ecg_segments_path):
    ecg_segment_path = os.path.join(ecg_segments_path, ecg_segment_name)
    segments = np.load(ecg_segment_path)

    quality_path = os.path.join(qualities_path, ecg_segment_name)
    qualities = np.load(quality_path)

# obtain corresponding qualities

    # -----------------------------------------
    # Step 1: Dynamically write data into h5py
    # -----------------------------------------

    # create base for all data and set appropriate chunk size
    with h5py.File(h5_file, 'w') as f:
        # create dataset to store all pairs
        dset_seg1 = f.create_dataset('seg1', shape=(0, signal_length), maxshape=(None, signal_length),
                                     chunks=(chunk_size, signal_length), compression="gzip", dtype='float64')
        dset_seg2 = f.create_dataset('seg2', shape=(0, signal_length), maxshape=(None, signal_length),
                                     chunks=(chunk_size, signal_length), compression="gzip", dtype='float64')
        dset_diff = f.create_dataset('diff', shape=(0,), maxshape=(None,),
                                     chunks=(chunk_size,), compression="gzip", dtype='float64')

        # cache to store the interim paris
        buffer_seg1, buffer_seg2, buffer_diff = [], [], []

        # Get pairs within 5 min, one segments last 0.5 min, so get
        n = len(segments)
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
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

    f.close()

