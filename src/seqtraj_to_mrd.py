#%%
import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp
import ismrmrd
import os

PULSEQ_MRD_FLAGS = {
    'NAV': ismrmrd.ACQ_IS_PHASECORR_DATA,
    'REV': ismrmrd.ACQ_IS_REVERSE,
    'REF': ismrmrd.ACQ_IS_PARALLEL_CALIBRATION,
    'IMA': ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING,
    'NOISE': ismrmrd.ACQ_IS_NOISE_MEASUREMENT,
    'PMC': ismrmrd.ACQ_IS_RTFEEDBACK_DATA,
}

def mr0_to_mrd(seq: pp.Sequence, mr0_signal: np.ndarray, mrd_path: str):
    """
    Write the simulated signal of the MR0 simulation and the k-space trajectory of the corresponding Pulseq sequence to an ISMRMRD dataset.

    Parameters
    ----------
    seq : pp.Sequence
        The Pulseq sequence object containing the sequence information.
    mr0_signal : np.ndarray
        The simulated MR0 signal data with shape (num_samples, num_channels).
    mrd_path : str
        The file path where the ISMRMRD dataset will be saved.

    Returns
    -------
    None
    """
    print("### Create MRD Dataset ###")
    if os.path.exists(mrd_path):
        os.remove(mrd_path)
    dataset = ismrmrd.Dataset(mrd_path, create_if_needed=True)
    
    kadc, _, _, _, tadc = seq.calculate_kspace()
    MaxSignalAdcSegment = seq.get_definition('MaxAdcSegmentLength').astype(int)

    adc_ids = [int(items[5]) for _, items in seq.block_events.items() if items[5] != 0]
    labels = seq.evaluate_labels(evolution='adc')
    start = 0
    scan_counter = 1
    for n_acq, adc_id in enumerate(adc_ids):
        num_samples_adc = seq.adc_library.data.get(adc_id)[0].astype(int)
        dwell = seq.adc_library.data.get(adc_id)[1]

        s_acq = mr0_signal[start:start+num_samples_adc, :]
        k_acq = kadc[:, start:start+num_samples_adc]
        t_acq = tadc[start:start+num_samples_adc]
        start += num_samples_adc

        num_channels = s_acq.shape[1]
        num_traj_dim = k_acq.shape[0]

        if num_samples_adc > MaxSignalAdcSegment:
            k_acq = k_acq.reshape(num_traj_dim, -1, MaxSignalAdcSegment)
            t_acq = t_acq.reshape(-1, MaxSignalAdcSegment)
            s_acq = s_acq.reshape(-1, MaxSignalAdcSegment, num_channels)
        else:
            k_acq = k_acq[:, None, :]
            t_acq = t_acq[None, :]
            s_acq = s_acq[None, ...]
            
        num_sets = k_acq.shape[1]
        for n_set in range(num_sets):

            k = k_acq[:, n_set, :]
            s = s_acq[n_set, ...]
            
            acq_labels = {key: int(value[n_acq]) for key, value in labels.items()}

            header = ismrmrd.AcquisitionHeader()
            header.scan_counter = scan_counter
            header.number_of_samples = s.shape[0]
            header.center_sample = s.shape[0] // 2
            header.trajectory_dimensions = num_traj_dim
            header.available_channels = s.shape[1]
            header.active_channels = s.shape[1]
            header.sample_time_us = dwell * 1e6
            header.idx.kspace_encode_step_1 = acq_labels.get('LIN', 0)
            header.idx.kspace_encode_step_2 = acq_labels.get('PAR', 0)
            header.idx.average = acq_labels.get('AVG', 0)
            header.idx.slice = acq_labels.get('SLC', 0)
            header.idx.contrast = acq_labels.get('ECO', 0)
            header.idx.phase = acq_labels.get('PHS', 0)
            header.idx.repetition = acq_labels.get('REP', 0)
            header.idx.set = acq_labels.get('SET', n_set)
            header.idx.segment = acq_labels.get('SEG', 0)
            
            for key, item in PULSEQ_MRD_FLAGS.items():
                label_value = acq_labels.get(key, 0)
                if label_value > 0:
                    header.set_flag(item)
            
            # Assuming header is already defined
            acquisition = ismrmrd.Acquisition()
            acquisition.setHead(header)
            
            # Set the trajectory data
            acquisition.traj[:] = k.T
            acquisition.data[:] = s.T
            
            dataset.append_acquisition(acquisition)

            scan_counter += 1

    # Close the dataset
    dataset.close()
    

def sequence_traj_to_mrd(seq: pp.Sequence, mrd_dataset: ismrmrd.Dataset):
    kadc, _, _, _, tadc = seq.calculate_kspace()
    
    MaxSignalAdcSegment = seq.get_definition('MaxAdcSegmentLength').astype(int)

    adc_ids = [int(items[5]) for _, items in seq.block_events.items() if items[5] != 0]
    labels = seq.evaluate_labels(evolution='adc')
    start = 0
    for n_acq, adc_id in enumerate(adc_ids):
        num_samples_adc = seq.adc_library.data.get(adc_id)[0].astype(int)
        dwell = seq.adc_library.data.get(adc_id)[1]

        k_acq = kadc[:, start:start+num_samples_adc]
        t_acq = tadc[start:start+num_samples_adc]
        start += num_samples_adc

        num_traj_dim = k_acq.shape[0]

        if num_samples_adc > MaxSignalAdcSegment:
            k_acq = k_acq.reshape(num_traj_dim, -1, MaxSignalAdcSegment)
            t_acq = t_acq.reshape(-1, MaxSignalAdcSegment)
        else:
            k_acq = k_acq[:, None, :]
            t_acq = t_acq[None, :]
            
        num_sets = k_acq.shape[1]
        for n_set in range(num_sets):

            k = k_acq[:, n_set, :]
            
            acq_labels = {key: int(value[n_acq]) for key, value in labels.items()}

            header = ismrmrd.AcquisitionHeader()
            header.scan_counter = n_acq+1
            header.number_of_samples = k.shape[1]
            header.center_sample = k.shape[1] // 2
            header.trajectory_dimensions = num_traj_dim
            header.available_channels = 0
            header.active_channels = 0
            header.sample_time_us = dwell * 1e6
            header.idx.kspace_encode_step_1 = acq_labels.get('LIN', 0)
            header.idx.kspace_encode_step_2 = acq_labels.get('PAR', 0)
            header.idx.average = acq_labels.get('AVG', 0)
            header.idx.slice = acq_labels.get('SLC', 0)
            header.idx.contrast = acq_labels.get('ECO', 0)
            header.idx.phase = acq_labels.get('PHS', 0)
            header.idx.repetition = acq_labels.get('REP', 0)
            header.idx.set = acq_labels.get('SET', n_set)
            header.idx.segment = acq_labels.get('SEG', 0)
            
            for key, item in PULSEQ_MRD_FLAGS.items():
                label_value = acq_labels.get(key, 0)
                if label_value > 0:
                    header.set_flag(item)
            
            # Assuming header is already defined
            acquisition = ismrmrd.Acquisition()
            acquisition.setHead(header)
            
            # Set the trajectory data
            acquisition.traj[:] = k.T
            
            mrd_dataset.append_acquisition(acquisition)

# def merge_signal_sequence_mrd(seq: pp.Sequence, signal_path, output_path):
#     # Create MRD dataset
#     print("### Create MRD Dataset ###")
#     dataset = ismrmrd.Dataset(output_path, create_if_needed=True)

#     #%%
#     kadc, _, _, _, tadc = seq.calculate_kspace()
#     signal = np.load(signal_path)
#     MaxSignalAdcSegment = seq.get_definition('MaxAdcSegmentLength').astype(int)

#     adc_ids = [int(items[5]) for _, items in seq.block_events.items() if items[5] != 0]
#     labels = seq.evaluate_labels(evolution='adc')
#     start = 0
#     for n_acq, adc_id in enumerate(adc_ids):
#         num_samples_adc = seq.adc_library.data.get(adc_id)[0].astype(int)
#         dwell = seq.adc_library.data.get(adc_id)[1]

#         s_acq = signal[start:start+num_samples_adc, :]
#         k_acq = kadc[:, start:start+num_samples_adc]
#         t_acq = tadc[start:start+num_samples_adc]
#         start += num_samples_adc

#         num_channels = s_acq.shape[1]
#         num_traj_dim = k_acq.shape[0]

#         if num_samples_adc > MaxSignalAdcSegment:
#             k_acq = k_acq.reshape(num_traj_dim, -1, MaxSignalAdcSegment)
#             t_acq = t_acq.reshape(-1, MaxSignalAdcSegment)
#             s_acq = s_acq.reshape(-1, MaxSignalAdcSegment, num_channels)
#         else:
#             k_acq = k_acq[:, None, :]
#             t_acq = t_acq[None, :]
#             s_acq = s_acq[None, ...]
            
#         num_sets = k_acq.shape[1]
#         for n_set in range(num_sets):

#             k = k_acq[:, n_set, :]
#             s = s_acq[n_set, ...]
            
#             acq_labels = {key: int(value[n_acq]) for key, value in labels.items()}

#             header = ismrmrd.AcquisitionHeader()
#             header.scan_counter = n_acq+1
#             header.number_of_samples = s.shape[0]
#             header.center_sample = s.shape[0] // 2
#             header.trajectory_dimensions = num_traj_dim
#             header.available_channels = s.shape[1]
#             header.active_channels = s.shape[1]
#             header.sample_time_us = dwell * 1e6
#             header.idx.kspace_encode_step_1 = acq_labels.get('LIN', 0)
#             header.idx.kspace_encode_step_2 = acq_labels.get('PAR', 0)
#             header.idx.average = acq_labels.get('AVG', 0)
#             header.idx.slice = acq_labels.get('SLC', 0)
#             header.idx.contrast = acq_labels.get('ECO', 0)
#             header.idx.phase = acq_labels.get('PHS', 0)
#             header.idx.repetition = acq_labels.get('REP', 0)
#             header.idx.set = acq_labels.get('SET', n_set)
#             header.idx.segment = acq_labels.get('SEG', 0)
            
#             for key, item in PULSEQ_MRD_FLAGS.items():
#                 label_value = acq_labels.get(key, 0)
#                 if label_value > 0:
#                     header.set_flag(item)
            
#             # Assuming header is already defined
#             acquisition = ismrmrd.Acquisition()
#             acquisition.setHead(header)
            
#             # Set the trajectory data
#             acquisition.traj[:] = k.T
#             acquisition.data[:] = s.T
            
#             dataset.append_acquisition(acquisition)

#     # Close the dataset
#     dataset.close()
