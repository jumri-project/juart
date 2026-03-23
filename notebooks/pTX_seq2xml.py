# seq2xml.py : converts Pulseq (.seq) files into JEMRIS (.xml) sequences
# Gehua Tong
# March 2020

from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
# import xml.etree.ElementTree as ET                            ## JF: modified to get some nicer xml format with pretty print
try:
    import lxml.etree as ET
    print("running with lxml.etree")
except ImportError:
    import xml.etree.ElementTree as ET
    print("running with Python's xml.etree.ElementTree")

import h5py
import os
import numpy as np
from math import pi

# Links:
# https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30601        # Paper from T. Roos
# https://github.com/Roosted7/ptx-pulseq                            # github repo refernced in the paper
# https://openmr.nl/                                                # Some examples for the Philips scanner


# Notes
# This is for generating an .xml file for input into JEMRIS simulator, from a Pulseq .seq file
# The opposite philosophies make the .xml encoding suboptimal for storage
# (because seq files consists of flattened-out Blocks while the JEMRIS format minimizes repetition using loops
#  and consists of many cross-referencing of parameters)

# Consider: for virtual scanner, have scripts that generate .xml and .seq at the same time (looped vs. flattened)
# (but isn't JEMRIS already doing that? JEMRIS does have an "output to pulseq" functionality)
# though then, having a Python interface instead of a MATLAB one is helpful in the open-source aspect


# Unit conversion constants (comment with units before & after)
rf_const = 2 * pi / 1000  # from Pulseq[Hz]=[1/s] to JEMRIS[rad/ms] rf magnitude conversion constant
g_const = 2 * pi / 1e6  # from Pulseq [Hz/m] to JEMRIS [(rad/ms)/mm] gradient conversion constant
slew_const = g_const / 1000  # from Pulseq [Hz/(m*s)] to JEMRIS [(rad/ms)/(mm*ms)]
ga_const = 2 * pi / 1000  # from Pulseq[1/m] to JEMRIS [2*pi/mm] gradient area conversion constant
sec2ms = 1000  # time conversion constant
m2mm = 1000 # convert m to mm
rad2deg = 180/pi
freq_const = 2 * pi / 1000 # From Hz to rad/ms

def ptx_seq2xml(seq, seq_name, out_folder):
    """
    # Takes a Pulseq sequence and converts it into .xml format for JEMRIS
    # All RF and gradient shapes are stored as .h5 files

    Inputs
    ------
    seq : pypulseq.Sequence.sequence.Sequence
    seq_name : name of output .xml file
    out_folder : str
        Path to output folder for .xml file

    Returns
    -------
    seq_tree : xml.etree.ElementTree
        Tree object used for generating the sequence .xml file
    seq_path : str
        Path to stored .xml sequence
    """

    # Parameters is the root of the xml
    root = ET.Element("Parameters")
    # Add gradient limits (seem to be the only parameters shared by both formats)
    # TODO check units
    root.set("GradMaxAmpl", str(seq.system.max_grad*g_const))
    root.set("GradSlewRate", str(seq.system.max_slew*slew_const))
    root.set("FOVx", str(seq.get_definition('FOV')[0]*m2mm))        # Added: JF
    root.set("FOVy", str(seq.get_definition('FOV')[1]*m2mm))        # Added: JF
    root.set("FOVz", str(seq.get_definition('FOV')[2]*m2mm))        # Added: JF
    

    # ConcatSequence is the element for the sequence itself;
    # Allows addition of multiple AtomicSequence
    C0 = ET.SubElement(root, "ConcatSequence")

    # Use helper functions to save all RF and only arbitrary gradient info
    rf_shapes_path_dict, NRFChannels = save_rf_library_info(seq, out_folder)
    grad_shapes_path_dict = save_grad_library_info(seq, out_folder)
    #print(grad_shapes_path_dict)
    #///////////////////////////////////////////////////////////////////////////


    rf_name_ind = 1
    grad_name_ind = 1
    delay_name_ind = 1
    adc_name_ind = 1


    ##### Main loop! #####
    # Go through all blocks and add in events; each block is one AtomicSequence
    for block_ind in range(1,len(seq.block_events)+1):
        blk = seq.get_block(block_ind).__dict__
        exists_adc = 'adc' in blk.keys()
        adc_already_added = False
        # Note: "EmptyPulse" class seems to allow variably spaced ADC sampling
        # Distinguish between delay and others
        # Question: in pulseq, does delay happen together with other events?
        #           (for now, we assume delay always happens by itself)
        # About name of atomic sequences: not adding names for now;
        # (Likely it will cause no problems because there is no cross-referencing)
        C_block = ET.SubElement(C0, "ATOMICSEQUENCE")
        C_block.set("Name", f'C{block_ind}')
        for key in blk.keys():
            # Case of RF pulse
            if key == 'rf':
                rf = blk['rf']
                if(rf != None):  
                    for ch in range(NRFChannels):                                                   # Added: JF
                        rf_atom = ET.SubElement(C_block, "EXTERNALRFPULSE")

                        rf_atom.set("Name", f'R{rf_name_ind}')
                        rf_name_ind += 1

                        rf_atom.set("Channel", str(ch))
                        rf_atom.set("InitialDelay", str(rf.delay*sec2ms))
                        rf_atom.set("InitialPhase", str(rf.phase_offset*rad2deg))
                        rf_atom.set("Frequency", str(rf.freq_offset*freq_const))
                        # Find ID of this rf event
                        rf_id = seq.block_events[block_ind][1]
                        
                        if NRFChannels > 1:
                            # add channel number to file name
                            RFFormFileName = os.path.splitext(rf_shapes_path_dict[rf_id])[0] + '_ch' + str(ch) + '.h5'
                        else:
                            RFFormFileName = rf_shapes_path_dict[rf_id]
                        rf_atom.set("Filename", RFFormFileName)

                        rf_atom.set("Scale","1")
                        rf_atom.set("Interpolate", "0") # Do interpolate

            gnames_map = {'gx':2, 'gy':3, 'gz':4}
            if key in ['gx', 'gy', 'gz']:
                g = blk[key]
                if(g != None):                                                      # Added: JF
                    if g.type == "trap":
                        if g.amplitude != 0:
                            g_atom = ET.SubElement(C_block, "TRAPGRADPULSE")
                            g_atom.set("Name", f'G{grad_name_ind}')
                            grad_name_ind += 1
                            if g.flat_time > 0:
                                # 1. Case where flat_time is nonzero
                                # Second, fix FlatTopArea and FlatTopTime
                                g_atom.set("FlatTopArea", str(g.flat_area*ga_const))
                                g_atom.set("FlatTopTime", str(g.flat_time*sec2ms))

                                # Last, set axis and delay
                            else:
                                # 2. Case of triangular pulse (i.e. no flat part)
                                g_atom.set("MaxAmpl", str(np.absolute(g.amplitude*g_const))) # limit amplitude
                                g_atom.set("Area", str(0.5*(g.rise_time + g.fall_time)*g.amplitude*ga_const))

                            # Third, limit duration by limiting slew rate
                            # g_atom.set("SlewRate", str((g.amplitude * g_const) / (g.fall_time * sec2ms)))

                            # Alternatively, set rise and fall time explicitley, since these are given
                            g_atom.set("RampUpTime", str(g.rise_time*sec2ms))
                            g_atom.set("RampDwonTime", str(g.fall_time*sec2ms))
                            
                            g_atom.set("Asymmetric", str(g.fall_time / g.rise_time))
                            g_atom.set("Axis", key.upper())
                            g_atom.set("InitialDelay", str(g.delay*sec2ms))


                        # Add ADC if it exists and then "mark as complete"
                    elif g.type == "grad":
                        # Set arbitrary grad parameters
                        # Need to load h5 file again, just like in RF
                        g_id = seq.block_events[block_ind][gnames_map[key]]
                        g_atom = ET.SubElement(C_block, "EXTERNALGRADPULSE")
                        g_atom.set("Name", f'G{grad_name_ind}')
                        grad_name_ind += 1

                        g_atom.set("Axis", key.upper())
                        g_atom.set("Filename", grad_shapes_path_dict[g_id])
                        g_atom.set("Scale","1")
                        g_atom.set("Interpolate","0")
                        g_atom.set("InitialDelay",str(g.delay*sec2ms))

                    else:
                        print(f'Gradient type "{g.type}" indicated')
                        raise ValueError("Gradient's type should be either trap or grad")

                if exists_adc and not adc_already_added:
                    adc = blk['adc']
                    if(adc != None):                                                # Added: JF
                        dwell = adc.dwell*sec2ms
                        adc_delay = adc.delay*sec2ms
                        Nro = adc.num_samples
                        phase_offset = adc.phase_offset*rad2deg

                        gzero_adc = ET.SubElement(C_block, "TRAPGRADPULSE")
                        gzero_adc.set("Name", f'S{adc_name_ind}')
                        adc_name_ind += 1

                        gzero_adc.set("ADCs", str(Nro))
                        gzero_adc.set("FlatTopTime", str(dwell*Nro))
                        gzero_adc.set("FlatTopArea","0")
                        gzero_adc.set("InitialDelay", str(adc_delay))
                        gzero_adc.set("Axis", "GX")  # Just set to X; seems not to work otherwise
                        gzero_adc.set("InitialPhase", str(phase_offset))
                        gzero_adc.set("ADCFlag", str(2))  # ADCFlag=2 means "this gradient is for ADC"

                        adc_already_added = True
                        # Now, it always attach ADC to the first gradient found among keys()
                        # This might be tricky
                        # suggestion 1: check the duration of gradient?
                        # suggestion 2: just do any gradient and hope it works
                        # suggestion 3: read the JEMRIS documentation/try on GUI

            if key == 'block_duration':
                delay_dur = blk['block_duration']
                if(delay_dur != 0.0):
                    delay_atom = ET.SubElement(C0, "DELAYATOMICSEQUENCE")

                    delay_atom.set("Name",f'D{delay_name_ind}')
                    delay_name_ind += 1

                    delay_atom.set("Delay",str(delay_dur*sec2ms))
                    delay_atom.set("DelayType","B2E")

                    # If the block does not only contain a delay, it must start with the previous atomic sequence
                    if (blk['rf'] != None) or \
                       (blk['gx'] != None) or \
                       (blk['gy'] != None) or \
                       (blk['gz'] != None) or \
                       (blk['adc'] != None):
                        delay_atom.set("StartSeq",f'C{block_ind}')

    # Output it!
    seq_tree = ET.ElementTree(root)

    seq_path = out_folder + '/' + seq_name + '.xml'
    seq_tree.write(seq_path,  pretty_print=True)

    return seq_tree, seq_path

def save_rf_library_info(seq, out_folder):
    """
    Helper function that stores distinct RF waveforms for seq2xml
    """
    # RF library
    rf_shapes_path_dict = {}
    for rf_id in list(seq.rf_library.data.keys()): # for each RF ID
        # JEMRIS wants:
        # "Filename":  A HDF5-file with a single dataset "extpulse"
        # of size N x 3 where the 1st column holds the time points,
        # and 2nd and 3rd column hold amplitudes and phases, respectively.
        # Phase units should be radians.
        # Time is assumed to increase and start at zero.
        # The last time point defines the length of the pulse.
        # file_path_partial = f'rf_{int(rf_id)}.h5'
        # file_path_full = out_folder + '/' + file_path_partial
        # De-compress using inbuilt PyPulseq method
        rf = seq.rf_from_lib_data(seq.rf_library.data[rf_id])
        # Only extract time, magnitude, and phase
        # We leave initial phase and freq offset to the main conversion loop)
        times = rf.t
        magnitude = np.absolute(rf.signal)
        phase = np.angle(rf.signal)

        # Get number of channels
        # If length of times if greater than the number of unique time pints, 
        # more than one channel exists
        NChannels = len(times) // len(np.unique(times))

        for ch in range(NChannels):
            file_path_partial = f'rf_{int(rf_id)}_ch{int(ch)}.h5'
            file_path_full = out_folder + '/' + file_path_partial

            times_ch = times[ch*len(times)//NChannels:(ch+1)*len(times)//NChannels]
            magnitude_ch = np.absolute(rf.signal[ch*len(times_ch):(ch+1)*len(times_ch)])
            phase_ch = np.angle(rf.signal[ch*len(times_ch):(ch+1)*len(times_ch)])
            
            N = len(times_ch)
            # Create file
            f = h5py.File(file_path_full, 'a')
            if "extpulse" in f.keys():
                del f["extpulse"]

            #f.create_dataset("extpulse", (N,3), dtype='f')
            f.create_dataset("extpulse",(3,N),dtype='f')

            times_ch = times_ch - times_ch[0]
            f["extpulse"][0,:] = times_ch*sec2ms #*sec2ms
            f["extpulse"][1,:] = magnitude_ch*rf_const
            f["extpulse"][2,:] = phase_ch        #"Phase should be radians"

            f.close()

        # we return the file name without channel number for sake of compatibility
        file_path_partial = f'rf_{int(rf_id)}.h5'
        rf_shapes_path_dict[rf_id] = file_path_partial

    return rf_shapes_path_dict, NChannels


# Helper function
def save_grad_library_info(seq, out_folder):
    """
    Helper function that stores distinct gradients for seq2xml
    """

    #file_paths = [out_folder + f'grad_{int(grad_id)}.h5' for grad_id in range(1,N_grad_id+1)]
    grad_shapes_path_dict = {}
    processed_g_inds = []

    for nb in range(1,len(seq.block_events)+1):
        gx_ind, gy_ind, gz_ind = seq.block_events[nb][2:5]
        for axis_ind, g_ind in enumerate([gx_ind, gy_ind, gz_ind]):
            # Only save a gradient file if ...(a) it has non-zero index
            #                                 (b) it is type 'grad', not 'trap'
            #      s                       and (c) its index has not been processed
            if g_ind != 0 and len(seq.grad_library.data[g_ind]) == 3 \
                and g_ind not in processed_g_inds:
                print(f'Adding Gradient Number {g_ind}')
                this_block = seq.get_block(nb)
                file_path_partial = f'grad_{int(g_ind)}.h5'
                file_path_full = out_folder + '/' + file_path_partial

                #TODO make it work for x/y/z
                if axis_ind == 0:
                    t_points = this_block.gx.t
                    g_shape = this_block.gx.waveform
                elif axis_ind == 1:
                    t_points = this_block.gy.t
                    g_shape = this_block.gy.waveform
                elif axis_ind == 2:
                    t_points = this_block.gz.t
                    g_shape = this_block.gz.waveform

                N = len(t_points)
                # Create file
                f = h5py.File(file_path_full, 'a')
                if "extpulse" in f.keys():
                    del f["extpulse"]
                f.create_dataset("extpulse", (2,N), dtype='f')
                f["extpulse"][0,:] = t_points * sec2ms
                f["extpulse"][1,:] = g_shape * g_const
                f.close()
                grad_shapes_path_dict[g_ind] = file_path_partial
                processed_g_inds.append(g_ind)

    return grad_shapes_path_dict

if __name__ == '__main__':
    print('')
    seq = Sequence()
    seq.read('notebooks/gre2d_pTxSingleChan_8Tx.seq')
    ptx_seq2xml(seq, seq_name='gre2d_pTxSingleChan_8Tx', out_folder='notebooks/gre2d_pTxSingleChan_8Tx')
#    seq.read('seq_files/spgr_gspoil_N16_Ns1_TE5ms_TR10ms_FA30deg.seq')
    #seq.read('benchmark_seq2xml/gre_jemris.seq')
#    seq.read('try_seq2xml/spgr_gspoil_N15_Ns1_TE5ms_TR10ms_FA30deg.seq')
    #seq.read('orc_test/seq_2020-02-26_ORC_54_9_384_1.seq')
    #stree = seq2xml(seq, seq_name="ORC-Marina", out_folder='orc_test')


