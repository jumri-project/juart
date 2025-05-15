"""
Author:     Jörg Felder
Copyright:  2023, Forschungszentrum Jülich 
"""

import fnmatch
import os
import warnings

import numpy as np
import pandas as pd

from dicom_parser import Image

def load_dcm_dir(DataDir, FilePattern, SeriesDescPattern='ALL', HeaderInfo=[]):

    """
    Travels through directory and loads all dicom files with a specified Series Description. 
    The DICOM images as well as selected header information are stored in a panda dataframe. 

    Parameters
    ----------

    DataDir:            String containing the DICOM directory
    FilePattern:        String containing file name pattern of the DICOM files (e.g. '*.IMA')
    SeriesDescPattern:  String pattern containg the DICOM series description to be loaded. 
                        If equal to 'ALL', all dicom files underneat the DataDir folder will 
                        be loaded.
    HeaderInfo:         Desired fields from the header files that should be copied into the 
                        data frame.


    Returns
    -------
    df:                 Dataframe containing the requested image and header information

    Notes
    -----

    """

    df = pd.DataFrame()

    for parentDir, dirnames, filenames in os.walk(DataDir):
        for filename in fnmatch.filter(filenames, FilePattern):

            # read images 
            image = Image(os.path.join(parentDir,filename))

            # Read series description
            strSeriesDesc = image.header.get('SeriesDescription')

            # check if we have the desired series
            if (fnmatch.filter([strSeriesDesc], SeriesDescPattern)) or (SeriesDescPattern=='ALL'):

                # store image data in temporary data frame
                df_data = pd.DataFrame({'data': [image.data]})

                # add selected header information
                for i in range(len(HeaderInfo)):                    
                    df_index = pd.DataFrame({str(HeaderInfo[i]):[image.header.get(HeaderInfo[i])]})
                    df_data = df_data.merge(df_index, how='left', left_index=True, right_index=True )
                    # print(image.header.get(HeaderInfo[i]))
                
                # add single image to dataframe
                df = pd.concat([df, df_data], ignore_index=True)
    
    print('Found {} matching DICOM files'.format(len(df)))

    return df




def extract_coil_info(df):

    """
    Extracts information on the used transmit and receive coil for each image and add this
    information to the suppplied dataframe

    Parameters
    ----------

    df:                 Data frame with the dicom information retrieved via load_dcm_dir
    

    Returns
    -------
    df:                 Dataframe containing augmented with the coil information

    Notes
    -----

    """

    # check if protocol exists in the dataframe
    if str((0x0029, 0x1020)) in df:

        # get number of transmit elements
        N_TX_Coils = len(df.loc[0][str((0x0029, 0x1020))]['MrPhoenixProtocol']['value']['sCoilSelectMeas']['aTxCoilSelectData'][0]['asList'])

        # add new colum to data frame
        df['pTX_Channel_Coeffs'] = None
        
        # iterate over rows of data frame
        for i in range(len(df)):

            # Allocate empty vector (needs to bre column vector!)
            pTX_channel_list = np.zeros((1,N_TX_Coils), dtype=complex)

            # get TX channel shim setting
            pTX_channel_info = df.loc[i][str((0x0029, 0x1020))]['MrPhoenixProtocol']['value']['sTXSPEC']['aTxScaleFactor']

            # Assign protocol B1+ shim coefficients to readable numpy array
            # Note: In single channel system len(pTX_channel_info) may be larger than the number of TX channels
            # TODO: Is this due to a higher number of slices acquired?
            for j in range( min(len(pTX_channel_info), N_TX_Coils) ):
            # for j in range(len(pTX_channel_info)):

                # Assign element wise
                if( pTX_channel_info[j] != None):
                    ShimValue = complex(0)
                    if 'dRe' in pTX_channel_info[j]:
                       ShimValue += pTX_channel_info[j]['dRe']
                    if 'dIm' in  pTX_channel_info[j]:
                        ShimValue += 1j*pTX_channel_info[j]['dIm']
                    pTX_channel_list[0,j] = ShimValue
            
            df.loc[i,('pTX_Channel_Coeffs')] = [pTX_channel_list]

        # Print information message
        print('TX Coil information has been added for {} entries'.format(len(df)))

    else:
        #print message that protocol information is not stored in the dataframe
        warnings.warn('Could not get DICOM tag (0x0029, 0x1020). Ignoring TX element extraction.',UserWarning)

    # check if receive element information exists
    if str((0x0051,0x100F)) in df:
        
        # add new column to data frame
        df['Active_Rx_Elements'] = None

        # iterate over rows of data frame
        for i in range(len(df)):
            df.loc[i,('Active_Rx_Elements')] = df.loc[i][str((0x0051,0x100F))]
    
        # Print information message
        print('RX Coil information has been added for {} entries'.format(len(df)))

    else:
        #print message that preceive element information is not stored in the dataframe
        warnings.warn('Could not get DICOM tag (0x0051,0x100F). Ignoring RX element extraction',UserWarning)

    # return modified data frame
    return df


def convert_df_to_np_using_coilindices( df ):
    """
    Create a numpy array of the data from df in column DataColumnName. 

    Parameters
    ----------

    df:                 Data frame with the dicom information retrieved via load_dcm_dir and 
                        coil information extracted via extract_coil_info

    Returns
    -------
    nparray:            Numpy array of dimensions
                            (NumberSequences, NTXCoils, NRXCoils, MatrixSize1, MatrixSize2)

    Notes
    -----

    """

    # check reuired columns are present
    if 'pTX_Channel_Coeffs' not in df:
        raise IndexError('Column pTX_Channel_Coeffs not present in data frame.')
    if str((0x0051,0x100F)) not in df:
        raise IndexError('Column ' +  str((0x0051,0x100F)) + ' not present in data frame.')
    
    
    # get number of TX elements
    NTxElements = df['pTX_Channel_Coeffs'][0][0].shape[1]
    print('Found {} Tx eleemnts in the array.'.format(NTxElements))

    # get number of RX elements
    TempRxElemetNames = df[str((0x0051,0x100F))].unique()
    if( len(TempRxElemetNames)==1 and (TempRxElemetNames == 'AC')):
    # Find if only combined images are present
        RxElemetNames = TempRxElemetNames
        NRxElements = 1
        print('Combined images only.')
    else: 
        # Remove AC entry which is for combining all elements
        RxElemetNames = np.delete(TempRxElemetNames, np.where(TempRxElemetNames == 'AC'))
        RxElemetNames = np.sort(RxElemetNames)
        NRxElements = len(RxElemetNames)
        print('Found {} Rx eleemnts in the array.'.format(NRxElements))
    
        
    # get matrix size of image
    MatrixSize1 = df.loc[0]['data'].shape[0]
    MatrixSize2 = df.loc[0]['data'].shape[1]
    print('Matrix size of images is {} x {}.'.format(MatrixSize1, MatrixSize2))

    # created temporary dataframe only containing single element RX
    Temp_df = df[df[str((0x0051,0x100F))].isin(RxElemetNames)]
    Temp_df.reset_index()
    NSequences = len(Temp_df)
    print('Found {} images per Rx element.'.format(NSequences))

    # allocate numpy array
    np_mag_data = np.zeros((NSequences, NTxElements, NRxElements, MatrixSize1, MatrixSize2), dtype=float)
    np_phase_data = np.zeros((NSequences, NTxElements, NRxElements, MatrixSize1, MatrixSize2), dtype=float)

    # iterate over data frame
    for i in range(len(Temp_df)):

        # Magnitude or phase image
        ImageMagPhaseType = Temp_df.iloc[i]['ImageType'][2]

        # Tx element index
        idx = np.where(Temp_df.iloc[i]['pTX_Channel_Coeffs'][0][0] != 0.0)
        TxElementIdx = idx[0][0]
        
        # Rx element index
        idx = np.where(RxElemetNames == Temp_df.iloc[i][str((0x0051,0x100F))])
        RxElementIdx = idx[0][0]

        if( (len(Temp_df) == 1) and (ImageMagPhaseType == 'P')):
            # Some sequences store phase only images (e.g. F0 map in AFI). These will returned in the magnitude
            # as otherwise we do not see the results.
            print('Found phase only image -> casting to magnitude data.')
            np_mag_data[i, TxElementIdx, RxElementIdx, ...] = Temp_df.iloc[i]['data']
        elif( ImageMagPhaseType == 'M' ):
            np_mag_data[i, TxElementIdx, RxElementIdx, ...] = Temp_df.iloc[i]['data']
        elif( ImageMagPhaseType == 'P' ):
            DicomPhase = Temp_df.iloc[i]['data']

            # convert phase data to radians
            # 
            # The Dicom data range is from 0 to 2^Nbits-1 (0 to 4095 for 12 bits). This 
            # is linearly transformed: (0028,1052) RescaleIntercept and (0028,1053) RescaleSlope. 
            # As an example for RescaleIntercept = -4096 and RescaleSlope = 2 the data range 
            # become -4096 .. 4095. This range needs to be mapped to [-pi .. pi).

            NBits = Temp_df.iloc[i]['BitsStored']
            RescaleIntercept = Temp_df.iloc[i]['RescaleIntercept']
            RescaleSlope = Temp_df.iloc[i][str((0x0028,0x1053))]

            MinVal = 0*RescaleSlope + RescaleIntercept
            MaxVal = (2**NBits-1)*RescaleSlope + RescaleIntercept
            
            RadianPhase = DicomPhase / (MaxVal-MinVal) * 2*np.pi  

            np_phase_data[i, TxElementIdx, RxElementIdx, ...] = RadianPhase


    return np_mag_data * np.exp(1j*np_phase_data)