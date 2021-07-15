## path where all DICOM images are stored
PATH_DICOM = '../MRIs/DICOM/IDs/'
# PATH_DICOM = '../MRIs/TEST1/IDs/'

## csv with image info like sequence type file paths etc.
PATH_IMAGE_INFO = '../data/patient_data/image_info.csv'

## baseline data like stroke etc.
PATH_BASELINE = '../data/patient_data/baseline_data_DWI.csv'

## comparison jpg vs dicom for cleanup
PATH_CLEAN_UP = '../data/patient_data/Data_Import.csv'

## h5py file with image and patient data
PATH_3D_H5 = '../data/imaging_datasets/dicom_3d_128x128x30.h5'
PATH_2D_H5_RAW = '../data/imaging_datasets/dicom_2d_192x192x3_raw.h5'
PATH_2D_H5_CLEAN = '../data/imaging_datasets/dicom_2d_192x192x3_clean.h5'
PATH_JPG_H5 = '../data/imaging_datasets/jpg_2d_192x192.h5'

## image dimension after scaling
IMAGE_DIMENSIONS_3D = (128, 128, 30)
IMAGE_DIMENSIONS_2D = (192, 192)

MAXSHAPE = tuple([None, 192, 192, 3])
