import os
import pandas as pd
import pydicom
import h5py
import numpy as np
from scipy import ndimage
from functions.config import *

def rename_folders_to_3char(path):
    subfolders = list(os.walk(path))[0][1]
    for folder in subfolders:
        folder_3c = format(int(folder), '03d')
        old_name = os.path.join(PATH_DICOM, folder)
        new_name = os.path.join(PATH_DICOM, folder_3c)
        os.rename(old_name, new_name)

def get_files_in_directory(dir):
    file_list = []

    for paths, dirnames, filenames in os.walk(dir, topdown = True):
        for file in filenames:
            if file != 'DICOMDIR' and file[-2:] != 'h5':
                file_list.append(os.path.join(paths,file))
    file_list.sort()
    return file_list


def create_file_list(image_dir, image_info_dir):

    if os.path.exists(image_info_dir):
        os.remove(image_info_dir)

    listFilesDICOM = get_files_in_directory(image_dir)
    batch_size = 500

    for i in range(0, len(listFilesDICOM), batch_size):
        print('reading files {}:{} of {}'.format(i, i+batch_size-1, len(listFilesDICOM)))

        batch = listFilesDICOM[i:i+batch_size]
        batch_df = pd.DataFrame()

        for index, file in enumerate(batch):
            DICOM_file = pydicom.dcmread(file, force=True)

            try:
                batch_df.loc[index, 'patient'] = file[18:21]
                batch_df.loc[index, 'description'] = DICOM_file.SeriesDescription ## 0008 103e
                batch_df.loc[index, 'filepath'] = file
            except Exception as e:
                print('file {} cannot be loaded.'.format(file))
                print(e)
            try:
                batch_df.loc[index, 'sequence'] = DICOM_file.SequenceName ## 0018 0024
            except Exception as e:
                print('file {}'.format(file))
                batch_df.loc[index, 'sequence'] = "unknown"
                print(e)

        with open(image_info_dir, 'a') as f:
            if i == 0:
                batch_df.to_csv(f, header = True, index = False)
            else:
                batch_df.to_csv(f, header = False, index = False)

    print('File has been saved here: {}'.format(image_info_dir))

def load_images_3D(file_list):

    ## loads DICOM images from file_list and creates a 3D array with pixel data
    ref_file = pydicom.dcmread(file_list[0], force=True)
    image_dims_3d = (int(ref_file.Rows), int(ref_file.Columns), len(file_list))
    # print("Original image size: {}".format(image_dims_3d))

    origArray = np.zeros(image_dims_3d, dtype=ref_file.pixel_array.dtype)

    # loop through all the DICOM files
    removed = 0
    file_list.sort()
    for filename in file_list:
        # read the file
        ds = pydicom.dcmread(filename, force=True)
        # store the raw image data

        if(ds.Rows != ref_file.Rows):
            removed  += 1
            continue

        origArray[:, :, file_list.index(filename)] = ds.pixel_array

    if(removed != 0):
        print('removed {} files for sequence {}'.format(removed, ref_file.SequenceName))

    return origArray

def load_images_2D(file_list):

    ## loads DICOM images from file_list and creates a nx2D array with pixel data
    ref_file = pydicom.dcmread(file_list[0], force=True)
    image_dims_2d = (len(file_list), int(ref_file.Rows), int(ref_file.Columns))
    # print("Original image size: {}".format(image_dims_2d))

    origArray = np.zeros(image_dims_2d, dtype=ref_file.pixel_array.dtype)

    # loop through all the DICOM files
    removed = 0
    file_list.sort()
    for filename in file_list:
        # read the file
        ds = pydicom.dcmread(filename, force=True)
        # store the raw image data

        if(ds.Rows != ref_file.Rows):
            removed  += 1
            continue

        origArray[file_list.index(filename), :, :] = ds.pixel_array

    if(removed != 0):
        print('removed {} files for sequence {}'.format(removed, ref_file.SequenceName))

    return origArray

def normalize_array(array):
    min = np.min(array)
    max = np.max(array)
    normalized = (array - min) / (max - min)
    return normalized

def scale_range(array, r = 255):
    min = np.min(array)
    max = np.max(array)
    scaled = ((array - min) * (1/(max - min) * r)).astype('uint8')
    return scaled

def scale_array_3D(array, dims):

    ## scales array to IMAGE_DIMENSIONS_3D and normalizes to [0,1]

    scaling_factor = [dims[0]/array.shape[0], dims[1]/array.shape[1], dims[2]/array.shape[2]]
    ## linear interpolation: order = 1
    scaledArray = ndimage.zoom(array, scaling_factor, order = 1)
    # scaledArray = normalize_array(scaledArray)

    # print("Scaled image size: {}".format(dims))
    return scaledArray

def scale_array_2D(array, dims):

    scaledArray = []
    ## scales array to n x IMAGE_DIMENSIONS_2D and normalizes to [0,1]
    for image in array:
        scaling_factor = [dims[0]/image.shape[0], dims[1]/image.shape[1]]
        ## linear interpolation: order = 1
        scaledImage = ndimage.zoom(image, scaling_factor, order = 1)
        # normalizedImage = normalize_array(scaledImage)
        # normalizedImage = scale_range(scaledImage)
        normalizedImage = scaledImage
        scaledArray.append(normalizedImage)

    scaledArray = np.array(scaledArray)
    print("Scaled image array: {}".format(scaledArray.shape))
    return scaledArray


def dicom_3d_to_h5py():

    first_patient = True

    if os.path.exists(PATH_3D_H5):
        os.remove(PATH_3D_H5)

    data_info = pd.read_csv(PATH_IMAGE_INFO)
    baseline_data = pd.read_csv(PATH_CLEAN_UP, sep = ";")
    data_info.patient = [format(id, '03d') for id in data_info.patient]
    baseline_data.pid = [format(id, '03d') for id in baseline_data.pid]

    patient_list = list(set(baseline_data.pid))
    patient_list.sort()

    with h5py.File(PATH_3D_H5, 'a') as f:
        for patient_number in patient_list:

            print('loading sequence for patient {}'.format(patient_number))
            if baseline_data.jpg_available.loc[baseline_data.pid == patient_number].values == 0:
                print('patient {} has no baseline values. Sequence will not be loaded. \n'.format(patient_number))
                continue

            ## copy patient specific filepaths
            patient_data = data_info[data_info.patient == patient_number].copy()

            ## check if patient has a b1000t sequence
            in_set = []
            if patient_data.sequence.isnull().sum() == len(patient_data.sequence): ## all entries are empty
                in_set.append(False)
            else:
                for sequence_type in set(patient_data.sequence):
                    in_set.append('b1000t' in sequence_type)

            ## if patient has no or more than one b1000t sequence: choose sequence by hand
            if sum(in_set) != 1:
                ## --> load images in amide, choose image by name
                print('which image should be loaded?')

                print('0 - none')
                for index, image in enumerate(set(patient_data.description)):
                    print('{} - {}'.format(index+1, image))

                choice = input('>')
                if choice == '0':
                    continue
                chosen_sequence = list(set(patient_data.description))[int(choice) - 1]

                print('{} {} \n'.format(' '.ljust(15), chosen_sequence))
                file_list = list(patient_data.filepath[patient_data.description == chosen_sequence])


            ## patient has one b1000t sequence: choose sequence automatically
            else:
                for sequence_type in set(patient_data.sequence):
                    if 'b1000t' in sequence_type:

                        print('{} {} \n'.format(sequence_type.ljust(15), \
                            set(patient_data.description[patient_data.sequence == sequence_type])))
                        file_list = list(patient_data.filepath[patient_data.sequence == sequence_type])
                    else:
                        continue

            raw_3d_image = load_images_3D(file_list)
            scaled_3d_image = scale_array_3D(raw_3d_image, IMAGE_DIMENSIONS_3D)

            # Image matrices
            X = scaled_3d_image[np.newaxis, :, :, :]
            # Patient ID's
            pat = np.string_([patient_number])
            # Path to images
            path = np.string_([file_list[0][:22]])
            # Patient labels (1=stroke, 0=TIA)
            Y_pat = baseline_data.stroke.loc[baseline_data.pid == patient_number].values
            Y = np.array(Y_pat)

            ## write to h5py sequentially
            ms = [id for id in IMAGE_DIMENSIONS_3D]
            ms.insert(0, None)
            ms = tuple(ms)

            if first_patient: ## initialize dataset
                f.create_dataset('X', data = X, maxshape = ms, chunks = True)
                f.create_dataset('stroke', data = Y_pat, maxshape = (None,), chunks = True)
                f.create_dataset('pat', data = pat, maxshape = (None,), chunks = True)
                f.create_dataset('path', data = path, maxshape = (None,), chunks = True)
                first_patient = False

            else: ## append dataset
                f['X'].resize((f['X'].shape[0] + X.shape[0]), axis = 0)
                f['X'][-X.shape[0]:, :, :, :] = X
                f['stroke'].resize((f['stroke'].shape[0] + Y_pat.shape[0]), axis = 0)
                f['stroke'][-Y_pat.shape[0]:] = Y_pat
                f['pat'].resize((f['pat'].shape[0] + pat.shape[0]), axis = 0)
                f['pat'][-pat.shape[0]:] = pat
                f['path'].resize((f['path'].shape[0] + path.shape[0]), axis = 0)
                f['path'][-path.shape[0]:] = path

def dicom_2d_to_h5py(data_info, baseline_data):

    first_patient = True

    if os.path.exists(PATH_2D_H5_RAW):
        os.remove(PATH_2D_H5_RAW)

    patient_list = list(set(data_info.patient))
    patient_list.sort()

    with h5py.File(PATH_2D_H5_RAW, 'a') as f:
        for patient_number in patient_list:

            print('loading sequence for patient {}'.format(patient_number))
            if patient_number not in list(baseline_data.p_id):
                print('patient {} has no baseline values. Sequence will not be loaded. \n'.format(patient_number))
                continue

            ## copy patient specific filepaths
            patient_data = data_info[data_info.patient == patient_number].copy()

            ## check if patient has a b1000t sequence
            in_set = []
            if patient_data.sequence.isnull().sum() == len(patient_data.sequence): ## all entries are empty
                in_set.append(False)
            else:
                for sequence_type in set(patient_data.sequence):
                    in_set.append('b1000t' in sequence_type)

            ## if patient has no or more than one b1000t sequence: choose sequence by hand
            if sum(in_set) != 1:
                ## --> load images in amide, choose image by name
                print('which image should be loaded?')

                print('0 - none')
                for index, image in enumerate(set(patient_data.description)):
                    print('{} - {}'.format(index+1, image))

                choice = input('>')
                if choice == '0':
                    continue
                chosen_sequence = list(set(patient_data.description))[int(choice) - 1]

                print('{} {} \n'.format(' '.ljust(15), chosen_sequence))
                file_list = list(patient_data.filepath[patient_data.description == chosen_sequence])


            ## patient has one b1000t sequence: choose sequence automatically
            else:
                for sequence_type in set(patient_data.sequence):
                    if 'b1000t' in sequence_type:

                        print('{} {} \n'.format(sequence_type.ljust(15), \
                            set(patient_data.description[patient_data.sequence == sequence_type])))
                        file_list = list(patient_data.filepath[patient_data.sequence == sequence_type])
                    else:
                        continue

            file_list.sort()

            raw_2d_images = load_images_2D(file_list)
            scaled_2d_images = scale_array_2D(raw_2d_images, IMAGE_DIMENSIONS_2D)

            # transform to rgb
            scaled_2d_images = np.stack((scaled_2d_images, scaled_2d_images, scaled_2d_images), axis=3)

            # Image matrices
            n_slices = scaled_2d_images.shape[0]
            X = scaled_2d_images ## dimensions: n_slices x IMAGE_DIMENSIONS_2D

            # Patient labels (1=stroke, 0=TIA)
            Y_pat = baseline_data.stroke.loc[baseline_data.p_id == patient_number].values
            Y_pat = np.reshape([Y_pat] * n_slices, n_slices)
            Y = np.array([3] * n_slices)

            # Image names/number
            # patient_number = format(patient_number, '03d')
            img_ids = np.string_(['{}_{:02d}'.format(patient_number, i) for i in range(n_slices)])
            # Patient ID's
            pat = np.string_([patient_number] * n_slices)
            # Path to images
            path = np.string_(list(file_list))


            ## write to h5py sequentially

            if first_patient: ## initialize dataset
                f.create_dataset('X', data = X, maxshape = MAXSHAPE, chunks = True)
                f.create_dataset('Y', data = Y, maxshape = (None,), chunks = True)
                f.create_dataset('img_id', data = img_ids, maxshape = (None,), chunks = True)
                f.create_dataset('pat', data = pat, maxshape = (None,), chunks = True)
                f.create_dataset('path', data = path, maxshape = (None,), chunks = True)
                f.create_dataset('stroke', data = Y_pat, maxshape = (None,), chunks = True)
                first_patient = False

            else: ## append dataset
                f['X'].resize((f['X'].shape[0] + X.shape[0]), axis = 0)
                f['X'][-X.shape[0]:, :, :] = X
                f['Y'].resize((f['Y'].shape[0] + Y.shape[0]), axis = 0)
                f['Y'][-Y.shape[0]:] = Y
                f['img_id'].resize((f['img_id'].shape[0] + img_ids.shape[0]), axis = 0)
                f['img_id'][-img_ids.shape[0]:] = img_ids
                f['stroke'].resize((f['stroke'].shape[0] + Y_pat.shape[0]), axis = 0)
                f['stroke'][-Y_pat.shape[0]:] = Y_pat
                f['pat'].resize((f['pat'].shape[0] + pat.shape[0]), axis = 0)
                f['pat'][-pat.shape[0]:] = pat
                f['path'].resize((f['path'].shape[0] + path.shape[0]), axis = 0)
                f['path'][-path.shape[0]:] = path

def load_labels_from_jpg_h5(file, patient_nb):
    with h5py.File(file, 'r') as dd:
        patients = [p.decode() for p in dd['pat_3c']]
        indices = [i for i, x in enumerate(patients) if x == patient_nb]
        labels = np.array(dd['Y'][indices])
    return labels

def load_data_from_dicom_h5(file, patient_nb):
    with h5py.File(file, 'r') as dd:
        patients = [p.decode() for p in dd['pat']]
        indices = [i for i, x in enumerate(patients) if x == patient_nb]
        X = dd['X'][indices]
        Y_pat = dd['stroke'][indices]
        pat = dd['pat'][indices]
        path = dd['path'][indices]
        img_id = dd['img_id'][indices]
    return X, Y_pat, pat, path, img_id

def load_image_from_jpg_h5(file, patient_nb):
    with h5py.File(file, 'r') as dd:
        patients = [p.decode() for p in dd['pat_3c']]
        indices = [i for i, x in enumerate(patients) if x == patient_nb]
        mri = np.array(dd['X'][indices])
        labels = np.array(dd['Y'][indices])
    return mri, labels

def cleanup_2d_h5py():
    if os.path.exists(PATH_2D_H5_CLEAN):
        os.remove(PATH_2D_H5_CLEAN)
    first_patient = True

    cleanup = pd.read_csv(PATH_CLEAN_UP, sep = ';')
    cleanup.pid = [format(id, '03d') for id in cleanup.pid]
    patients = cleanup.pid

    for patient in patients:
        status = cleanup.needs_adjustment[cleanup.pid == patient].values

        ## check if status of patient is ok or nok
        if(status == 0):
            print('patient {}: writing labels as is'.format(patient))
            X, Y_pat, pat, path, img_id = load_data_from_dicom_h5(PATH_2D_H5_RAW, patient)
            Y = load_labels_from_jpg_h5(PATH_JPG_H5, patient)

        elif(status == 1): # status is not ok --> print compare jpg and dicom

            reason = cleanup.reason_for_adjustment[cleanup.pid == patient].values

            if(reason == 'dicom image missing'):
                print('patient {}: {}'.format(patient, reason))
                issue = 0
            if(reason == 'jpg image missing'):
                print('patient {}: {}'.format(patient, reason))
                issue = 1
            if(reason == 'inverse'):
                print('patient {}: {}'.format(patient, reason))
                issue = 2

            if(issue == 0): ## if dicom slice is missing: remove label
                p = cleanup.removed[cleanup.pid == patient].item()
                p = str(p)
                rm_label = [int(i) for i in p.split(',')]
                X, Y_pat, pat, path, img_id = load_data_from_dicom_h5(PATH_2D_H5_RAW, patient)
                Y = load_labels_from_jpg_h5(PATH_JPG_H5, patient)
                Y = np.delete(np.array(Y), rm_label)
            if(issue == 1): ## if jpg slice is missing: remove dicom slice
                p = cleanup.removed[cleanup.pid == patient].item()
                p = str(p)
                rm_slice = [int(i) for i in p.split(',')]
                X, Y_pat, pat, path, img_id = load_data_from_dicom_h5(PATH_2D_H5_RAW, patient)
                X = np.delete(np.array(X), rm_slice, 0)
                Y_pat = np.delete(np.array(Y_pat), rm_slice)
                pat = np.delete(np.array(pat), rm_slice)
                path = np.delete(np.array(path), rm_slice)
                img_id = np.delete(np.array(img_id), rm_slice)
                Y = load_labels_from_jpg_h5(PATH_JPG_H5, patient)

            if(issue == 2): ## order of images is inversed
                X, Y_pat, pat, path, img_id = load_data_from_dicom_h5(PATH_2D_H5_RAW, patient)
                Y = load_labels_from_jpg_h5(PATH_JPG_H5, patient)
                Y = Y[::-1]
                print(Y)

        else:
            print('patient {}: will get removed'.format(patient))
            continue

        with h5py.File(PATH_2D_H5_CLEAN, 'a') as f:
            if first_patient: ## initialize dataset
                f.create_dataset('X', data = X, maxshape = MAXSHAPE, chunks = True)
                f.create_dataset('Y', data = Y, maxshape = (None,), chunks = True)
                f.create_dataset('img_id', data = img_id, maxshape = (None,), chunks = True)
                f.create_dataset('stroke', data = Y_pat, maxshape = (None,), chunks = True)
                f.create_dataset('pat', data = pat, maxshape = (None,), chunks = True)
                f.create_dataset('path', data = path, maxshape = (None,), chunks = True)
                first_patient = False

            else: ## append dataset
                f['X'].resize((f['X'].shape[0] + X.shape[0]), axis = 0)
                f['X'][-X.shape[0]:, :, :] = X
                f['Y'].resize((f['Y'].shape[0] + Y.shape[0]), axis = 0)
                f['Y'][-Y.shape[0]:] = Y
                f['img_id'].resize((f['img_id'].shape[0] + img_id.shape[0]), axis = 0)
                f['img_id'][-img_id.shape[0]:] = img_id
                f['stroke'].resize((f['stroke'].shape[0] + Y_pat.shape[0]), axis = 0)
                f['stroke'][-Y_pat.shape[0]:] = Y_pat
                f['pat'].resize((f['pat'].shape[0] + pat.shape[0]), axis = 0)
                f['pat'][-pat.shape[0]:] = pat
                f['path'].resize((f['path'].shape[0] + path.shape[0]), axis = 0)
                f['path'][-path.shape[0]:] = path

#cleanup_2d_h5py()
