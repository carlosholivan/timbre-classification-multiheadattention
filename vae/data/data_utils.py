import os
import shutil
import librosa
from pydub import AudioSegment
import numpy as np

# Our modules
from vae import configs


def convert_mp3_to_wav(audio_file_path, delete_mp3=False):
    """This function converts an mp3 file to a wav file"""

    if audio_file_path.split(".", 1)[1] == 'mp3':
        try:
            sound = AudioSegment.from_mp3(audio_file_path)
            audio_wav_file_path = audio_file_path.split(".", 1)[0] + ".wav"
            sound.export(audio_wav_file_path, format="wav")  # convert to wav file
            print(audio_file_path, 'file converted from', audio_file_path.split(".", 1)[1], 'to wav format')

            # delete mp3 file
            if delete_mp3:
                os.remove(audio_file_path)

            return audio_wav_file_path

        except:
            raise ValueError('File cannot be converted to wav')
    else:
        raise ValueError('Inserted file is not an mp3 file')


def compute_input(audio_file_path):
    """This function computes the cqt of an audio file given its path"""

    try:
        y, sr = librosa.load(audio_file_path, sr=None)

    except:
        # Call convert_mp3_to_wav to convert mp3 to wav
        new_wav_path = convert_mp3_to_wav(audio_file_path)
        y, sr = librosa.load(new_wav_path, sr=None)

    # centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    cqt = np.abs(librosa.cqt(y,
                             hop_length=configs.InputsConfig.HOP_LENGTH,
                             fmin=configs.InputsConfig.F_MIN,
                             n_bins=configs.InputsConfig.BINS,
                             bins_per_octave=configs.InputsConfig.BINS_PER_OCTAVE))

    return cqt


def store_inputs(dataset_path):

    """This function computes the cqts of all the audio files in
    dataset_path and stores them in data directory in our module."""

    data_path = './data'

    # Create data directory to store the input arrays
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    for (im_dirpath, im_dirnames, im_filenames) in os.walk(dataset_path):
        for f in im_filenames:  # loop in files
            file_name = f.split(".", 1)[0]
            new_path = os.path.join(data_path, file_name + '.npy')

            if os.path.exists(new_path):  # if file is already in data, skip it
                print(new_path, 'already exists')
                continue
            else:
                try:
                    audio_file_path = os.path.join(im_dirpath, f)  # get the audio file path
                    inputs = compute_input(audio_file_path)  # compute the centroid of the audio file
                    np.save(new_path, inputs)  # stores arrays in data directory

                except:
                    print('Skipping file:', f)
                    continue


def dataset_split_table(train_split=0.85, val_split=0.1, 
                        dataset_path=configs.ParamsConfig.CQT_DATA_PATH, export_excel=True):

    # divide dataset in train, validation and test
    all_instruments = []  # all instruments in dataset (also repeated instruments)
    non_repeated_instruments = []  # all instruments (non repeated) in dataset
    for f in os.listdir(dataset_path):
        if f.split(".", 1)[1] == 'npy':
                input_file_path = os.path.join(dataset_path, f)  # get the audio file path
                instrument = os.path.split(input_file_path)[1].split('.')[0]
                instrument_name = os.path.split(instrument)[1].split('_')[0]
                if instrument_name not in all_instruments:
                    non_repeated_instruments.append(instrument_name)
                all_instruments.append(instrument_name)

        counts = [all_instruments.count(instrument) for instrument in non_repeated_instruments]

        # rows: instrumen names, cols: count, train samples (85%), validation samples (10%) and test samples (5%)
        data = {'instrument': [instrument for instrument in non_repeated_instruments],
                'total samples': counts,
                'train samples': [round(count * train_split) for count in counts],
                'val samples': [round(count * val_split) for count in counts],
                'test samples': [count - round(count * train_split) - round(count * val_split) for count in counts]
               }

        df = pd.DataFrame(data=data)

    if export_excel:
        df.to_excel("../data_split.xlsx")

    return df


def store_split_dataset(dataset_path=configs.ParamsConfig.CQT_DATA_PATH, train_split=0.85, val_split=0.1):

    df = dataset_split_table(train_split, val_split)

    instrument_names = []
    for f in os.listdir(dataset_path):  # loop in files
        if f.split(".", 1)[1] == 'npy':
            input_file_path = os.path.join(dataset_path, f)  # get the audio file path
            instrument = os.path.split(input_file_path)[1].split('.')[0]
            instrument_name = os.path.split(instrument)[1].split('_')[0]
            instrument_names.append(instrument_name)
            count = instrument_names.count(instrument_name)
            data = df.loc[df['instrument'] == instrument_name]
            train_samples = int(data['train samples'])
            val_samples = int(data['val samples'])
            total_samples = int(data['total samples'])

            # create train, val and test subdirectories inside data directory if they don't already exist
            if not os.path.exists(configs.ParamsConfig.DATA_TRAIN_PATH):
                os.mkdir(configs.ParamsConfig.DATA_TRAIN_PATH)
            if not os.path.exists(configs.ParamsConfig.DATA_VAL_PATH):
                os.mkdir(configs.ParamsConfig.DATA_VAL_PATH)
            if not os.path.exists(configs.ParamsConfig.DATA_TEST_PATH):
                os.mkdir(configs.ParamsConfig.DATA_TEST_PATH)

            # save split dataset
            if count <= train_samples:
                shutil.move(input_file_path, configs.ParamsConfig.DATA_TRAIN_PATH) 
                print('Moved file:', input_file_path, 'to train path:', configs.ParamsConfig.DATA_TRAIN_PATH + '/' + f)
            elif (count > train_samples) and (count <= train_samples + val_samples):
                shutil.move(input_file_path, configs.ParamsConfig.DATA_VAL_PATH) 
                print('Moved file:', input_file_path, 'to val path:', configs.ParamsConfig.DATA_VAL_PATH + '/' + f)
            elif (count > train_samples + val_samples):
                shutil.move(input_file_path, configs.ParamsConfig.DATA_TEST_PATH) 
                print('Moved file:', input_file_path, 'to test path:', configs.ParamsConfig.DATA_TEST_PATH + '/' + f)

    return


def get_max_frames(dataset_path=configs.ParamsConfig.CQT_DATA_PATH):

    max_time_dim = 0
    subdirs = os.listdir(dataset_path)

    for subdir in subdirs:
        files = os.listdir(os.path.join(dataset_path, subdir))
        for f in files:
            if f.split(".", 1)[1] == 'npy':
                input_file_path = os.path.join(dataset_path, subdir, f)
                input_file_data = np.load(input_file_path)
                time_dim = input_file_data.shape[1]
                if time_dim > max_time_dim:
                    max_time_dim = time_dim
    return max_time_dim


def padding(data, max_time):

    time_dim = data['input'].shape[2]

    if time_dim < max_time:
        padding_factor = max_time - time_dim  # frames (cols) to add to the CQT
        pad_image = np.full((data['input'].shape[1], padding_factor), 0) 
        pad_image = pad_image[np.newaxis,:,:]

        # concatenate padding to original CQT
        data['input'] = np.concatenate((data['input'], pad_image), axis=-1)

    return data


def cut_input(data, threshold=0.1, stride=22):
    cqt = data
    threshold = 0.1
    first_frame = 0
    for frame in range(cqt.shape[2]):
        max_value = np.max(cqt[:, :, frame])
        if max_value > threshold:
            first_frame = frame
            break
        else:
            continue

    new_frames_length = first_frame + stride

    #pad silence if we do not hace enough timeframes
    if cqt.shape[2] < new_frames_length:
        padding_factor = new_frames_length - cqt.shape[2]
        pad_image = np.full((cqt.shape[1], padding_factor), 0)
        pad_image = pad_image[np.newaxis,:,:]
        cqt_padded = np.concatenate((cqt, pad_image), axis=-1)
        data = cqt_padded[:, :, first_frame:first_frame+stride]
    else:
        data = cqt[:, :, first_frame:first_frame+stride]
    return data

import sklearn
def normalize(data):
    cqt = data[0, ...]
    new_cqt = sklearn.preprocessing.normalize(X=cqt, norm='l2', axis=0) #freq axis
    data = new_cqt[np.newaxis, ...]
    return data

def compress_input(cqt):
    cqt = cqt[0, ...]
    factor = int(np.ceil(cqt.shape[1] / 128))
    t = 0
    compressed_cqt = np.zeros(shape=(cqt.shape[0], 128))
    for i in range(0, cqt.shape[0], 1):
        j = 0
        for t in range(0, cqt.shape[1], factor):
            mean_y_axis = np.sum(cqt[i, t:t+factor]) / 2
            compressed_cqt[i, j] = mean_y_axis
            j += 1
    compressed_cqt = compressed_cqt[np.newaxis, ...]
    return compressed_cqt
