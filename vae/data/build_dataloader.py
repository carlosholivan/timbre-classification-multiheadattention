import os
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Our modules
import sys
sys.path.append('.')
sys.path.append('..')

from vae.data import data_utils
from vae import configs

classes = ['violin', 'viola', 'cello', 'double-bass',
                'clarinet', 'bass-clarinet', 'saxophone', 'flute', 'oboe', 'bassoon', 'contrabassoon',
                'french-horn', 'trombone', 'trumpet', 'tuba', 'english-horn',
                'guitar', 'mandolin', 'banjo', 'chromatic-percussion']

chromatic_perc = ['agogo-bells', 'banana-shaker', 'bass-drum', 'bell-tree', 'cabasa', 'Chinese-hand-cymbals',
                        'castanets', 'Chinese-cymbal', 'clash-cymbals', 'cowbell', 'djembe', 'djundjun', 'flexatone', 'guiro',
                        'lemon-shaker',  'motor-horn',  'ratchet', 'sheeps-toenails', 'sizzle-cymbal', 'sleigh-bells', 'snare-drum',
                        'spring-coil', 'squeaker', 'strawberry-shaker', 'surdo', 'suspended-cymbal', 'swanee-whistle',
                        'tambourine', 'tam-tam', 'tenor-drum', 'Thai-gong', 'tom-toms', 'train-whistle', 'triangle',
                        'vibraslap', 'washboard', 'whip', 'wind-chimes', 'woodblock']

# Dataloader
def build_dataset(input, model_name):

    if input == 'mel':
        data_train_path = configs.ParamsConfig.MEL_DATA_TRAIN_PATH
        data_val_path = configs.ParamsConfig.MEL_DATA_VAL_PATH
        transforms = None
    elif input == 'mel_cut':
        data_train_path = configs.ParamsConfig.MEL_DATA_TRAIN_PATH
        data_val_path = configs.ParamsConfig.MEL_DATA_VAL_PATH
        transforms = [data_utils.cut_input]
    elif input == 'cqt':
        data_train_path = configs.ParamsConfig.CQT_DATA_TRAIN_PATH
        data_val_path = configs.ParamsConfig.CQT_DATA_VAL_PATH
        transforms = [data_utils.compress_input, data_utils.normalize]

    if model_name.split('_')[0] == 'supervised':
        train_dataset = AudioSupervisedDataset(data_path=data_train_path,
                                transforms=transforms)

        val_dataset = AudioSupervisedDataset(data_path=data_val_path,
                                transforms=transforms)
       
    else:
        train_dataset = AudioDataset(data_path=data_train_path,
                                transforms=transforms)

        val_dataset = AudioDataset(data_path=data_val_path,
                                transforms=transforms)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=configs.ParamsConfig.BATCH_SIZE,
                                  num_workers=10,
                                  shuffle=True,
                                  pin_memory=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=configs.ParamsConfig.BATCH_SIZE,
                                num_workers=10,
                                shuffle=False,
                                pin_memory=True)

    return train_dataset, train_dataloader, val_dataset, val_dataloader


def build_testset(input, model_name):

    if input == 'mel':
        data_test_path = configs.ParamsConfig.MEL_DATA_VAL_PATH
        transforms = None
    elif input == 'mel_cut':
        data_test_path = configs.ParamsConfig.MEL_DATA_VAL_PATH
        transforms = [data_utils.cut_input]
    elif input == 'cqt':
        data_test_path = configs.ParamsConfig.CQT_DATA_VAL_PATH
        transforms = [data_utils.compress_input, data_utils.normalize]

    if model_name.split('_')[0] == 'supervised':
        test_dataset = AudioSupervisedDataset(data_path=data_test_path,
                                transforms=transforms)
    else:
        test_dataset = AudioDataset(data_path=data_test_path,
                                    transforms=transforms)

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=10,
                                  shuffle=True,
                                  pin_memory=True)


    return test_dataset, test_dataloader


class AudioDataset(Dataset):

    def __init__(self, data_path, transforms=None):

        """
        Args
        ----
            data_path : Path to all the array files
            audio_file_path : Path of a single audio file
        """

        self.data_path = data_path
        self.input_data = []
        self.files_path = []
        #self.time_dims = []

        for (im_dirpath, im_dirnames, im_filenames) in os.walk(self.data_path):
            for f in im_filenames:  # loop in files
                if f.split(".", 1)[1] == 'npy':
                    input_file_path = os.path.join(im_dirpath, f)  # get the audio file path
                    input_file_data = np.load(input_file_path)  # load npy file
                    #self.time_dims.append(input_file_data.shape[1])

                    # append variables to lists
                    self.input_data.append(input_file_data)
                    self.files_path.append(input_file_path)

        self.transforms = transforms

    def __len__(self):
        """count audio files"""
        return len(self.input_data)

    def __getitem__(self, index):
        """take audio file form list"""

        input_data = self.input_data[index]
        input_data = input_data[np.newaxis, :, :]  # add axis for batch

        audio_file = self.files_path[index]
        instrument_class = self.files_path[index]

        file = os.path.split(audio_file)[1].split('.')[0]
        input_data = input_data.astype('float64')

        
        if self.transforms is not None:
            for t in self.transforms:
                input_data = t(input_data)

        return input_data, file


class AudioSupervisedDataset(Dataset):

    def __init__(self, data_path, transforms=None):

        """
        Args
        ----
            data_path : Path to all the array files
            audio_file_path : Path of a single audio file
        """

        self.data_path = data_path
        self.input_data = []
        self.files_path = []
        #self.time_dims = []

        for (im_dirpath, im_dirnames, im_filenames) in os.walk(self.data_path):
            for f in im_filenames:  # loop in files
                if f.split(".", 1)[1] == 'npy':
                    input_file_path = os.path.join(im_dirpath, f)  # get the audio file path
                    input_file_data = np.load(input_file_path)  # load npy file
                    #self.time_dims.append(input_file_data.shape[1])

                    # append variables to lists
                    self.input_data.append(input_file_data)
                    self.files_path.append(input_file_path)

        self.transforms = transforms

    def __len__(self):
        """count audio files"""
        return len(self.input_data)

    def __getitem__(self, index):
        """take audio file form list"""

        input_data = self.input_data[index]
        input_data = input_data[np.newaxis, :, :]  # add axis for batch

        audio_file = self.files_path[index]
        instrument_class = self.files_path[index]

        file = os.path.split(audio_file)[1].split('.')[0]
        file_name = file.split('_')[0]
        if file_name in chromatic_perc:
            file_name = 'chromatic-percussion'
        y = classes.index(file_name)

        input_data = input_data.astype('float64')
        
        if self.transforms is not None:
            for t in self.transforms:
                input_data = t(input_data)

        return input_data, file, y


