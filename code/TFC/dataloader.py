import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD, mixup_datasets
import torch.fft as fft
import itertools as it

def generate_freq(dataset, config):
    X_train = dataset["samples"]
    y_train = dataset['labels']
    # shuffle
    data = list(zip(X_train, y_train))
    np.random.shuffle(data)
    data = data[:10000] # take a subset for testing.
    X_train, y_train = zip(*data)
    X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

    """Align the TS length between source and target datasets"""
    X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

    if isinstance(X_train, np.ndarray):
        x_data = torch.from_numpy(X_train)
    else:
        x_data = X_train

    """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft,
    the output shape is half of the time window."""

    x_data_f = fft.fft(x_data).abs() #/(window_length) # rfft for real value inputs.
    return (X_train, y_train, x_data_f)

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = y_train.long()

        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = self.adjust_ts_length(X_train, config.TSlength_aligned) # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            print('Using subset for debugging, the datasize is:', y_train.shape[0])

        #print("Type of X_train: ", type(X_train))
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train
        #print("Memory usage of X_train in KB: ", X_train.element_size() * X_train.nelement()//1024 )
        #print("Memory usage of y_train in KB: ", y_train.element_size() * y_train.nelement()//1024 )
        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft,
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]

        """Augmentation"""
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def adjust_ts_length(self, X_train, length):

        if X_train.shape[2] >= length:
            return X_train[:, :1, :int(length)]  # X_train[:, :1, :int(config.TSlength_aligned)]

        # Otherwise the dataset's time-series length isn't long enough
        # Create the Fourier-transform
        x_data_f = fft.fft(X_train)

        # Put 0's to the end of the Fourier representation
        shape = list(x_data_f.shape)
        shape[2] = length - x_data_f.shape[2]
        filler = torch.zeros(shape)

        x_data_f = torch.cat((x_data_f, filler), 2)

        # Cast it back to time-series
        x_data_t = fft.ifft(x_data_f)
        x_data_t = x_data_t.real

        return x_data_t[:, :1, :int(length)]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset=True, add_target = False, epoch_v_iter = "epoch"):

    # Loading in the torch arrays
    print(f"The order of the datsets: {sourcedata_path}")
    train_datasets = [] # If there are multiple datasets for pre.training load in all
    for source_path in sourcedata_path:
        new_train_dataset = torch.load(os.path.join(source_path, "train.pt"))
        train_datasets.append(new_train_dataset)

    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))  # train.pt
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))  # test.pt

    """In pre-training:
    train_dataset: [371055, 1, 178] from SleepEEG.
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""

    # Creating datasets
    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_datasets.pop(0), configs, training_mode,
                                    target_dataset_size=configs.target_batch_size, subset=subset)
    
    # If there are more than one pre_train dataset merge them together
    for train_set in train_datasets:
        current_train = Load_Dataset(train_set, configs, training_mode,
                                        target_dataset_size=configs.target_batch_size, subset=subset)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, current_train])

    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode,
                                    target_dataset_size=configs.target_batch_size, subset=subset)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode,
                                target_dataset_size=configs.target_batch_size, subset=False)

    # Add the target training dataset to the pre-train [include the aricle.]
    if add_target:
        print("Target train dataset added to pre_train dataset")
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, finetune_dataset])

    # Create the dataloaders
    if epoch_v_iter == "n_sample":
        print(f" Num iter is fixed to: {configs.num_iter}")
        sampler = RandomSampler(train_dataset, replacement=True, num_samples = configs.num_iter // configs.pre_train_num_epoch)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.target_batch_size,
                                               shuffle=False, drop_last=configs.drop_last, sampler = sampler,
                                               num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.target_batch_size,
                                                shuffle=True, drop_last=configs.drop_last,
                                                num_workers=0)
        
    finetune_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=True,
                                              num_workers=0)

    return train_loader, test_loader, finetune_loader 