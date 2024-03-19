import numpy as np
import torch
import torch.fft as fft

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug

# def DataTransform_TD(sample, config):
#     """Weak and strong augmentations"""
#     weak_aug = sample
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio) #masking(sample)
#     return weak_aug, strong_aug
#
# def DataTransform_FD(sample, config):
#     """Weak and strong augmentations in Frequency domain """
#     # weak_aug =  remove_frequency(sample, 0.1)
#     strong_aug = add_frequency(sample, 0.1)
#     return weak_aug, strong_aug
def DataTransform_TD(sample, config):
    """Simplely use the jittering augmentation. Feel free to add more autmentations you want,
    but we noticed that in TF-C framework, the augmentation has litter impact on the final tranfering performance."""
    aug = jitter(sample, config.augmentation.jitter_ratio)
    return aug


def DataTransform_TD_bank(sample, config):
    """Augmentation bank that includes four augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function."""
    aug_1 = jitter(sample, config.augmentation.jitter_ratio)
    aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)
    aug_4 = masking(sample, keepratio=0.9)

    li = np.random.randint(0, 4, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    aug_1 = aug_1 * li_onehot[:, 0][:, None, None]  # the rows that are not selected are set as zero.
    aug_2 = aug_2 * li_onehot[:, 0][:, None, None]
    aug_3 = aug_3 * li_onehot[:, 0][:, None, None]
    aug_4 = aug_4 * li_onehot[:, 0][:, None, None]
    aug_T = aug_1 + aug_2 + aug_3 + aug_4
    return aug_T

def DataTransform_FD(sample, config):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def mixup_datasets(dataset_left, dataset_right, config, alpha = 0.2):
    # Administrative steps for dataset handling
    X_train_left = dataset_left["samples"]
    X_train_right = dataset_right["samples"]

    y_train_left = dataset_left["labels"]
    y_train_right = dataset_right["labels"]

    if isinstance(X_train_left, np.ndarray):
        X_train_left = torch.from_numpy(X_train_left)
        y_train_left = y_train_left.long()
    if isinstance(X_train_right, np.ndarray):
        X_train_right = torch.from_numpy(X_train_right)
        y_train_right = y_train_right.long()


    # shuffle
    data_left = list(zip(X_train_left, y_train_left))
    data_right = list(zip(X_train_right, y_train_right))

    np.random.shuffle(data_left)
    np.random.shuffle(data_right)

    X_train_left, y_train_left = zip(*data_left)
    X_train_right, y_train_right = zip(*data_right)

    X_train_left, y_train_left = torch.stack(list(X_train_left), dim=0), torch.stack(list(y_train_left), dim=0)
    X_train_right, y_train_right = torch.stack(list(X_train_right), dim=0), torch.stack(list(y_train_right), dim=0)

    if len(X_train_left.shape) < 3:
        X_train_left = X_train_left.unsqueeze(2)
    if X_train_left.shape.index(min(X_train_left.shape)) != 1:  # make sure the Channels in second dim
        X_train_left = X_train_left.permute(0, 2, 1)

    if len(X_train_right.shape) < 3:
        X_train_right = X_train_right.unsqueeze(2)
    if X_train_right.shape.index(min(X_train_right.shape)) != 1:  # make sure the Channels in second dim
        X_train_right = X_train_right.permute(0, 2, 1)

    # If there are more than one channel take the first
    X_train_left = X_train_left[:,:1,:] 
    X_train_right = X_train_right[:,:1,:]

    # print("Shapes after:")
    # print("X_train_left.shape: ", X_train_left.shape)
    # print("X_train_right.shape: ", X_train_right.shape)

    """Cast the time-series into frequency domain"""
    """Align the TS length between source and target datasets"""
    """ Maybe a random TSLength_aligned segment would be better"""
    X_train_left = X_train_left[:, :1, :int(config.TSlength_aligned)]
    X_train_right = X_train_right[:, :1, :int(config.TSlength_aligned)]

    X_left_f = fft.fft(X_train_left).abs()
    X_right_f = fft.fft(X_train_left).abs()
    
    # Generate new points if slow #TODO improve
    # new_dataset = {"samples": torch.tensor, "labels": torch.LongTensor }
    new_X = []
    new_y = []
    print(f"Generating lambda with alpha: {alpha}")
    for (x_1, y_1, x_2,y_2) in zip(X_left_f, y_train_left, X_right_f, y_train_right): # For now, the smaller dataset acts as cap 
        lam = np.random.beta(alpha, alpha)
        x = (lam * x_1 + (1. - lam) * x_2)
        y = (lam * y_1 + (1. - lam) * y_2)

        if x.shape[1] < int(config.TSlength_aligned):
            tmp = torch.zeroes(x.shape[0],int(config.TSlength_aligned) - x.shape[1] )
            x = torch.cat(x,tmp)

        """Cast the data back to time-series domain"""
        x = fft.ifft(x)
        x = x.unsqueeze(0)

        new_X.append(x)
        new_y.append(y)

    #print("type of newX:", type(new_X))
    mix_dataset = {"samples" : torch.cat(new_X,0), "labels" : torch.tensor(new_y)}
    # print("mix_dataset samples", type(mix_dataset["samples"]), mix_dataset["samples"].shape)
    # print("mix_dataset labels", type(mix_dataset["labels"]), mix_dataset["labels"].shape)
    return mix_dataset

        
