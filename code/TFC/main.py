"""Updated implementation for TF-C -- Xiang Zhang, Jan 16, 2023"""

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
from conv_model import *
from dataloader import data_generator
from trainer import Trainer
import wandb

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')

parser.add_argument('--seed', default=42, type=int, help='seed value')

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    help='pre_train, fine_tune_test')

parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str, nargs = "+",
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')

parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')

parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')

parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--use_mixup', default="False", type=str,
                    help='The use of mixup strategy during pre-train if there are two or more pre-train dataset')

parser.add_argument('--add_target_to_pretrain', default = False, type=str, 
                    help = "Whether to add the target dataset to the pre_train dataset(s)." )

parser.add_argument('--fix_n_epoch_or_n_sample', default = "epoch", type = str,
                    help = 'Whether to have a fix epoch number to iterate through the dataset epoch times, or to fix number of sample seen. Choice: {epoch, n_sample}')

parser.add_argument('--tags', default="", type=str,
                    help='Tags for wandb.')

parser.add_argument('--pre_train_seed', default=42, type=str,
                    help='Which pre-train seed to use for fine_tune model')



args, unknown = parser.parse_known_args()

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")

print('We are using %s now.' %device)

pretrain_dataset = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(pretrain_dataset) + '_2_' + str(targetdata) + '_conv'

method = 'TF-C'
training_mode = args.training_mode
run_description = args.run_description # + '_mixup_' + str(args.use_mixup)
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.SleepEEG_Configs import Config as Configs') #TODO change this, so it depends only on the fine_tune dataset
configs = Configs()

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_2layertransformer")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {pretrain_dataset}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
sourcedata_path = [f"../../datasets/{pre}" for pre in pretrain_dataset]
targetdata_path = f"../../datasets/{targetdata}"
subset = False  # if subset= true, use a subset for debugging.
mixup = False if args.use_mixup == "False" else True
add_target = False  if args.add_target_to_pretrain == "False" else True
epoch_v_iter = args.fix_n_epoch_or_n_sample
print("Are we using mixup? ", mixup)
tags = args.tags.split(',') if args.tags != "" else []
pre_train_seed = args.pre_train_seed

wandb.init(
    # set the wandb project where this run will be logged
    entity="marauders",
    project="TFC_pre-training",
    
    name = experiment_description + '_' + training_mode + '_' + str(mixup) + '_' + str(SEED),
    # track hyperparameters and run metadata
    config = configs,
    tags=tags
)

wandb.log({"experiment_dir" : experiment_log_dir,
            "pre_train_dataset" : pretrain_dataset,
            "target_dataset" : targetdata,
            "method" : method, 
            "mode" : training_mode,
            "mixup" : mixup,
            "alpha" : configs.alpha,
            "subset" : subset,
            "epoch_v_iter" : epoch_v_iter,
            "add_target" : add_target,
            "seed": SEED})

train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset = subset, add_target = add_target, epoch_v_iter = epoch_v_iter)
logger.debug("Data loaded ...")

# Load Model
"""Here are two models, one basemodel, another is temporal contrastive model"""
TFC_model = TFC(configs).to(device)
classifier = target_classifier(configs).to(device)
temporal_contr_model = None

verbose = False
if verbose == True:
    #from torchsummary import summary
    #summary(TFC_model, input_size=(1, 5120), batch_size=64)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"Total memory: {t // 1024**2} MB")
    print(f"Reserved: {r // 1024**2} MB")
    print(f"allocated: {a // 1024**2} MB")
    print(f"Free: {f // 1024**2} MB")

if training_mode == "fine_tune_test":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{pre_train_seed}_2layertransformer", "saved_models")) #SEED

    wandb.log({"pre_trained_model_dir" : load_from})

    print("The loading file path", load_from)
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    TFC_model.load_state_dict(pretrained_dict)

model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainer
Trainer(TFC_model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, experiment_log_dir, training_mode, use_mixup = mixup)

logger.debug(f"Training time is : {datetime.now()-start_time}")

wandb.finish()