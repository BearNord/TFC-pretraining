python main.py --training_mode pre_train --pretrain_dataset HAR FD_A  --target_dataset Gesture --use_mixup True
python main.py --training_mode fine_tune_test --pretrain_dataset HAR FD_A  --target_dataset Gesture --use_mixup True
python main.py --training_mode pre_train --pretrain_dataset HAR FD_A ECG --target_dataset Gesture --use_mixup True
python main.py --training_mode fine_tune_test --pretrain_dataset HAR FD_A ECG --target_dataset Gesture --use_mixup True
python main.py --training_mode pre_train --pretrain_dataset HAR FD_A ECG SleepEEG --target_dataset Gesture --use_mixup True
python main.py --training_mode fine_tune_test --pretrain_dataset HAR FD_A ECG SleepEEG --target_dataset Gesture --use_mixup True