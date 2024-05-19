for dataset in SleepEEG FD_A HAR ECG 
do
    for mixup in False True
    do 
        for add_target in False True
        do
            for epoch_v_iter in epoch n_sample
            do
                python main.py --training_mode fine_tune_test --pretrain_dataset $dataset --target_dataset Epilepsy --use_mixup $mixup --seed 0 --add_target_to_pretrain $add_target --fix_n_epoch_or_n_sample $epoch_v_iter --pre_train_seed 0
            done
        done
    done
done