import os
import sys
sys.path.append("..")
import wandb

from loss import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from model import * 
import numpy as np

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def mixup_data(data, aug1, data_f, aug1_f, alpha = 1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # print("Incoming dimensions: ", data.shape, aug1.shape, data_f.shape, aug1_f.shape)
    batch_size = data.size()[0]
    index = torch.randperm(batch_size)

    mixed_data = lam * data + (1 - lam) * data[index, :]
    mixed_aug1 = lam * aug1 + (1 - lam) * aug1[index, :]
    mixed_data_f = lam * data_f + (1 - lam) * data_f[index, :]
    mixed_aug1_f = lam * aug1_f + (1 - lam) * aug1_f[index, :]
    # print("Outgoing dimensions: ", mixed_data.shape, mixed_aug1.shape, mixed_data_f.shape, mixed_aug1_f.shape)

    return mixed_data, mixed_aug1, mixed_data_f, mixed_aug1_f

def Trainer(model,  model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, use_mixup):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    global step
    
    if training_mode == 'pre_train':
        print('Pretraining on source dataset')
        for epoch in range(1, config.pre_train_num_epoch + 1):
            step = epoch
            step += 1
            # Train and validate
            """Train. In fine-tuning, this part is also trained???"""
            train_loss = model_pretrain(model, model_optimizer, criterion, train_dl, config, device, training_mode, use_mixup)
            logger.debug(f'\nPre-training Epoch : {epoch} \nTrain Loss : {train_loss:.4f}')

        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))

    """Fine-tuning and Test"""
    if training_mode != 'pre_train':
        """fine-tune"""
        print('Fine-tune on Fine-tuning set')
        performance_list = []
        total_f1 = []
        KNN_f1 = []
        global emb_finetune, label_finetune, emb_test, label_test

        update_counter = 0

        for epoch in range(1, config.fine_tune_num_epoch + 1):
            logger.debug(f'\nEpoch : {epoch}')
            step = epoch
            step += 1

            valid_loss, emb_finetune, label_finetune, F1 = model_finetune(model, model_optimizer, valid_dl, config,
                                  device, training_mode, classifier=classifier, classifier_optimizer=classifier_optimizer)
            scheduler.step(valid_loss)


            # save best fine-tuning model""
            global arch
            arch = 'sleepedf2eplipsy'
            if len(total_f1) == 0 or F1 > max(total_f1):
                print('update fine-tuned model')
                update_counter += 1
                os.makedirs(os.path.join(experiment_log_dir,'finetunemodel'), exist_ok=True) #, os.path.join(experiment_log_dir,finetunemodel)  exist_ok=True
                torch.save(model.state_dict(), os.path.join(experiment_log_dir,'finetunemodel/' + arch + '_model.pt'))
                torch.save(classifier.state_dict(), os.path.join(experiment_log_dir,'finetunemodel/' + arch + '_classifier.pt'))
            total_f1.append(F1)

            # evaluate on the test set
            """Testing set"""
            logger.debug('Test on Target datasts test set')
            model.load_state_dict(torch.load( os.path.join(experiment_log_dir,'finetunemodel/' + arch + '_model.pt')))
            classifier.load_state_dict(torch.load(os.path.join(experiment_log_dir,'finetunemodel/' + arch + '_classifier.pt')))
            test_loss, test_acc, test_auc, test_prc, emb_test, label_test, performance = model_test(model, test_dl, config, device, training_mode,
                                                             classifier=classifier, classifier_optimizer=classifier_optimizer)
            performance_list.append(performance)

            # Model and classifier learns for 1 epoch. If the classifier's last batch of predictions had higher F1 score, 
            # than previous highest F1 then save the weights, otherwise revert back to previous 

            """Use KNN as another classifier; it's an alternation of the MLP classifier in function model_test. 
            Experiments show KNN and MLP may work differently in different settings, so here we provide both."""
            # train classifier: KNN
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(emb_finetune, label_finetune)
            knn_acc_train = neigh.score(emb_finetune, label_finetune)
            # print('KNN finetune acc:', knn_acc_train)
            representation_test = emb_test.detach().cpu().numpy()

            knn_result = neigh.predict(representation_test)
            knn_result_score = neigh.predict_proba(representation_test)
            one_hot_label_test = one_hot_encoding(label_test)
            # print(classification_report(label_test, knn_result, digits=4))
            # print(confusion_matrix(label_test, knn_result))
            knn_acc = accuracy_score(label_test, knn_result)
            precision = precision_score(label_test, knn_result, average='macro', )
            recall = recall_score(label_test, knn_result, average='macro', )
            F1 = f1_score(label_test, knn_result, average='macro')
            auc = roc_auc_score(one_hot_label_test, knn_result_score, average="macro", multi_class="ovr")
            prc = average_precision_score(one_hot_label_test, knn_result_score, average="macro")
            print('KNN Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'%
                  (knn_acc, precision, recall, F1, auc, prc))
            KNN_f1.append(F1)
        torch.save(model.state_dict(), os.path.join(experiment_log_dir,'finetunemodel/' + arch + '_model.pt'))
        torch.save(classifier.state_dict(), os.path.join(experiment_log_dir,'finetunemodel/' + arch + '_classifier.pt'))
        logger.debug("\n################## Best testing performance! #########################")
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:,0], axis=0)]
        print('Best Testing Performance: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f '
              '| AUPRC=%.4f' % (best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                                best_performance[4], best_performance[5]))
        print('Best KNN F1', max(KNN_f1))
        print("Total number of updates to weights: ", update_counter)


    logger.debug("\n################## Training is Done! #########################")

def model_pretrain(model, model_optimizer, criterion, train_loader, config, device, training_mode, use_mixup):
    total_loss = []
    model.train()
    global loss, loss_t, loss_f, l_TF, loss_c, data_test, data_f_test

    # optimizer
    model_optimizer.zero_grad()

    # TODO We want to use proper mixup here
    # No need to mixup the labels too, we dont use them here
    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):

        # The mixup happens here
        if use_mixup == True:
            alpha = float(config.alpha)
            data, aug1, data_f, aug1_f = mixup_data(data, aug1, data_f, aug1_f, alpha = alpha)

        data = data.float().to(device) #, labels.long().to(device) # data: [128, 1, 178], labels: [128]
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

        """Produce embeddings"""
        # print("Data shape:", data.shape)
        # print("data_f", data_f.shape)

        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config.target_batch_size, config.Context_Cont.temperature,
                                       config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

        # print("Here is a problem with the dimensions: ")
        # print(h_t.shape, h_t_aug.shape)
        
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

        # This is the pair loss
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam*(loss_t + loss_f) + (1 - lam)*loss_c # l_TF 

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
    
    print('Pretraining: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss, loss_t, loss_f, l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    wandb.log({"pre_train/loss_t" : loss_t,
                   "pre_train/loss_f" : loss_f,
                   "pre_train/l_TF" : l_TF,
                   "pre_train/loss_c" : loss_c, 
                   "pre_train/loss" : ave_loss
                   }, 
                   step = step
                  ) # Is this okay here? 

    return ave_loss


def model_finetune(model, model_optimizer, val_dl, config, device, training_mode, classifier=None, classifier_optimizer=None):
    global labels, pred_numpy, fea_concat_flat
    model.train()
    classifier.train()

    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))

    # param_size = 0
    # for param in classifier.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in classifier.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('classifier size: {:.3f}MB'.format(size_all_mb))

    total_loss = []
    total_acc = []
    total_auc = []  # it should be outside of the loop
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    feas = np.array([])

    #print("I am here")
    for data, labels, aug1, data_f, aug1_f in val_dl:
        #print("I am in the loop")
        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        verbose = False
        if verbose == True:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a  # free inside reserved
            print(f"Total memory: {t // 1024**2} MB")
            print(f"Reserved: {r // 1024**2} MB")
            print(f"allocated: {a // 1024**2} MB")
            print(f"Free: {f // 1024**2} MB")
        # print("Memory usage of X_train in MB: ", data.element_size() * data.nelement()//1024**2 )
        # print("Memory usage of y_train in MB: ", data_f.element_size() * data_f.nelement()//1024**2 )
        # print("Memory usage of X_train in MB: ", aug1.element_size() * aug1.nelement()//1024**2 )
        # print("Memory usage of y_train in MB: ", aug1_f.element_size() * aug1_f.nelement()//1024**2 )
 
        """if random initialization:"""
        model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        nt_xent_criterion = NTXentLoss_poly(device, config.target_batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        #print("ht shape", h_t.shape)
        #print("ht_aug shape", h_t_aug.shape)

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)

        # l_TF = nt_xent_criterion(z_t, z_f)

        # l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), \
        #                nt_xent_criterion(z_t_aug, z_f_aug)
        # loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3) #


        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        # print(fea_concat.shape, "fccc")
        # sdcsdc
        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        
        #print("Fea_concat_flat.shape", fea_concat_flat.shape)
        #print("Labels: ", labels)

        loss_p = criterion(predictions, labels)

        lam = 0.1
        loss = loss_p  + lam*(loss_t + loss_f) # + l_TF # I don't think this is right

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()

        onehot_label = F.one_hot(labels, num_classes = config.num_classes_target)
        pred_numpy = predictions.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr" )
        except:
            auc_bs = np.float64(0)
        prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)

        total_acc.append(acc_bs)
        total_auc.append(auc_bs)
        total_prc.append(prc_bs)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        if training_mode != "pre_train":
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            feas = np.append(feas, fea_concat_flat.data.cpu().numpy())

    wandb.log({"fine_tune/losses/loss_t" : loss_t,
                   "fine_tune/losses/loss_f" : loss_f,
                   "fine_tune/losses/loss_c" : loss_p, 
                   "fine_tune/losses/loss" : loss
                   },
                   step = step
                  ) # Is this okay here? 

    feas = feas.reshape([len(trgs), -1])  # produce the learned embeddings

    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)

    print("Predictions during fine_tune: ", pred_numpy)
 
    precision = precision_score(labels_numpy, pred_numpy, average='macro', )
    recall = recall_score(labels_numpy, pred_numpy, average='macro', )
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', )
    ave_loss = torch.tensor(total_loss).mean()
    ave_acc = torch.tensor(total_acc).mean()
    ave_auc = torch.tensor(total_auc).mean()
    ave_prc = torch.tensor(total_prc).mean()

    wandb.log({
        "fine_tune/metrics/ave_loss" : ave_loss,
        "fine_tune/metrics/ave_acc" : ave_acc*100,
        "fine_tune/metrics/precision" : precision*100,
        "fine_tune/metrics/recall" : recall*100,
        "fine_tune/metrics/F1" : F1*100,
        "fine_tune/metrics/ave_auc" : ave_auc*100, 
        "fine_tune/metrics/ave_prc" : ave_prc*100
    }, 
    step = step )

    print(' Finetune: loss = %.4f| Acc=%.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f| AUROC=%.4f | AUPRC = %.4f'
          % (ave_loss, ave_acc*100, precision * 100, recall * 100, F1 * 100, ave_auc * 100, ave_prc *100))

    return ave_loss, feas, trgs, F1

def model_test(model,  test_dl, config,  device, training_mode, classifier=None, classifier_optimizer=None):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss() # the loss for downstream classifier
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _,data_f, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions_test = classifier(fea_concat)
            #print("Predictions test shape: ", predictions_test.shape)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)

            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels, num_classes = config.num_classes_target)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,
                                   average="macro", multi_class="ovr")
            except:
                auc_bs = np.float64(0)
            #print("pred numpy shape: ", pred_numpy.shape)
            #print("The other shape", onehot_label.detach().cpu().numpy().shape)
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
            pred_numpy = np.argmax(pred_numpy, axis=1)
            

            total_acc.append(acc_bs)
            total_auc.append(auc_bs)
            total_prc.append(prc_bs)

            total_loss.append(loss.item())
            pred = predictions_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
        print("Predictions during test: ", pred_numpy)
        # print("Real labels during test", labels)

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
    # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    wandb.log({
        "test/metrics/Acc" : acc*100,
        "test/metrics/precision" : precision*100,
        "test/metrics/recall" : recall*100,
        "test/metrics/F1" : F1*100,
        "test/metrics/total_auc" : total_auc*100, 
        "test/metrics/total_prc" : total_prc*100
    })

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    print('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'
          % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100, total_prc*100))
    emb_test_all = torch.concat(tuple(emb_test_all))
    return total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance
