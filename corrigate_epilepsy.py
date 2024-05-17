import torch
import os
from sklearn.model_selection import train_test_split

targetdata_path = 'datasets/Epilepsy'

finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))  # train.pt
test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))  # test.pt

print("finetune old shape: ", finetune_dataset["samples"].shape)
print("test old shape: ", test_dataset["samples"].shape)

merged = {}
merged["samples"] = torch.cat([finetune_dataset["samples"], test_dataset["samples"]], dim = 0)
merged["labels"] = torch.cat([finetune_dataset["labels"], test_dataset["labels"]], dim = 0)

X_train, X_test, y_train, y_test = train_test_split(merged["samples"], merged["labels"], test_size=0.5, random_state=42)

finetune_dataset = {"samples" : X_train, "labels" : y_train}
test_dataset = {"samples" : X_test, "labels" : y_test}

print("finetune new shape: ", X_train.shape)
print("test new shape: ", X_train.shape)

torch.save(finetune_dataset, os.path.join(targetdata_path, "train.pt"))
torch.save(test_dataset, os.path.join(targetdata_path, "test.pt"))

print('Succesfully saved')