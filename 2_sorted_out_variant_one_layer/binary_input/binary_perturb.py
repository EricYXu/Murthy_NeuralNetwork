import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from types import SimpleNamespace 
sys.path.append('../')
import network

"""
sparse_to_dense.py

Script to determine the effects of weight perturbation on a network trained on binary data.

Usage:
    python binary_perturb.py

"""


# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../binary_input_figures"
seed = 1
network.set_random_mode(True, seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 


# ===== MODEL SETUP =====
m = 50
n = 100
init_parameters = SimpleNamespace()
init_parameters.m = m 
init_parameters.n = n 
init_parameters.seed = seed
init_parameters.W = torch.randn(n,m).to(device)


# ===== MODEL CREATION =====
enose_network = network.enose(init_parameters)


# ===== DATASET SETUP & CREATION =====
num_data=10000
training_frac=0.8
sparsity = 0.99
num_training = int(num_data*training_frac)
dataset = network.generate_binary_dataset(m,n,device,num_data,training_frac,sparsity)
train_conc = dataset.train_conc
train_labels = dataset.train_labels
test_conc = dataset.test_conc
test_labels = dataset.test_labels
test_batch_size = 20
test_iteration_count = 100


# ===== MODEL TRAINING =====
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(enose_network.parameters(), lr = 0.001)
lr = 0.001
batch_size = 10
steps = 10000
loss_vec = torch.zeros(steps)
test_interval = 50 
print_interval = 10000 
test_batch_size = 20 
accuracy_vec = torch.zeros(steps // test_interval)
loss_vec, accuracy_vec = network.train_model(
    enose_network,
    criterion,
    optimizer,
    lr,
    train_conc,
    train_labels,
    test_conc,
    test_labels,
    num_data,
    num_training,
    steps,
    batch_size,
    test_interval,
    print_interval,
    test_batch_size,
    test_iteration_count,
    device
)


# ===== PRE-PERTURBATION STATISTICS =====
num_step = len(accuracy_vec)
mean_accuracy = torch.mean(accuracy_vec[num_step//2:-1])
print("Mean Accuracy: ", mean_accuracy)
first_exceed = 0
for step in range(1,len(accuracy_vec)+1):
    if torch.mean(accuracy_vec[0:step]) >= 0.90:
        first_exceed = step
        break


# ===== PRE-PERTURBATION PLOTS =====
plt.figure(figsize=(8, 4))
plt.plot(torch.arange(0, steps, 50), accuracy_vec, label='Test Accuracy')
plt.xlabel("Training Step")
plt.ylabel("Accuracy")
plt.title(f"Odor Detection Accuracy Over Training with Sparsity={sparsity}, with a mean of {mean_accuracy:.3f}\n90% Mean Accuracy Achieved at Step {first_exceed}")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
if store_results:
    plt.savefig(f"{output_folder}/test_accuracy_during_training_sparsity={sparsity}.pdf")
plt.show()
