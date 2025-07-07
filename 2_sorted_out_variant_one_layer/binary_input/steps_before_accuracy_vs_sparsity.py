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
steps_before_accuracy_vs_sparsity.py

Script to demonstrate the relationship between training dataset sparsity and 
the amount of training steps required to reach at least [threshold_accuracy] trial-averaged testing accuracy. 

"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../binary_input_figures"
seed = 1
network.set_random_mode(True, seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 


# ===== TRACKING STEPS BEFORE REACHING THRESHOLD ACCURACY =====
threshold_accuracy = 0.95
sparsities_to_test = [0.05 * i for i in range(1, 20)]
steps_before_threshold = []


# ===== MODEL SETUP =====
m = 50
n = 100
init_parameters = SimpleNamespace()
init_parameters.m = m 
init_parameters.n = n 
init_parameters.seed = seed
init_parameters.W = torch.randn(n,m).to(device)

for sparsity in sparsities_to_test:
    print(f"Testing Sparsity={sparsity}...")

    # ===== MODEL CREATION =====
    enose_network = network.enose(init_parameters)


    # ===== DATASET SETUP & CREATION =====
    num_data=10000
    training_frac=0.8
    # sparsity = 0.99 <-- SPECIFIED IN FOR LOOP
    num_training = int(num_data*training_frac)
    dataset = network.generate_binary_dataset(m,n,device,num_data,training_frac,sparsity)
    train_conc = dataset.train_conc
    train_labels = dataset.train_labels
    test_conc = dataset.test_conc
    test_labels = dataset.test_labels
    test_batch_size = 20
    test_iteration_count = 5


    # ===== MODEL TRAINING =====
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(enose_network.parameters(), lr = 0.001)
    lr = 0.001
    batch_size = 10
    steps = 10000
    loss_vec = torch.zeros(steps)
    test_interval = 1
    print_interval = 1000 
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


    # ===== STEPS BEFORE EXCEEDING THRESHOLD ACCURACY =====
    first_exceed = 0
    for step in range(1,len(accuracy_vec)+1):
        if torch.mean(accuracy_vec[0:step]) >= threshold_accuracy:
            first_exceed = step
            break
    steps_before_threshold.append(first_exceed)
    print("First Exceed",first_exceed)
    print(f"Finished Testing Sparsity={sparsity}.")


# ===== PLOTS =====
plt.figure(figsize=(10, 6))
plt.plot(sparsities_to_test, steps_before_threshold, label="Steps")
plt.xlabel("Dataset Sparsity")
plt.ylabel("Steps before Trial-Averaged Test Accuracy Reached")
plt.title(f"Steps before Trial-Averaged Test Accuracy of {threshold_accuracy} Reached vs. Dataset Sparsity (m={m}, n={n})")
plt.legend()
plt.tight_layout()
if store_results:
    plt.savefig(f"{output_folder}/steps_before_accuracy={threshold_accuracy}_reached_vs_dataset_sparsity_m={m}_n={n}.pdf")
    print("Saved figure!")
plt.show()
