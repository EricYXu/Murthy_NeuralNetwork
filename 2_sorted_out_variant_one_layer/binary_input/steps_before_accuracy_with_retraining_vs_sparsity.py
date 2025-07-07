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
steps_before_accuracy_with_retraining_vs_sparsity.py

Script to demonstrate the relationship between training dataset sparsity and 
the number of training steps required to reach at least [threshold_accuracy] trial-averaged testing accuracy. 
Then the odorant-to-measurement mapping is perturbed, and the number of retraining steps required to reach [threshold_accuracy] 
trial-averaged testing accuracy is recorded.
"""

# ===== PROGRAM SPECIFICATIONS =====
store_results = True
output_folder = "../binary_input_figures"
seed = 1
network.set_random_mode(True, seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 


# ===== TRACKING TRAINING & RETRAINING STEPS BEFORE REACHING THRESHOLD ACCURACY =====
threshold_accuracy = 0.95
sparsities_to_test = [0.1 * i for i in range(1, 10)]
x_vals = [0,1]
training_steps_before_threshold = []
retraining_steps_before_threshold = []
noise_type = "multiplicative"


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
    test_interval = 1
    print_interval = 1000 
    test_batch_size = 20
    training_loss_vec = torch.zeros(steps)
    training_accuracy_vec = torch.zeros(steps // test_interval)
    training_loss_vec, training_accuracy_vec = network.train_model(
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


    # ===== TRAINING STEPS BEFORE EXCEEDING THRESHOLD ACCURACY =====
    training_exceed = 0
    for step in range(1,len(training_accuracy_vec)+1):
        if torch.mean(training_accuracy_vec[0:step]) >= threshold_accuracy:
            training_exceed = step
            break
    training_steps_before_threshold.append(training_exceed)
    print("Training Exceed", training_exceed)


    # ===== PERTURB THE NETWORK =====
    if noise_type == "additive":
        enose_network.perturb_weights(True, 0, 1, False)
    elif noise_type == "multiplicative":
        enose_network.perturb_weights(True, 0, 1, True)


    # ===== MODEL TRAINING =====
    retraining_loss_vec = torch.zeros(steps)
    retraining_accuracy_vec = torch.zeros(steps // test_interval)
    retraining_loss_vec, retraining_accuracy_vec = network.train_model(
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

    # ===== RETRAINING STEPS BEFORE EXCEEDING THRESHOLD ACCURACY =====
    retraining_exceed = 0
    for step in range(1,len(retraining_accuracy_vec)+1):
        if torch.mean(retraining_accuracy_vec[0:step]) >= threshold_accuracy:
            retraining_exceed = step
            break
    retraining_steps_before_threshold.append(retraining_exceed)
    print("Retraining Exceed", retraining_exceed)

    print(f"Finished Testing Sparsity={sparsity}.")


# ===== PLOTS =====
plt.figure(figsize=(10, 6))

# ===== PREPROCESSING EXCEED VALUES TO FORMAT PLOTS =====
custom_colors = [
    'red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta',
    'olive', 'teal', 'darkblue', 'darkgreen', 'darkred', 'gold', 'lime', 'indigo', 'maroon', 'navy'
]
for i in range(len(sparsities_to_test)):
    plt.plot(x_vals, [training_steps_before_threshold[i], retraining_steps_before_threshold[i]], color=custom_colors[i], label=f"Sparsity={sparsities_to_test[i]}")

plt.xlabel(f"Before {noise_type} Perturbation (0), After {noise_type} Perturbation (1)")
plt.ylabel("Steps before Trial-Averaged Test Accuracy Reached")
plt.title(f"Steps before Trial-Averaged Test Accuracy of {threshold_accuracy} Reached during Training & Retraining (m={m}, n={n})")
plt.legend()
plt.tight_layout()
if store_results:
    plt.savefig(f"{output_folder}/steps_before_accuracy={threshold_accuracy}_reached_during_training&retraining_w_{noise_type}noise_m={m}_n={n}.pdf")
    print("Saved figure!")
plt.show()