"""
Script to more formally obtain training loss vs. step and test accuracy vs. step plots with different 
degrees of sparcity in the training dataset. 
"""

# Imports
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace 
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# Iterate over different sparsities
list_of_sparsities = np.arange(0,99,10).tolist()
for sparsity_index in list_of_sparsities:
    print(f"Starting Run with Sparsity Index={sparsity_index}...")

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    m = 50 # input dimension
    n = 100 # measurement dimension
    seed = 1
    network.set_random_mode(True, seed)

    init_parameters = SimpleNamespace()
    init_parameters.m = m 
    init_parameters.n = n 
    init_parameters.seed = seed
    init_parameters.W = torch.randn(n,m).to(device)
    store_image = True

    # Creates the network and establishes parameters
    enose_network = network.enose(init_parameters)
    num_data=10000
    sparse_training_frac=0.8
    dense_training_frac=0.2
    is_balanced=True
    threshold=0.1
    x_hat = threshold
    num_training = int(num_data*sparse_training_frac)
    bounds = (0,1)

    # Creates sparse dataset
    sparse_dataset = network.generate_dataset(m, n, device, num_data, sparse_training_frac, is_balanced, threshold, bounds, sparsity_index)
    sparse_train_conc = sparse_dataset.train_conc
    sparse_train_labels = sparse_dataset.train_labels

    # Creates dense dataset
    dense_dataset = network.generate_dataset(m, n, device, num_data, dense_training_frac, is_balanced, threshold, bounds, None)
    dense_test_conc = dense_dataset.test_conc
    dense_test_labels = dense_dataset.test_labels
    test_batch_size = 50

    # Trains the network on the sparse dataset
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
        sparse_train_conc,
        sparse_train_labels,
        dense_test_conc,
        dense_test_labels,
        num_data,
        num_training,
        steps,
        batch_size,
        test_interval,
        print_interval,
        test_batch_size,
        100,
        device
    )

    # Get mean accuracy
    num_step = len(accuracy_vec)
    mean_accuracy = torch.mean(accuracy_vec[num_step//2:-1])

    # Saves plot of training loss
    folder = "./eric_sparse_to_dense_figures"
    plt.plot(loss_vec)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time (lr=" + str(lr) + ")")
    if store_image:
        plt.savefig(f"{folder}/loss_sparsity={sparsity_index}_m={m}_n={n}.pdf") 
    plt.close()

    # Saves plot of test accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(torch.arange(0, steps, 50), accuracy_vec, label='Test Accuracy')
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.title(f"Odor Detection Accuracy Over Training, with a mean of {mean_accuracy:.3f}")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    if store_image:
        plt.savefig(f"{folder}/accuracy_sparsity={sparsity_index}_m={m}_n={n}.pdf")
    plt.close()

    # Saves plot of learned function and threshold
    W = enose_network.W.cpu().numpy()
    W1 = enose_network.fc1.weight.detach().cpu().numpy()
    b1 = enose_network.fc1.bias.detach().cpu().numpy()
    W_h = W1 @ W
    x1_hat_dagger = -b1[0]/W_h[0,0].item()
    x = torch.linspace(-1, 2, 200)
    y = torch.sigmoid( W_h[0][0]*x + b1 )
    y_2 = torch.sigmoid( 200*(x - x_hat) )
    plt.plot(x, y, label =' learned function')
    plt.plot(x, y_2, label = ' desired shape')
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    plt.plot(x_hat, 0.5, 'o', color='red', label=r'$\hat{x}_1$')
    plt.plot(x1_hat_dagger, 0.5, 'o', color='purple', label=r'$\hat{x}_1^\dagger$')
    plt.title(fr'$\hat{{x}}_1 = {x_hat:.2f} $, with the learned $\hat{{x}}_1^\dagger = {x1_hat_dagger:.4f} $')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$\operatorname{Sigmoid}\left[ W_{\operatorname{tt},1} x_1 + b_1 ) \right]$')
    plt.legend()
    plt.grid(True)
    if store_image:
        plt.savefig(f"{folder}/learned_func_threshold_sparsity={sparsity_index}_m={m}_n={n}.pdf", bbox_inches='tight')
    plt.close()

    print(f"Finished Run with Sparsity Index={sparsity_index}...")









