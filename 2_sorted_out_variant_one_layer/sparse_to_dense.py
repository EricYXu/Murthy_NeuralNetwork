"""
sparse_to_dense.py 

Script to obtain test accuracy plots of already-trained
neural networks with different types and degrees of sparsity.

Usage:
    python3 sparse_to_dense.py

"""

import sys
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace 
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    seed = 1
    network.set_random_mode(True, seed)
    store_image = True

    odorant_dim = 100
    measurement_dim = 50
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
    num_data = 10000
    dense_training_frac = 0.2
    num_training = int(num_data*(1 - dense_training_frac))
    is_balanced = True
    threshold = 0.1
    bounds = (0,1)
    test_batch_size = 20
    test_iteration_count = 100
    train_test_iteration_count = 1
    dense_dataset = network.generate_dataset(odorant_dim, measurement_dim, device, num_data, dense_training_frac, is_balanced, threshold, bounds, None)
    dense_test_conc = dense_dataset.test_conc
    dense_test_labels = dense_dataset.test_labels

    sparsities = np.arange(0, odorant_dim + 1, 5)  
    original_accuracy = torch.ones(len(sparsities)) 
    sparsity_accuracies = torch.zeros(len(sparsities))

    for idx, sparsity in enumerate(sparsities):
        print(f"Started Run with Sparsity={sparsity}...")

        sparse_training_frac = 0.8
        sparse_dataset = network.generate_dataset(odorant_dim, measurement_dim, device, num_data, sparse_training_frac, is_balanced, threshold, bounds, sparsity)
        sparse_train_conc = sparse_dataset.train_conc
        sparse_train_labels = sparse_dataset.train_labels

        init_parameters = SimpleNamespace()
        init_parameters.m = odorant_dim
        init_parameters.n = measurement_dim
        init_parameters.seed = seed
        init_parameters.W = torch.randn(measurement_dim,odorant_dim).to(device)
        enose_network = network.enose(init_parameters)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(enose_network.parameters(), lr = 0.001)
        lr = 0.001
        batch_size = 10
        steps = 10000
        _ = torch.zeros(steps)
        test_interval = 50 
        print_interval = 10000 
        test_batch_size = 20 
        __ = torch.zeros(steps // test_interval)
        _, __ = network.train_model(
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
            train_test_iteration_count,
            device
        )

        sparsity_accuracy = torch.mean(network.eval_model_accuracy(enose_network, dense_test_conc, dense_test_labels, num_data, num_training, test_batch_size, test_iteration_count, device))
        if idx == 0:
            original_accuracy = sparsity_accuracy * original_accuracy 
        else:
            sparsity_accuracies[idx - 1] = sparsity_accuracy
        print(f"Finished Run with Sparsity={sparsity}...")

    folder = "./eric_sparse_to_dense_figures"
    plt.figure(figsize=(8, 4))
    plt.plot(sparsities, original_accuracy, label='Test Accuracy w/o Sparsity')
    plt.plot(sparsities, sparsity_accuracies, label='Test Accuracy w/ Sparsity')
    plt.xlabel("Sparsity")
    plt.ylabel(f"Mean Test Accuracy over {test_iteration_count} Iterations")
    plt.title(f"Mean Accuracy vs. Sparsity")
    plt.ylim(0, 1.10) 
    if store_image:
        plt.savefig(f"{folder}/accuracy_by_sparsity_odorantdim={odorant_dim}_measuredim={measurement_dim}.pdf")
    plt.close()

if __name__ == "__main__":
    sys.exit(main())