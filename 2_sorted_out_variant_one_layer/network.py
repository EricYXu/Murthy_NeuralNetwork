import torch
import torch.distributions.normal as DN
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from types import SimpleNamespace
import random
import numpy as np

# ============ NETWORK ARCHITECTURE ============

class enose(nn.Module):
    def __init__(self, init_parameters, activation = nn.Identity()): # initializes the network params
        super().__init__()
        self.m = init_parameters.m # input dimension
        self.n = init_parameters.n # meansurement dimension
        self.W  = init_parameters.W  ## if init_parameters.W else
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.fc1 = nn.Linear( self.n, 1 ).to(device) # fully-connected layer that takes self.n inputs to produce 1 output
        self.activation = activation # sets the activation 

    def forward(self, x): # runs a forward pass 
        x = torch.matmul(self.W, x) 
        output = F.sigmoid(self.fc1(x.mT)) 
        return output

    def perturb_weights(self, normal_mode, mean, stddev, is_multiplicative):
        """ 
        Perturbs matrix W using an additive/multiplicative noise term that is a scalar or normally distributed.

        Args:
            normal_mode (bool): If True, proceed with generating normally-distributed term. If False, the scalar is specified by the mean paramter.
            self.W (Tensor): The [n x m] odorant-to-measurement matrix. 
            mean (float): The mean parameter for the Normal noise.
            stddev (float): The standard deviation parameter for the Normal noise. 
            is_multiplicative (bool): If True, an expression involving noise term will be multiplied to weight entries.
            is_correlated (bool): If True, then an expression involving the noise term will be multiplied to increase/decrease each weight entry.
            is_increase (bool): If True, then the matrix entries will increase. 
        """
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if not normal_mode:
            if is_multiplicative:
                self.W *= mean
            else: 
                self.W += mean
        elif normal_mode:
            noise = stddev * torch.randn(self.W.shape[0], self.W.shape[1]).to(device)
            if is_multiplicative:
                self.W = self.W * (1 + noise)
            else:
                self.W = self.W + noise

    def set_weights(self, weights):
        """
        Sets the weights of the neural network.

        Args: 
            weights: New version of odorant-to-measurement matrix.
        """

        self.W = weights.clone()

# ============ DATASET AND PRECISION EVALUATION ============

def generate_dataset(m, n, device, num_data=10000, training_frac=0.8, is_balanced=True, threshold=0.1, bounds=(0,1), sparsity=None):
    """
    Generate a balanced/unbalanced synthetic odor concentration data and labels based on a threshold.
    
    Args:
        m (int): Input dimension
        n (int): Output dimension (not used here, just for image folder naming)
        device (torch.device): Computation device
        num_data (int): Total number of samples
        training_frac (float): Fraction of samples used for training
        store_image (bool): If True, creates a folder for saving images
        threshold (float): Threshold for label generation
        bounds (tuple): Lower and upper Bounds to sample values from
        is_balanced (bool): If True, creates an approximately equal number of above-threshold and below-threshold data points
        sparsity_idx (int): Row number where rows at index and below are set to zero 
    
    Returns:
        A SimpleNamespace containing:
            - odor_conc: [m x num_data] full concentration tensor
            - labels: [num_data] binary labels
            - train_conc, train_labels
            - test_conc, test_labels
    """
    input_dim = m
    num_training = int(num_data * training_frac)

    # Generate input data on [bounds[0], bounds[1])
    pre_odor_conc = torch.rand(input_dim, num_data)
    pre_odor_conc = pre_odor_conc * (bounds[1] - bounds[0]) + torch.ones_like(pre_odor_conc) * bounds[0]

    # Checks if user wants a balanced simulated odor dataset; if True, makes the dataset roughly balanced 
    if is_balanced:
        index_tensor = (torch.rand(num_data) > 0.5).int()
        below_tensor = (torch.rand(num_data) * (threshold - bounds[0]) + torch.ones(num_data) * bounds[0]) * index_tensor
        above_tensor = (torch.rand(num_data) * (bounds[1] - threshold) + torch.ones(num_data) * threshold) * (-1 * (index_tensor - 1))
        pre_odor_conc[0, :] = below_tensor + above_tensor
        
    # Randomly chooses [sparsity] random entries to be equal to zero
    if sparsity != None:
        for data_idx in range(num_data):
            sparse_indices = np.random.choice(range(1,m),(sparsity,),replace=False)
            for idx in sparse_indices:
                pre_odor_conc[idx, data_idx] = 0
    
    odor_conc = pre_odor_conc.to(device) 

    # Generate labels based on threshold on first odor channel
    labels = (odor_conc[0, :] > threshold).long().to(device)

    # Split data
    train_conc = odor_conc[:, :num_training]
    train_labels = labels[:num_training]
    test_conc = odor_conc[:, num_training:]
    test_labels = labels[num_training:]

    return SimpleNamespace(
        odor_conc=odor_conc,
        labels=labels,
        train_conc=train_conc,
        train_labels=train_labels,
        test_conc=test_conc,
        test_labels=test_labels
    )

def evaluate_random_batch(model, dataset, test_batch_size=20, threshold=0.5):
    """
    Evaluate a batch of test samples and predict using the given model.

    Args:
        model: A PyTorch model with a .forward() method.
        dataset: A SimpleNamespace with `test_conc` and `test_labels`.
        test_batch_size (int): Number of random samples to draw from test set.
        threshold (float): Classification threshold for binary prediction.

    Returns:
        Tuple of (x_test, y_test, pred_probs, predicted_labels)
    """
    test_conc = dataset.test_conc
    test_labels = dataset.test_labels
    num_test = test_conc.shape[1]

    # Random test indices
    test_idx = torch.randint(0, num_test, (test_batch_size,))
    x_test = test_conc[:, test_idx]
    y_test = test_labels[test_idx].view(-1, 1).float()

    with torch.no_grad():
        pred_test = model(x_test)
        pred_label = (pred_test > threshold).float()

    accuracy = (pred_label == y_test).float().mean().item()

    return accuracy


# ============ TRAINING PROCEDURE ============

def train_model(
    model,
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
):
    """
    Train a binary classifier with BCE loss and Adam optimizer.

    Args:
        model: PyTorch model (must be already moved to device).
        train_conc, train_labels: Training data and labels.
        test_conc, test_labels: Testing data and labels.
        num_data: Total data size.
        num_training: Number of training samples.
        steps: Number of optimization steps.
        batch_size: Mini-batch size for training.
        test_interval: How often to evaluate on test set.
        print_interval: How often to print loss/accuracy.
        test_batch_size: Batch size for testing accuracy.
        lr: Learning rate for Adam optimizer.
        device: torch.device to use.

    Returns:
        loss_vec: [steps] training loss at each step
        accuracy_vec: [steps // test_interval] test accuracy over time
    """
    model.train()
    
    loss_vec = torch.zeros(steps)
    accuracy_vec = torch.zeros(steps // test_interval)

    for step in range(steps):
        idx = torch.randint(0, num_training, (batch_size,))
        x_batch = train_conc[:, idx].to(device)
        y_batch = train_labels[idx].view(-1, 1).float().to(device)

        # Forward & loss
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss_vec[step] = loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        if step % test_interval == 0:
            model.eval()
            with torch.no_grad():
                step_test_accuracy = torch.zeros(test_iteration_count)
                for idx in range(test_iteration_count):
                    test_idx = torch.randint(0, num_data - num_training, (test_batch_size,))
                    x_test = test_conc[:, test_idx].to(device)
                    y_test = test_labels[test_idx].view(-1, 1).float().to(device)

                    pred = model(x_test)
                    predicted_labels = (pred > 0.5).float()
                    accuracy = (predicted_labels == y_test).float().mean().item()
                    step_test_accuracy[idx] = accuracy
                
                accuracy_vec[step // test_interval] = torch.mean(step_test_accuracy)

                if step % print_interval == 0:
                    print(f"[Step {step}] Mean Accuracy over {test_iteration_count} Iterations: {torch.mean(step_test_accuracy):.4f} | Train Loss: {loss.item():.4f}")
            model.train()

    return loss_vec, accuracy_vec


# ============ TESTING PROCEDURE ============
def eval_model_accuracy(
    model,
    test_conc,
    test_labels,
    num_data,
    num_training,
    test_batch_size,
    test_iteration_count,
    device):
    """
    Evaluates the model accuracy by testing it on a series of random test batches.

    Args:
        model: PyTorch model (must be already moved to device).
        test_conc, test_labels: Testing data and labels.
        num_data: Total data size.
        num_training: Number of training samples.
        test_batch_size: Batch size for testing accuracy.
        test_iteration_count: Number of batches to test accuracy. 
        device: torch.device to use.

    Returns:
        Tensor of length test_iteration_count containing test_batch accuracies. 
    """

    model.eval()
    test_accuracy = torch.zeros(test_iteration_count)

    with torch.no_grad():
        for idx in range(test_iteration_count):
            test_idx = torch.randint(0, num_data - num_training, (test_batch_size,))
            x_test = test_conc[:, test_idx].to(device)
            y_test = test_labels[test_idx].view(-1, 1).float().to(device)

            pred = model(x_test)
            predicted_labels = (pred > 0.5).float()
            accuracy = (predicted_labels == y_test).float().mean().item()
            test_accuracy[idx] = accuracy

    return test_accuracy


# ============ SOME BASIC FUNCTIONS ============

def set_random_mode(deterministic=True, seed=1000):
    """
    Toggle between deterministic and random behavior.

    Args:
        deterministic (bool): If True, set fixed seed. If False, use random seed.
        seed (int): The fixed seed to use when deterministic is True.
    """
    if deterministic:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"[Random Mode] Deterministic with seed {seed}")
    else:
        # Use entropy from OS or time to re-randomize
        new_seed = int.from_bytes(os.urandom(4), "little")
        torch.manual_seed(new_seed)
        random.seed(new_seed)
        np.random.seed(new_seed)
        print(f"[Random Mode] Randomized with seed {new_seed}")

