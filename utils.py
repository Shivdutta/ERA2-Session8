#from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import numpy as np
import math
from typing import NoReturn
#from torchsummary import summary
#from tqdm import tqdm
#import torch.nn.functional as F

def get_cifar_property(images, operation):
    """
    Get the property on each channel of the CIFAR
    """
    param_r = eval('images[:, 0, :, :].' + operation + '()')
    param_g = eval('images[:, 1, :, :].' + operation + '()')
    param_b = eval('images[:, 2, :, :].' + operation + '()')
    return param_r, param_g, param_b

def get_cifar_statistics(data_set, data_set_type='Train'):
    """
    Function to get the statistical information of the CIFAR dataset
    :param data_set: Training set of CIFAR
    :param data_set_type: Training or Test data
    """
    # Images in the dataset
    images = [item[0] for item in data_set]
    images = torch.stack(images, dim=0).numpy()

    # Calculate mean over each channel
    mean_r, mean_g, mean_b = get_cifar_property(images, 'mean')

    # Calculate Standard deviation over each channel
    std_r, std_g, std_b = get_cifar_property(images, 'std')

    # Calculate min value over each channel
    min_r, min_g, min_b = get_cifar_property(images, 'min')

    # Calculate max value over each channel
    max_r, max_g, max_b = get_cifar_property(images, 'max')

    # Calculate variance value over each channel
    var_r, var_g, var_b = get_cifar_property(images, 'var')

    print(f'[{data_set_type}]')
    print(f' - Total {data_set_type} Images: {len(data_set)}')
    print(f' - Tensor Shape: {images[0].shape}')
    print(f' - min: {min_r, min_g, min_b}')
    print(f' - max: {max_r, max_g, max_b}')
    print(f' - mean: {mean_r, mean_g, mean_b}')
    print(f' - std: {std_r, std_g, std_b}')
    print(f' - var: {var_r, var_g, var_b}')

    # Let's visualize some of the images
    plt.imshow(np.transpose(images[1].squeeze(), (1, 2, 0)))


def plot_accuracy_losses(train_losses,train_acc,test_losses,test_acc):
  """
    Plots the training and test losses along with training and test accuracies.

    Args:
        train_losses (list): List of training losses.
        train_acc (list): List of training accuracies.
        test_losses (list): List of test losses.
        test_acc (list): List of test accuracies.

    Returns:
        None
  """
  t = [t_items.item() for t_items in train_losses]
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(t)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def display_data_samples(data_set, number_of_samples: int, classes: list, dataset: str="CIFAR"):
    """
    Function to display samples for data_set
    :param data_set: Train or Test data_set
    :param number_of_samples: Number of samples to be displayed
    """
    # Get batch from the data_set
    batch_data = []
    batch_label = []
    for count, item in enumerate(data_set):
        if not count <= number_of_samples:
            break
        batch_data.append(item[0])
        batch_label.append(item[1])
    batch_data = torch.stack(batch_data, dim=0).numpy()

    # Plot the samples from the batch
    fig = plt.figure()
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples/x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        plt.tight_layout()
        if dataset == "MNIST":
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        else:
            plt.imshow(np.transpose(batch_data[i].squeeze(), (1, 2, 0)))
        plt.title(classes[batch_label[i]])
        plt.xticks([])
        plt.yticks([])

def plot_data(data, classes, inv_normalize, number_of_samples=10, dataset="CIFAR"):
    """
    Function to plot images with labels
    :param data: List[Tuple(image, label)]
    :param number_of_samples: Number of images to print
    """
    fig = plt.figure(figsize=(8, 5))

    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples/x_count)

    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        if dataset == "MNIST":
            plt.imshow(data[i][0].squeeze(0).to('cpu'), cmap='gray')
        else:
            img = data[i][0].squeeze().to('cpu')
            img = inv_normalize(img)
            plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])

def display_loss_and_accuracies(train_losses: list,
                                train_acc: list,
                                test_losses: list,
                                test_acc: list,
                                plot_size: tuple = (10, 10)) -> NoReturn:
    """
    Function to display training and test information(losses and accuracies)
    :param train_losses: List containing training loss of each epoch
    :param train_acc: List containing training accuracy of each epoch
    :param test_losses: List containing test loss of each epoch
    :param test_acc: List containing test accuracy of each epoch
    :param plot_size: Size of the plot
    """
    # Create a plot of 2x2 of size
    fig, axs = plt.subplots(2, 2, figsize=plot_size)

    # Plot the training loss and accuracy for each epoch
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    # Plot the test loss and accuracy for each epoch
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()
    
    # List to store misclassified Images
    misclassified_data = []
    
    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:
            
            # Migrate the data to the device
            data, target = data.to(device), target.to(device)
            
            # Extract single image, label from the batch
            for image, label in zip(data, target):
                
                # Add batch dimension to the image
                image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(image)
                
                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)
                
                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data
