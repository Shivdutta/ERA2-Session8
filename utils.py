#from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import torch
#from torchsummary import summary
#from tqdm import tqdm
#import torch.nn.functional as F

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



# def train(model, device, train_loader, optimizer, epoch,scheduler):
#   """
#     Trains the specified model using the given data loader and optimizer for one epoch.

#     Args:
#         model (torch.nn.Module): The neural network model to train.
#         device (torch.device): The device to perform training on (e.g., "cuda" or "cpu").
#         train_loader (torch.utils.data.DataLoader): DataLoader for training data.
#         optimizer (torch.optim.Optimizer): Optimizer to use for training.
#         epoch (int): Current epoch number.
#         scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler (default: None).

#     Returns:
#         tuple: A tuple containing the training accuracy and losses as lists.
#   """
#   train_losses = []
#   train_acc = []
#   model.train()
#   pbar = tqdm(train_loader)
#   correct = 0
#   processed = 0
#   for batch_idx, (data, target) in enumerate(pbar):
#     # get samples
#     data, target = data.to(device), target.to(device)

#     # Init
#     optimizer.zero_grad()
#     # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
#     # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

#     # Predict
#     y_pred = model(data)

#     # Calculate loss

#     #below 2 lines are for debug
#     #print('y_pred.shape',y_pred.shape)
#     #print('target.shape',target.shape)
#     loss = F.nll_loss(y_pred, target)
#     train_losses.append(loss)
  
#     # Backpropagation
#     loss.backward()
#     optimizer.step()

#     if not  scheduler is None:
#       scheduler.step()

#     # Update pbar-tqdm

#     pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#     correct += pred.eq(target.view_as(pred)).sum().item()
#     processed += len(data)

#     pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
#     train_acc.append(100*correct/processed)
    
#   return train_acc,train_losses  

  
# def test(model, device, test_loader):    
#     """
#     Evaluates the specified model on the test dataset.

#     Args:
#         model (torch.nn.Module): The neural network model to evaluate.
#         device (torch.device): The device to perform evaluation on (e.g., "cuda" or "cpu").
#         test_loader (torch.utils.data.DataLoader): DataLoader for test data.

#     Returns:
#         tuple: A tuple containing the test accuracy and loss as lists.
#     """
#     test_losses = []
#     test_acc = []   
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
    
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

#     test_acc.append(100. * correct / len(test_loader.dataset))
#     return test_acc,test_losses
