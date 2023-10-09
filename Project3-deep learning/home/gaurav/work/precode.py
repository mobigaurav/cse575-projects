import numpy as np 
import random
import pdb
import sys
import time
import os
from Neural import Net
import matplotlib.pyplot as plt

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []

### Cross-Entropy Loss function
def cross_entropy(inputs, labels):
    out_num = labels.shape[0]
    p = np.sum(labels.reshape(1,out_num)*inputs)
    loss = -np.log(p)
    return loss

def init_subset(id):
    from mnist import train_images, train_labels, test_images, test_labels
    #S = int(id) % 1000
    S = int(id) % 210
    
    ### Create a subset of MNIST dataset with only 4 classes
    num_classes = 4
    sub_idx = np.sort(np.random.RandomState(seed=S).permutation(10)[:4])
    train_sub_size = 500
    test_sub_size = 100
    train_images = train_images() #[60000, 28, 28]
    train_labels = train_labels()
    test_images = test_images()
    test_labels = test_labels()

    ### Preprocessing the data
    print('Preparing data......')
    train_images -= int(np.mean(train_images))
    train_images = train_images // int(np.std(train_images))
    test_images -= int(np.mean(test_images))
    test_images = test_images // int(np.std(test_images))
    
    #pdb.set_trace()
    training_data = train_images.reshape(60000, 1, 28, 28)
    testing_data = test_images.reshape(10000, 1, 28, 28)
    ### Generate the New subset of training and testing samples
    sub_training_images, sub_training_labels = subset_extraction(S, sub_idx, train_sub_size, training_data, train_labels, num_classes,train=True)
    sub_testing_images, sub_testing_labels = subset_extraction(S, sub_idx, test_sub_size, testing_data, test_labels, num_classes,train=False)
    return sub_training_images, sub_training_labels, sub_testing_images, sub_testing_labels
    
### Function of creating the subset of MNIST dataset
def subset_extraction(S, idx, sub_size, images, labels, num_classes, train=True):
    temp_img = []
    temp_labels = []
    for i in range(num_classes):
        ind = labels == idx[i]
        A = images[ind,:,:,:]
        A = A[:sub_size,:,:,:]
        temp_img.append(A)
        label_list = [i] * A.shape[0]
        temp_labels += label_list

    sub_images = np.vstack(temp_img)
    sub_labels = np.asarray(temp_labels)
    # shuffle the subset samples
    shuffle_idx = np.random.RandomState(seed=S).permutation(sub_images.shape[0])
    final_images = sub_images[shuffle_idx,:,:]
    final_labels = sub_labels[shuffle_idx]
    final_labels = np.eye(num_classes)[final_labels]
    return final_images, final_labels

"""
The subset of MNIST is created based on the last 4 digits of your ASUID. There are 4 categories and all returned 
samples are preprocessed and shuffled. 
"""
print('Loading data......')
sub_train_images, sub_train_labels, sub_test_images, sub_test_labels = init_subset('1284') # input your ASUID last 4 digits here to generate the subset samples of MNIST for training and testing

net = Net()
epoch = 10            ### Default number of epochs
batch_size = 100      ### Default batch size
num_batch = sub_train_images.shape[0]/batch_size

test_size = sub_test_images.shape[0]      # Obtain the size of testing samples
train_size = sub_train_images.shape[0]    # Obtain the size of training samples

### ---------- ###
"""
Please compile your own evaluation code based on the training code 
to evaluate the trained network.
The function name and the inputs of the function have been predifined and please finish the remaining part.
"""

def evaluate(net, images, labels):
    total_samples = images.shape[0]  # Total number of samples in the dataset
    correct_predictions = 0  # Counter for correctly predicted samples
    total_loss = 0 # Cumulative sum of the losses for each sample

    # Iterate through each sample in the dataset
    for i in range(total_samples):
        x = images[i] # Retrieve the i-th image/sample
        y = labels[i] # Retrieve the i-th label
        output = x
        # Forward pass through each layer in the network
        for layer in net.layers:
            output = layer.forward(output)  # Update 'output' at each layer
        
         # Add the loss for the current sample to the total loss
        total_loss += cross_entropy(output, y)
        # Check if the predicted label matches the true label
        # If yes, increment the counter for correct predictions
        if np.argmax(output) == np.argmax(y):
            correct_predictions += 1
    # Calculate the accuracy as the ratio of correct predictions to total samples
    accuracy = correct_predictions / total_samples
    # Calculate the average loss by dividing the total loss by the number of samples
    average_loss = total_loss / total_samples
    # Return the accuracy and average loss
    return accuracy, average_loss

### Start training process
for e in range(epoch):
    total_acc = 0    
    total_loss = 0
    print('Epoch %d' % e)
    for batch_index in range(0, sub_train_images.shape[0], batch_size):
        # batch input
        if batch_index + batch_size < sub_train_images.shape[0]:
            data = sub_train_images[batch_index:batch_index+batch_size]
            label = sub_train_labels[batch_index:batch_index + batch_size]
        else:
            data = sub_train_images[batch_index:sub_train_images.shape[0]]
            label = sub_train_labels[batch_index:sub_train_labels.shape[0]]
        # Compute the remaining time
        start_time = time.time()
        batch_loss,batch_acc = net.train(data, label)  # Train the network with samples in one batch 
        
        end_time = time.time()
        batch_time = end_time-start_time
        remain_time = (sub_train_images.shape[0]-batch_index)/batch_size*batch_time
        hrs = int(remain_time/3600)
        mins = int((remain_time/60-hrs*60))
        secs = int(remain_time-mins*60-hrs*3600)
        print('=== Iter:{0:d} === Remain: {1:d} Hrs {2:d} Mins {3:d} Secs ==='.format(int(batch_index+batch_size),int(hrs),int(mins),int(secs)))
    
    # Print out the Performance
    train_acc, train_loss = evaluate(net, sub_train_images, sub_train_labels)  # Use the evaluation code to obtain the training accuracy and loss
    test_acc, test_loss = evaluate(net, sub_test_images, sub_test_labels)      # Use the evaluation code to obtain the testing accuracy and loss
    print('=== Epoch:{0:d} Train Size:{1:d}, Train Acc:{2:.3f}, Train Loss:{3:.3f} ==='.format(e, train_size,train_acc,train_loss))

    print('=== Epoch:{0:d} Test Size:{1:d}, Test Acc:{2:.3f}, Test Loss:{3:.3f} ==='.format(e, test_size, test_acc,test_loss))
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

# Plot training accuracy vs epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_acc_list, '-o')
plt.title('Training Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')

# Plot training loss vs epochs
plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_loss_list, '-o')
plt.title('Training Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.tight_layout()
plt.show()

# Plot testing accuracy vs epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(epoch), test_acc_list, '-o')
plt.title('Testing Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Testing Accuracy')

# Plot testing loss vs epochs
plt.subplot(1, 2, 2)
plt.plot(range(epoch), test_loss_list, '-o')
plt.title('Testing Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Testing Loss')
plt.tight_layout()
plt.show()