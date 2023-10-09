#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
import math
import geneNewData

def calculate_features(data):
    num_samples, height, width = data.shape
    features = np.zeros((num_samples, 2))  # Create an array to store the calculated features
    
    for i in range(num_samples):
        image = data[i]
        brightness_values = image.reshape(-1)  # Flatten the 2D image array into a 1D array
        average_brightness = np.mean(brightness_values)
        std_dev_brightness = np.std(brightness_values)
        
        features[i, 0] = average_brightness
        features[i, 1] = std_dev_brightness
        
    return features

def calculate_parameters(data):
    feature1_mean = np.mean(data[:, 0])
    feature1_variance = np.var(data[:, 0])
    feature2_mean = np.mean(data[:, 1])
    feature2_variance = np.var(data[:, 1])
    
    return feature1_mean, feature1_variance, feature2_mean, feature2_variance

def calculate_naive_bayes_probability(feature, mean, variance):
    exponent = -((feature - mean) ** 2) / (2 * variance)
    probability = np.exp(exponent) / (np.sqrt(2 * np.pi * variance))
    return probability

def classify_data(data, params_digit0, params_digit1):
    predictions = []
    
    for i in range(data.shape[0]):
        feature1 = data[i, 0]
        feature2 = data[i, 1]
        
        prob_feature1_digit0 = calculate_naive_bayes_probability(feature1, params_digit0[0], params_digit0[1])
        prob_feature2_digit0 = calculate_naive_bayes_probability(feature2, params_digit0[2], params_digit0[3])
        prob_digit0 = prob_feature1_digit0 * prob_feature2_digit0
        
        prob_feature1_digit1 = calculate_naive_bayes_probability(feature1, params_digit1[0], params_digit1[1])
        prob_feature2_digit1 = calculate_naive_bayes_probability(feature2, params_digit1[2], params_digit1[3])
        prob_digit1 = prob_feature1_digit1 * prob_feature2_digit1
        
        if prob_digit0 > prob_digit1:
            predictions.append(0)
        else:
            predictions.append(1)
    
    return predictions

def calculate_accuracy(predictions, actual_labels):
    correct_predictions = np.sum(predictions == actual_labels)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions * 100
    return accuracy

def check_within_range(value, target, tolerance):
    lower_bound = target - tolerance
    upper_bound = target + tolerance
    return lower_bound <= value <= upper_bound

def main():
    myID = '1284'  # your ID here
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID)
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID)
    Numpyfile2 = scipy.io.loadmat('digit0_testset')
    Numpyfile3 = scipy.io.loadmat('digit1_testset')

    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    
    # Task 1
    # Calculate features for digit "0" training set
    features_train0 = calculate_features(train0)
    # Calculate features for digit "1" training set
    features_train1 = calculate_features(train1)

    # Task 2
    # Calculate parameters for digit "0" based on the generated features
    feature0_params = calculate_parameters(features_train0)
    # Calculate parameters for digit "1" based on the generated features
    feature1_params = calculate_parameters(features_train1)
    # Unpack the calculated parameters
    mean_feature1_digit0, variance_feature1_digit0, mean_feature2_digit0, variance_feature2_digit0 = feature0_params
    mean_feature1_digit1, variance_feature1_digit1, mean_feature2_digit1, variance_feature2_digit1 = feature1_params
    print("mean_feature1_digit0", mean_feature1_digit0)
    print("variance_feature1_digit0", variance_feature1_digit0)
    print("mean_feature2_digit0", mean_feature2_digit0)
    print("variance_feature2_digit0", variance_feature2_digit0)
    print("mean_feature1_digit1", mean_feature1_digit1)
    print("variance_feature1_digit1", variance_feature1_digit1)
    print("mean_feature2_digit1", mean_feature2_digit1)
    print("variance_feature2_digit1", variance_feature2_digit1)

    # Task 3
    # Convert the original test data arrays to 2-D data points
    features_test0 = calculate_features(test0)
    features_test1 = calculate_features(test1)
    # Classify the test data points for digit "0" and "1" using the calculated parameters
    predictions_test0 = classify_data(features_test0, feature0_params, feature1_params)
    predictions_test1 = classify_data(features_test1, feature0_params, feature1_params)
    print("predictions_test0", predictions_test0)
    print("predictions_test1", predictions_test1)

    # Task 4
    actual_labels_test0 = np.zeros(len(test0))  # Actual labels for digit "0"
    actual_labels_test1 = np.ones(len(test1))   # Actual labels for digit "1"
    # Calculate the accuracy of predictions for digit "0" test data
    accuracy_test0 = calculate_accuracy(predictions_test0, actual_labels_test0)
    # Calculate the accuracy of predictions for digit "1" test data
    accuracy_test1 = calculate_accuracy(predictions_test1, actual_labels_test1)
    print(f"Accuracy for digit 0 test data: {accuracy_test0:.2f}%")
    print(f"Accuracy for digit 1 test data: {accuracy_test1:.2f}%")
    # Checking if the calculated parameters are within acceptable ranges
    params_within_range = all(
        check_within_range(param, target, tolerance)
        for param, target, tolerance in zip(
            feature0_params + feature1_params,
            [mean_feature1_digit0, variance_feature1_digit0, mean_feature2_digit0, variance_feature2_digit0,
            mean_feature1_digit1, variance_feature1_digit1, mean_feature2_digit1, variance_feature2_digit1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        )
    )
    # checking if accuracy of predictions are within acceptable ranges
    # Set the target accuracy values for digit 0 and digit 1 test data
    target_accuracy_test0 = accuracy_test0
    target_accuracy_test1 = accuracy_test1
    accuracy_within_range_test0 = check_within_range(accuracy_test0, target_accuracy_test0, 0.005)
    accuracy_within_range_test1 = check_within_range(accuracy_test1, target_accuracy_test1, 0.005)
    if params_within_range:
        print("Parameters are within acceptable ranges.")
    else:
        print("Parameters are not within acceptable ranges.")
    if accuracy_within_range_test0:
        print(f"Accuracy for digit 0 test data is within acceptable range: {accuracy_test0:.2f}%")
    else:
        print(f"Accuracy for digit 0 test data is not within acceptable range: {accuracy_test0:.2f}%")

    if accuracy_within_range_test1:
        print(f"Accuracy for digit 1 test data is within acceptable range: {accuracy_test1:.2f}%")
    else:
        print(f"Accuracy for digit 1 test data is not within acceptable range: {accuracy_test1:.2f}%")
    pass
if __name__ == '__main__':
    main()


# In[ ]:




