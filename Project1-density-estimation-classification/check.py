
#Given an input array of binary features values for a single feature ,f, and a input array of binary 
#class labels,y, this function computes MLE for P(f=0 | y=1) and P(f=1 | y=1). 
#This will return the computed values in array, in the form [P(f=0|y=1), P(f=1,y=1)]. 

def max_likelihood_estimate(features, labels):
    count_y = 0
    count_f0_y1 = 0
    count_f1_y1 = 0
    for i in range(len(features)):
        if labels[i] == 1:
            count_y += 1
            if features[i] == 0:
                count_f0_y1 += 1
            else:
                count_f1_y1 += 1
    return [float(count_f0_y1)/count_y, float(count_f1_y1)/count_y]


# Example usage:
features = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
labels = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
likelihood_estimates = max_likelihood_estimate(features, labels)
print("likelihood_estimates", likelihood_estimates)

# Incorrect. Your functions does not compute the MLE correctly.
#  The input feature values are: [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
#  The input labels are: [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
# Your function computed the likelihoods as: [0, 0], but the correct likelihoods are: [0.25, 0.75]