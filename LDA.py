import numpy as np
from data_processing_general import *

# Compute probability P(y=0) and P(y=1)
# Input is a cleaned dataset (ie, wine_data, cancer_data)
# Return an array [P(y=0), P(y=1)]
class LDA:
    def probability(dataIn):
        positive = 0
        negative = 0
        for i in range(dataIn.shape[0]):
            if dataIn[i, -1] == 0:
                negative = negative + 1
            else:
                positive = positive + 1
        prob_negative = negative / (positive + negative)
        prob_positive = positive / (positive + negative)

        return np.array([prob_negative, prob_positive])


    # Compute μ(mean) for class 0 or 1
    # Input is one sub-datasets (wine_data_0 or wine_data_1 or cancer_data_0 or cancer_data_1)
    # Return an int that is the mean of this data set
    def mean(dataIn):
        rows = dataIn.shape[0]
        columns = dataIn.shape[1]
        # Creat an array to store row means
        # array shape is: 1 by num of columns(features)
        mean_vector = np.zeros((1, columns))

        for i in range(columns):  # by column
            sum = 0
            for j in range(rows):  # by row
                sum = sum + dataIn[j][i]
            mean_vector[0][i] = sum / rows
        return mean_vector


# Compute Sigma(Summation) for the dataset
# mean_0: wine_data_0 or cancer_data_0
# mean_1: cancer_data_1 or cancer_data_1
# dataIn: wine_data_pure (without last column)
#      or cancer_data_pure (without first and last column)
# Y: Y_wine or Y_cancer
# Return a 2D array that is the covariance matrix(expectation)
    def sigma(mean_0, mean_1, dataIn, Y):
        # Use a copy to avoid original dataset being modified
        dataSet = dataIn.copy()

        for i in range(dataSet.shape[0]):
            if (Y[i] == 0):
                dataSet[i] = dataSet[i] - mean_0
            else:
                dataSet[i] = dataSet[i] - mean_1
        summation = np.matmul(np.transpose(dataSet), dataSet)
        return summation / (dataSet.shape[0] - 2)

# fit function
# X: original data ie, wine_data or cancer_data
# X_0: wine_data_0 or cancer_data_0
# X_1: wine_data_1 or cancer_data_1
# X_pure:  wine_data_pure (without last column)
#          or cancer_data_pure (without first and last column)
# Y: true labels(ground truth) -> either Y_wine or Y_cancer
# Return weights as a [2,1] array
# Note: LDA Model: log-odds ratio = w_0 + w_1 * Xi
    def fit_LDA(X, X_0, X_1, X_pure, Y):
        # Step-1: get P(y)
        probs = LDA.probability(X)
        # Step-2: get means μ
        mean_0_trans = LDA.mean(X_0)             #1,11
        mean_1_trans = LDA.mean(X_1)             #1,11
        # Step-3: get sigma(summation)
        sig = LDA.sigma(mean_0_trans, mean_1_trans, X_pure, Y)
        # Step-4: compute log-odds ration
        log_p = np.log(probs[1] / probs[0])
        mean_1 = np.transpose(mean_1_trans)   #11,1
        mean_0 = np.transpose(mean_0_trans)   #11,1
        sig_inv = np.linalg.inv(sig)          #11,11
        mean_diff = mean_1 - mean_0           #11,1
        #weight_0
        w_0 = log_p - np.matmul([1/2],np.matmul(np.matmul(mean_1_trans, sig_inv), mean_1)) + \
                np.matmul([1/2],np.matmul(np.matmul(mean_0_trans, sig_inv), mean_0))
        #weight_1
        w_1 = np.matmul(sig_inv,mean_diff)

        #print("w_0",w_0)
        #print("w_0_shape", w_0.shape) # 1，1
        #print("w_1",w_1)
        #print("w_1_shape", w_1.shape) # 11，1
        return np.array([w_0,w_1])


# predict function
# weights: [w_0,w_1] in array form
# X_pure:  wine_data_pure (without last column)
#          or cancer_data_pure (without first and last column)
# Return predictions(log-odds ratios) ie, as an array(rows,1)
    def predict_LDA(weights, X_pure):
        w_0 = weights[0]
        w_1 = weights[1]

        predictions = np.zeros((X_pure.shape[0], 1))
        for i in range(X_pure.shape[0]):
            log_odds = w_0 + np.matmul(X_pure[i],w_1)
            if log_odds > 0:
                predictions[i] = 1
            else: predictions[i] = 0

        return predictions

# evaluate accuracy function
# Y: true labels (Y_wine or Y_cancer)
# Y_predicted: target labels (outputs of predict function)
# Return accuracy score
    def evaluate_acc(Y,Y_predicted):
        match = 0
        total = Y_predicted.shape[0]
        for i in range(total):
            if Y_predicted[i] == Y[i]:
                match = match + 1
        accuracy = match / total
        return accuracy

