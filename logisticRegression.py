import numpy as np
import math

# this is the logistic model
class LogisticRegression:

    def logisticFunc(a):
        return 1.0 / (1 + np.exp(-a))


    # predict method, w=weights, X=training data, returns predicted outcome(s)
    def predict(w, X):
        z = w[0]
        for i in range(X.shape[1]):
            z += w[i + 1] * np.array(X[:, i])
        a = np.array(z)

        return LogisticRegression.logisticFunc(a)


    # cross entropy loss function, w=weights, X=training data, Y=ground truth
    def crossEntropyFunc(w, X, Y):
        y_prediction = LogisticRegression.predict(w, X)
        return -1 * sum(Y * np.log(y_prediction) + (1 - Y) * np.log(1 - y_prediction))


    # gradient function,returns array of gradients
    def grad(w, X, Y):
        y_prediction = LogisticRegression.predict(w, X)
        gra = [0] * (X.shape[1] + 1)
        gra[0] = sum(Y - y_prediction)
        for i in range(X.shape[1]):
            gra[i + 1] = sum(X[:, i] * (Y - y_prediction))

        return gra


    def descent(w_new, w_prev, lr, n, X, Y):
        # print("old weights", w_prev)
        # print("error", crossEntropyFunc(w_prev, X, Y))
        j = 0
        while True:
            w_prev = w_new
            w = [0] * (X.shape[1] + 1)
            for i in range(X.shape[1] + 1):
                w[i] = w_prev[i] + lr * LogisticRegression.grad(w_prev, X, Y)[i]

            w_new = np.array(w)

            # print("new weights", w_new)
            # print("new error", crossEntropyFunc(w_prev, X, Y))

            if j > n:
                return w_new
            j += 1

            result = 0
            for i in range(X.shape[1] + 1):
                result = result + (w_new[i] - w_prev[i]) ** 2
            if result < pow(10, -6):
                return w_new




    # initialize all weights to 0.1(error function returns nan if weights are set to 1)

    # the training method, X=pure_data, Y=ground truth, lr=learning rate, n=number of iterations
    # lr should be initialized to 0.000001
    # n should be initialized to 100
    def fit(w, X, Y, lr, n):
        return LogisticRegression.descent(w, w, lr, n, X, Y)

    # accuracy method
    def acc(Y,pred):
        success=0
        for i in range(Y.shape[0]):
            if pred[i]>=0.5:
                pred[i]=1
            else: pred[i]=0
            if pred[i] == Y[i]:
                success+=1
        return success/Y.shape[0]
