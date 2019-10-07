import numpy as np
import matplotlib.pyplot as plt

# Load the datasets into numpy objects (i.e., arrays or matrices) in Python.
wine_data = np.genfromtxt('winequality-red.csv',delimiter=";", skip_header=1, dtype=np.float)
cancer_data = np.genfromtxt("breast-cancer-wisconsin.data",delimiter =",")

def wineData_processing(wineData):
    wine_data = wineData.copy()
    # Convert wine dataset into a binary classification
    [wine_rows, wine_cols] = wine_data.shape
    for i in range(wine_rows):
        if wine_data[i][-1] < 6:
            wine_data[i][-1] = 0
        else:
            wine_data[i][-1] = 1

    # Selecting only the valid elements
    wine_data = wine_data[~np.isnan(wine_data).any(axis=1)]

    # Seperate datasets according to their class - or + and store as an array
    wine_data_0 = np.asarray(list(filter(negative, wine_data)))
    wine_data_1 = np.asarray(list(filter(positive, wine_data)))

    # We need delete last column of wine_data_0 and _1
    wine_data_0 = np.delete(wine_data_0, -1, axis=1)
    wine_data_1 = np.delete(wine_data_1, -1, axis=1)

    # wine_data_pure means original wine data without last column
    wine_data_pure = np.delete(wine_data, -1, axis=1)

    # Y_wine is the ground truth (last column) from wine_data
    Y_wine = np.array(wine_data[:, -1])

    result=np.array([wine_data,wine_data_0,wine_data_1,wine_data_pure,Y_wine])
    return result

def cancerData_processing(cancerData):
    cancer_data=cancerData.copy()

    # Convert cancer dataset into a binary classification
    [cancer_rows, cancer_cols] = cancer_data.shape
    for i in range(cancer_rows):
        if cancer_data[i][-1] == 4:
            cancer_data[i][-1] = 0
        else:
            cancer_data[i][-1] = 1

    # Selecting only the valid elements
    cancer_data = cancer_data[~np.isnan(cancer_data).any(axis=1)]

    # Seperate datasets according to their class - or + and store as an array
    cancer_data_0 = np.asarray(list(filter(negative, cancer_data)))
    cancer_data_1 = np.asarray(list(filter(positive, cancer_data)))

    # We need delete first column(id) and last column of cancer_data_0 and _1
    cancer_data_0 = np.delete(cancer_data_0, 0, axis=1)
    cancer_data_1 = np.delete(cancer_data_1, 0, axis=1)
    cancer_data_0 = np.delete(cancer_data_0, -1, axis=1)
    cancer_data_1 = np.delete(cancer_data_1, -1, axis=1)

    # cancer_data_pure means original cancer data without first and last column
    cancer_data_pure = np.delete(cancer_data, 0, axis=1)
    cancer_data_pure = np.delete(cancer_data_pure, -1, axis=1)

    # Y_cancer is the ground truth (last column) from cancer_data
    Y_cancer = np.array(cancer_data[:, -1])

    result=np.array([cancer_data,cancer_data_0,cancer_data_1,cancer_data_pure,Y_cancer])

    return result

# Defining filters to split success and failures into two classes
# binary class for wine data: 0 for negative, 1 for positive
# binary class for cancer data: 0 for malignant, 1 for benign
def positive(arr):
    if arr[-1] == 0:
        return False
    else: return True
def negative(arr):
    return not(positive(arr))


