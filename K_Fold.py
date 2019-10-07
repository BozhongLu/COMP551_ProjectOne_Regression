from LDA import *
from data_processing_general import *
from logisticRegression import *
import time

# split data into k sections
def split_data (data_In , k):
    dataSet=data_In.copy()
    kFolds = np.array_split(dataSet, k)
    result = np.array([])
    for i in range(k):
        entireSet = kFolds.copy()
        validationSet=entireSet[i]
        #delete the i th element(validationSet) from the splitted array
        del entireSet[i]
        #concatenate the rest of the splitted array into one array as training set
        trainingSet = np.concatenate(entireSet)
        result=np.append(result,[trainingSet,validationSet])
    return result

# compute accuracy of LDA model for both wine_data and cancer_data

# compute accuracy of LDA model for wine_data
model_LDA = LDA
def get_LDA_accuracy_wine (dataOfWine,k):


    wine_data_splitted=split_data(dataOfWine,k)
    sum_accuracy = 0
    for i in range(0,len(wine_data_splitted),2):
        wine_processed=wineData_processing(wine_data_splitted[i])
        #purify validation data by deleting last column if wine
        wine_validationData=wine_data_splitted[i+1]
        wine_validationData_processed=wineData_processing(wine_validationData)
        wine_validationData = np.delete(wine_validationData,-1,axis=1)
        wineWeights=model_LDA.fit_LDA(wine_processed[0],wine_processed[1],wine_processed[2],wine_processed[3],wine_processed[4])
        prediction=model_LDA.predict_LDA(wineWeights,wine_validationData)
        accuracy=model_LDA.evaluate_acc(wine_validationData_processed[4],prediction)
        sum_accuracy = sum_accuracy + accuracy
        average_accuracy = sum_accuracy/k
    print("When k=",k," Average wine data accuracy=",average_accuracy)
    return average_accuracy

# compute average accuracy of LDA model for cancer_data
def get_LDA_accuracy_breastCancer (dataOfCancer,k):
    cancer_data_splitted=split_data(dataOfCancer,k)
    sum_accuracy = 0
    for i in range(0,len(cancer_data_splitted),2):
        cancer_processed=cancerData_processing(cancer_data_splitted[i])
        #purify validation data by deleting first and last column if cancer
        cancer_validationData=cancer_data_splitted[i+1]
        cancer_validationData_processed=cancerData_processing(cancer_validationData)
        cancer_validationData = np.delete(cancer_validationData,-1,axis=1)
        cancer_validationData = np.delete(cancer_validationData,0,axis=1)
        cancerWeights=model_LDA.fit_LDA(cancer_processed[0],cancer_processed[1],cancer_processed[2],cancer_processed[3],cancer_processed[4])
        prediction=model_LDA.predict_LDA(cancerWeights,cancer_validationData_processed[3])
        accuracy=model_LDA.evaluate_acc(cancer_validationData_processed[4],prediction)
        sum_accuracy = sum_accuracy + accuracy
        average_accuracy = sum_accuracy/k
    print("When k=",k," Average cancer data accuracy=",average_accuracy)
    return average_accuracy


#------------------------------------------------------------------------------------

# compute accuracy of logisticRegression model for both wine_data and cancer_data

# compute average accuracy of logisticRegression model for wine_data
model_LR = LogisticRegression
def get_logistic_accuracy_wine (dataOfWine,k,lr,iteration):
    wine_data_splitted=split_data(dataOfWine,k)
    sum_accuracy = 0
    w_initial=[0.1]*12

    for i in range(0,len(wine_data_splitted),2):
        wine_processed=wineData_processing(wine_data_splitted[i])
        #purify validation data by deleting last column if wine
        wine_validationData=wine_data_splitted[i+1]
        wine_validationData_processed=wineData_processing(wine_validationData)
        wine_validationData = np.delete(wine_validationData,-1,axis=1)
        wineWeights=model_LR.fit(w_initial,wine_processed[3],wine_processed[4],lr,iteration)
        prediction=model_LR.predict(wineWeights,wine_validationData)
        accuracy=model_LR.acc(wine_validationData_processed[4],prediction)
        sum_accuracy = sum_accuracy + accuracy
        average_accuracy = sum_accuracy/k
    print("When k=",k," Average wine data accuracy=",average_accuracy)
    return average_accuracy

# compute average accuracy of logisticRegression model for cancer_data
def get_logistic_accuracy_breastCancer (dataOfCancer,k,lr,iteration):
    cancer_data_splitted=split_data(dataOfCancer,k)
    sum_accuracy = 0
    for i in range(0,len(cancer_data_splitted),2):
        cancer_processed=cancerData_processing(cancer_data_splitted[i])
        #purify validation data by deleting first and last column if cancer
        cancer_validationData=cancer_data_splitted[i+1]
        cancer_validationData_processed=cancerData_processing(cancer_validationData)
        cancer_validationData = np.delete(cancer_validationData,-1,axis=1)
        cancer_validationData = np.delete(cancer_validationData,0,axis=1)
        cancerWeights=model_LR.fit([0.1]*11,cancer_processed[3],cancer_processed[4],lr,iteration)
        prediction=model_LR.predict(cancerWeights,cancer_validationData_processed[3])
        accuracy=model_LR.acc(cancer_validationData_processed[4],prediction)
        sum_accuracy = sum_accuracy + accuracy
        average_accuracy = sum_accuracy/k
    print("When k=",k," Average cancer data accuracy=",average_accuracy)
    return average_accuracy

#print("LR=0.1",(1-get_logistic_accuracy_wine(wine_data,5,0.1,100)))
#print("LR=0.1",get_logistic_accuracy_breastCancer(cancer_data,5,0.1,100))
#print("------------------------------------------------------------------------------")
#print("LR=0.01",(1-get_logistic_accuracy_wine(wine_data,5,0.01,100)))
#print("LR=0.01",get_logistic_accuracy_breastCancer(cancer_data,5,0.01,100))
#print("------------------------------------------------------------------------------")
#print("LR=0.001",(1-get_logistic_accuracy_wine(wine_data,5,0.001,100)))
#print("LR=0.001",get_logistic_accuracy_breastCancer(cancer_data,5,0.001,100))
#print("------------------------------------------------------------------------------")
#print("LR=0.0001",(1-get_logistic_accuracy_wine(wine_data,5,0.0001,100)))
#print("LR=0.001",get_logistic_accuracy_breastCancer(cancer_data,5,0.0001,100))
#print("------------------------------------------------------------------------------")
#print("LR=0.00001",(1-get_logistic_accuracy_wine(wine_data,5,0.00001,100)))
#print("LR=0.00001",get_logistic_accuracy_breastCancer(cancer_data,5,0.00001,100))
#print("------------------------------------------------------------------------------")
#print("LR=0.000001",get_logistic_accuracy_wine(wine_data,5,0.000001,100))
#print("LR=0.000001",get_logistic_accuracy_breastCancer(cancer_data,5,0.000001,100))
#print("------------------------------------------------------------------------------")
#print("LR=0.0000001",get_logistic_accuracy_wine(wine_data,5,0.0000001,100))
#print("LR=0.0000001",get_logistic_accuracy_breastCancer(cancer_data,5,0.0000001,100))

#print("LR")
#start_time = time.time()
#print("LR=0.000001",get_logistic_accuracy_wine(wine_data,5,0.000001,100))
#print("--- %s seconds LR wine---" % (time.time() - start_time))

#start_time = time.time()
#print("LR=0.01",get_logistic_accuracy_breastCancer(cancer_data,5,0.01,100))
#print("--- %s seconds LR cancer---" % (time.time() - start_time))

#print("------------------------------------------------------------------------------")
#print("LDA")
#start_time = time.time()
#print(get_LDA_accuracy_wine(wine_data,5))
#print("--- %s seconds LDA wine---" % (time.time() - start_time))

#start_time = time.time()
#print(get_LDA_accuracy_breastCancer(cancer_data,5))
#print("--- %s seconds LDA cancer---" % (time.time() - start_time))
