import numpy as np
import pandas as pd
import math
from scipy.stats import mode

#Formula to calculate the euclidian distance 
def eucledianDist(p1,p2):
    distance = np.sqrt(np.sum((p1-p2)**2))
    return distance
 
#method to get KNN and spit prediction
def predict(x_train, y , x_input, k):
    results = []
     
    #Loop through the test data
    for item in x_input: 
         
        #Array to store distances
        point_distance = []
         
        #iterate all the training data
        for j in range(len(x_train)): 
            distances = eucledianDist(np.array(x_train[j,:]) , item) 
            
            point_distance.append(distances) 
        point_distance = np.array(point_distance) 
         
        #Sort the array while also preserving the index
        #Still keep the first K datapoints
        dist = np.argsort(point_distance)[:k] 
         
        #Labels of the K datapoints from above
        labels = y[dist]
         
        #Get ans return most common label
        lab = mode(labels) 
        lab = lab.mode[0]
        results.append(lab)
 
    return results

df_1 = pd.read_csv("TrainData1Updated.txt", sep = '\t', header=None)
df_1_label = pd.read_csv("TrainLabel1.txt", sep='\t', header=None)
df_1_test = pd.read_csv("TestData1.txt", sep='\t', header=None)

k = int(math.sqrt(len(df_1))) + 1
df_1_pred = predict(df_1.to_numpy(), df_1_label.to_numpy(), df_1_test.to_numpy(), k)
with open("KollaBlancoClassification1", "w") as outfile:
    outfile.write("\n".join(str(item) for item in df_1_pred))


df_1 = pd.read_csv("TrainData2.txt", sep = '  ', header=None)
df_1_label = pd.read_csv("TrainLabel2.txt", sep='  ', header=None)
df_1_test = pd.read_csv("TestData2.txt", sep='  ', header=None)

k = int(math.sqrt(len(df_1))) + 1
df_1_pred = predict(df_1.to_numpy(), df_1_label.to_numpy(), df_1_test.to_numpy(), k)

with open("KollaBlancoClassification2", "w") as outfile:
    outfile.write("\n".join(str(item) for item in df_1_pred))


df_1 = pd.read_csv("TrainData3Updated.txt", sep = '\t', header=None)
df_1_label = pd.read_csv("TrainLabel3.txt", sep='\t', header=None)
df_1_test = pd.read_csv("TestData3.txt", sep=',', header=None)

k = int(math.sqrt(len(df_1))) + 1
df_1_pred = predict(df_1.to_numpy(), df_1_label.to_numpy(), df_1_test.to_numpy(), k)

with open("KollaBlancoClassification3", "w") as outfile:
    outfile.write("\n".join(str(item) for item in df_1_pred))


df_1 = pd.read_csv("TrainData4.txt", sep = '  ', header=None)
df_1_label = pd.read_csv("TrainLabel4.txt", sep='\t', header=None)
df_1_test = pd.read_csv("TestData4.txt", sep='  ', header=None)

k = int(math.sqrt(len(df_1))) + 1
df_1_pred = predict(df_1.to_numpy(), df_1_label.to_numpy(), df_1_test.to_numpy(), k)

with open("KollaBlancoClassification4", "w") as outfile:
    outfile.write("\n".join(str(item) for item in df_1_pred))


df_1 = pd.read_csv("TrainData5.txt", sep = '  ', header=None)
df_1_label = pd.read_csv("TrainLabel5.txt", sep='  ', header=None)
df_1_test = pd.read_csv("TestData5.txt", sep='  ', header=None)

k = int(math.sqrt(len(df_1))) + 1
df_1_pred = predict(df_1.to_numpy(), df_1_label.to_numpy(), df_1_test.to_numpy(), k)
print(type(df_1_pred))
with open("KollaBlancoClassification5", "w") as outfile:
    outfile.write("\n".join(str(item) for item in df_1_pred))
