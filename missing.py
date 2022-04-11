import csv
import math

filename = open("C:/Users/KASH/Desktop/ML_DATA/TrainData1.txt", "r")
reader = csv.reader(filename, delimiter="\t")

# this nested list contains all the data from the dataset text file
dataset = []
for data in reader:
    dataset.append(data)


# returns a nested list that contains each tuple's distance to every other tuple in the dataset
def find_distance(dataset):
    tuples = len(dataset)
    tot_attributes = len(dataset[0])
    # the nested list that contains the distances for every tuple
    distances = [[0 for j in range(tuples)] for i in range(tuples)]

    # loops through each tuple in the dataset
    for i in range(tuples):
        # loops through every other tuple and find the distance between each of them and the current tuple from loop i
        for j in range(tuples):
            dist = 0
            pres_attributes = 0

            # goes to the next tuple if it gets to the current tuple from loop i
            if j == i:
                continue
            else:
                # loops through each attribute of the two tuples being compared
                for k in range(tot_attributes):
                    # if the current attribute of either tuple is a missing value, the loop moves on to the next attribute
                    if (
                        dataset[i][k] == "1.00000000000000e+99"
                        or dataset[j][k] == "1.00000000000000e+99"
                    ):
                        continue
                    else:
                        pres_attributes += 1
                        dist += (float(dataset[i][k]) - float(dataset[j][k])) ** 2

            # weight is calclulated depending on how many present attributes there are
            weight = tot_attributes / pres_attributes
            distance = math.sqrt(weight * dist)
            distances[i][j] = distance

    return distances


# iterates through the dataset and finds which tuples have missing values
# the function calculate_missing_value is then called on that tuple
def find_missing_value(dataset, distances):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] == "1.00000000000000e+99":
                calculate_missing_value(dataset, distances, i)
                break


# estimates the missing values for the tuple that is inputted in the function call
def calculate_missing_value(dataset, distances, tuple_i):
    # contains the inputted tuple's distance to every other tuple in the dataset
    tuple_dist = distances[tuple_i]
    # contains the four lowest distances from the inputted tuple's distance list
    nearest = sorted(tuple_dist)[:4]
    # contains the indexes of the three nearest neighbors to the inputted tuple
    neighbors = []
    # contains the indexes of the missing values of the inputted tuple
    missing_index = []

    # finds the indexes of the three nearest tuples in the dataset as n is equal to 3 in this implementation
    # and appends the indexes to the neighbors list
    # starts from index 1 as index 0 represents the distance between the tuple and itself
    for i in nearest[1:]:
        neighbors.append(tuple_dist.index(i))

    # finds the index of the missing values of the inputted tuple and appends them to the missing_index list
    for i in range(len(dataset[tuple_i])):
        if dataset[tuple_i][i] == "1.00000000000000e+99":
            missing_index.append(i)

    # loops through the indexes of the missing values of the inputted tuple
    for i in range(len(missing_index)):
        sum = 0
        values = 0

        # loops through each of the nearest neighbors and sums the values that are at the same index of the missing index
        for j in range(len(neighbors)):
            # if the current value is missing, it is not contributed towards the estimation
            if dataset[neighbors[j]][missing_index[i]] == "1.00000000000000e+99":
                continue
            else:
                values += 1
                sum += float(dataset[neighbors[j]][missing_index[i]])

        # if at least one of the nearest neighbors has a valid value, the missing value is calculated accordingly
        if values != 0:
            # the mean of the values of the three nearest neighbors that are at the same index of the missing index is calculated
            imputed_val = sum / values
            # the estimated value is then imputed into the dataset
            dataset[tuple_i][missing_index[i]] = str(imputed_val)
        # if none of the nearest neighbors have a valid value, the missing value is estimated in a naive way
        # it estimates the missing value by finding the mean of all of the valid attributes/values of the inputted tuple
        else:
            naive = 0
            values1 = 0

            for k in range(len(dataset[tuple_i])):
                if (
                    dataset[tuple_i][k] == 0
                    or dataset[tuple_i][k] == "1.00000000000000e+99"
                ):
                    continue
                else:
                    values1 += 1
                    naive += float(dataset[tuple_i][k])

            naive = naive / values1
            dataset[tuple_i][missing_index[i]] = str(naive)


# once the two functions below run, the "dataset" list will now be updated with the missing values replaced with the estimated values
distances = find_distance(dataset)
find_missing_value(dataset, distances)


# takes the updated dataset and writes it to a text file
with open("TrainData1Updated.txt", "w") as file:
    for item in dataset:
        file.write("\t".join(item))
        file.write("\n")
