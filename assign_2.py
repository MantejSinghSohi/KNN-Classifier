import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

class KNN_Normal:
  def __init__(self, k_list):
    self.k= k_list

  def fit(self, X, y):
    self.X_train= X
    self.y_train= y

  def predict(self, X):
    predictions= [self._predict(x) for x in X]# predictions for inputs : list of lists
    return predictions

  def _predict(self, x):
    pred= []# this will contain outputs for differnet values of k
    for k_curr in self.k:
      distances= [self.Euclid(x, x_train) for x_train in self.X_train]# contains distance of the new data point (under observation) from all points in dataset
      indices= np.argsort(distances)[:k_curr]# slices till k nearest indices and stores it
      nearest_labels= self.y_train[indices]# collected labels for voting
      unique_labels, counts = np.unique(nearest_labels, return_counts=True)# voting
      most_common_label = unique_labels[np.argmax(counts)]
      pred.append(most_common_label)
    return pred

  def Euclid(self, x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

  def accuracy(self, predictions, y_test):
    array= np.array(predictions)
    accuracy= []
    for i in range (array.shape[1]):
      acc = (np.sum(y_test==array[:,i]))/(array.shape[0])
      accuracy.append(acc)
    return accuracy

  def best_k(self, accuracy):
    a= max(accuracy)
    best_k= []
    for i in range (len(accuracy)):
      if(accuracy[i]==a):
        best_k.append(self.k[i])
    return best_k

class KNN_Weighted:
  def __init__(self, k_list):
    self.k= k_list

  def fit(self, X, y):
    self.X_train= X
    self.y_train= y

  def predict(self, X):
    predictions= [self._predict(x) for x in X]# predictions for inputs : list of lists
    return predictions

  def _predict(self, x):
    pred= []# this will contain outputs for differnet values of k
    for k_curr in self.k:
      distances= [self.Euclid(x, x_train) for x_train in self.X_train]# contains distance of the new data point (under observation) from all points in dataset
      indices= np.argsort(distances)[:k_curr]# slices till k nearest indices and stores it
      score= [0, 0, 0]
      for i in indices:
        if(self.y_train[i]==0):
          if distances[i] == 0:
            continue
          score[0]= score[0] + 1/distances[i]
        if(self.y_train[i]==1):
          if distances[i] == 0:
            continue
          score[1]= score[1] + 1/distances[i]
        if(self.y_train[i]==2):
          if distances[i] == 0:
            continue
          score[2]= score[2] + 1/distances[i]
      predicted_label= score.index(max(score))
      pred.append(predicted_label)
    return pred

  def Euclid(self, x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

  def accuracy(self, predictions, y_test):
    array= np.array(predictions)
    accuracy= []
    for i in range (array.shape[1]):
      acc = (np.sum(y_test==array[:,i]))/(array.shape[0])
      accuracy.append(acc)
    return accuracy

  def best_k(self, accuracy):
    a= max(accuracy)
    best_k= []
    for i in range (len(accuracy)):
      if(accuracy[i]==a):
        best_k.append(self.k[i])
    return best_k

# loading of the dataset
iris_data= load_iris()
features= iris_data.data
targets= iris_data.target

# z score normalization of the data without using scikit learn
means = np.mean(features, axis =0)
deviations = np.std(features, axis =0)
_features = (features-means)/deviations

K= [1, 3, 5, 10, 20]
X_train, X_test, y_train, y_test = train_test_split(_features, targets, test_size=0.3, random_state=7)

Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
optimum_K= Knn_N.best_k(accuracy)

# Experiment 1
# Report
_accuracy= [100*x for x in accuracy]
print('''REPORT: EFFECT OF VARYING K IN KNN_NORMAL ON TEST DATA:
Accuracy is coming out to be largest for K= 1 and K=10 ; when analysed using KNN_Normal
Although, I believe selecting K= 10 will a better option as it is neither too big or nor too small value
Selecting K= 1 just based upon these results can be very risky as it will be very sensitive to noise and
may lead to significan dip in performance of the algorithm in case of some other test samples

''')

# printing best value of K --- I am selecting on the basis of maximum accuracy
print(f"Optimum value/ value's of hyperparameter K = {optimum_K} \n")

# plotting Percentage Accuracy vs K
plt.plot(K, _accuracy, marker= 'o', color= 'b', label= "Accuracy")
plt.xlabel("K Values")
plt.ylabel("Percentage Accuracy")
plt.xticks(K)
plt.yticks(_accuracy)
plt.legend()
plt.title("Percentage Accuracy vs K")
plt.show()
print("\n")

# Confusion matrix for best K -- I am plotting for K= 10 as I fell that its a better candidate (reason stated above)
K_considered= optimum_K[1]
target_index= K.index(K_considered)
_pred= np.array(predictions)
conf_matrix = confusion_matrix(y_test, _pred[:,target_index])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels= iris_data.target_names, yticklabels=iris_data.target_names)
plt.title('Confusion Matrix for K= 10')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# loading of the dataset
iris_data= load_iris()
features= iris_data.data
targets= iris_data.target

# z score normalization of the data without using scikit learn
means = np.mean(features, axis =0)
deviations = np.std(features, axis =0)
_features = (features-means)/deviations

K= [1, 3, 5, 10, 20]
X_train, X_test, y_train, y_test = train_test_split(_features, targets, test_size=0.3, random_state=7)

Knn_W = KNN_Weighted(K)
Knn_W.fit(X_train, y_train)
predictions= Knn_W.predict(X_test)
accuracy= Knn_W.accuracy(predictions, y_test)
optimum_K= Knn_W.best_k(accuracy)

# Experiment 2
# Report
_accuracy= [100*x for x in accuracy]
print('''REPORT: EFFECT OF VARYING K IN KNN_WEIGHTED ON TEST DATA:
Optimun value of K is obtained to be 5
Either if we try to reduce the value of k or we try to increase the value of k, the accuracy tends to drop
for lower value of k, it is the noise samples that reduce the accuracy and for larger valures of k,
if we encounter any example at the border line, it tends to produce wrong results
''')

# printing best value of K --- I am selecting on the basis of maximum accuracy
print(f"Optimum value/ value's of hyperparameter K = {optimum_K} \n")

# plotting Percentage Accuracy vs K
plt.plot(K, _accuracy, marker= 'o', color= 'g', label= "Accuracy")
plt.xlabel("K Values")
plt.ylabel("Percentage Accuracy")
plt.xticks(K)
plt.yticks(_accuracy)
plt.legend()
plt.title("Percentage Accuracy vs K")
plt.show()
print("\n")

# Confusion matrix for best K
target_index= K.index(optimum_K[0])
_pred= np.array(predictions)
conf_matrix = confusion_matrix(y_test, _pred[:,target_index])
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels= iris_data.target_names, yticklabels=iris_data.target_names)
plt.title(f'Confusion Matrix for K= 5')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Experiment 3
iris_data= load_iris()
features= iris_data.data
targets= iris_data.target

# z score normalization of the data without using scikit learn
means = np.mean(features, axis =0)
deviations = np.std(features, axis =0)
_features = (features-means)/deviations

X_train, X_test, y_train, y_test = train_test_split(_features, targets, test_size=0.3, random_state=7)

# adding noise to 10% of the training data
np.random.seed(12)
size= (int(X_train.shape[0]*0.1), X_train.shape[1])
noise_sample = np.random.normal(loc=0, scale=1, size=size)
random_indices= np.random.randint(0,105,size=10)
X_train[random_indices]= noise_sample

# Experiment 3 -- APPLYING KNN_NORMAL with K= 1 and K=10
print('''REPORT: VARIATION IN PERFORMANCE OF KNN_NORMAL AFTER NOISE
As it was expected, after adding noise, the results for k=1 dipped down in accuracy.
This is the drawback of choosing a model that makes decision on the basis of less neighbours
But, the accuracy for k= 10 is still entact as it is not sensitive to presence noise samples
as significant number of neighbours are considered before making any decision
''')

K= [1, 10]
Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
optimum_K= Knn_N.best_k(accuracy)
_accuracy= [100*x for x in accuracy]

# plotting Percentage Accuracy vs K
plt.plot(K, _accuracy, marker= 'o', color= 'b', label= "Accuracy")
plt.xlabel("K Values")
plt.ylabel("Percentage Accuracy")
plt.xticks(K)
plt.yticks(_accuracy)
plt.legend()
plt.title("Percentage Accuracy vs K")
plt.show()

# Experiment 3
iris_data= load_iris()
features= iris_data.data
targets= iris_data.target

# z score normalization of the data without using scikit learn
means = np.mean(features, axis =0)
deviations = np.std(features, axis =0)
_features = (features-means)/deviations

K= [1, 3, 5, 10, 20]
X_train, X_test, y_train, y_test = train_test_split(_features, targets, test_size=0.3, random_state=7)

# adding noise to 10% of the training data
np.random.seed(12)
size= (int(X_train.shape[0]*0.1), X_train.shape[1])
noise_sample = np.random.normal(loc=0, scale=1, size=size)
random_indices= np.random.randint(0,105,size=10)
X_train[random_indices]= noise_sample

# EXPERIMENT 3 -- APPLYING KNN_WEIGHTED with K= 5
print('''REPORT: VARIATION IN PERFORMANCE OF KNN_WEIGHTED AFTER NOISE
The performance has dropped a little bit (in terms of accuracy) as expected because of introduction of noise samples
 ''')

K= [5]
Knn_W = KNN_Weighted(K)
Knn_W.fit(X_train, y_train)
predictions= Knn_W.predict(X_test)
accuracy= Knn_W.accuracy(predictions, y_test)
optimum_K= Knn_W.best_k(accuracy)
_accuracy= [100*x for x in accuracy]

# plotting Percentage Accuracy vs K
plt.plot(K, _accuracy, marker= 'o', color= 'g', label= "Accuracy")
plt.xlabel("K Values")
plt.ylabel("Percentage Accuracy")
plt.xticks(K)
plt.yticks(_accuracy)
plt.legend()
plt.title("Percentage Accuracy vs K")
plt.show()

# Experiment 4
print('''REPORT: CURSE OF DIMENSIONALITY
Considering only sepal parameters, petal parameters, width parameters, or length parameters individually might lead to
a reduction in dimensionality, potentially simplifying the analysis. However, such reduction may result in the loss of valuable information.
We should give less importance to less important parameters and higher weightage to effective and meaningful parameters. This is the real
problem i.e. selecting appropriate number of QUALITY features

IT SEEMS FROM THE ANALYSIS THAT CONSIDERING WIDTH PARAMETERS ONLY GIVES A LOT OF INFROMATION ABOUT THE DATA AND SAVES THE COMPUTATION TOO!
''')

K= [10] # i.e. optimal K value obtained
Accuracy= []
iris_data= load_iris()
features= iris_data.data
targets= iris_data.target

# z score normalization of the data without using scikit learn
means = np.mean(features, axis =0)
deviations = np.std(features, axis =0)
_features = (features-means)/deviations

# CASE A
X_train, X_test, y_train, y_test = train_test_split(_features, targets, test_size=0.3, random_state=7)
Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
Accuracy.append(accuracy[0])

# CASE B
features_b= _features[:,[2, 3]]
X_train, X_test, y_train, y_test = train_test_split(features_b, targets, test_size=0.3, random_state=7)
Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
Accuracy.append(accuracy[0])

# CASE C
features_c= _features[:,[0,1]]
X_train, X_test, y_train, y_test = train_test_split(features_c, targets, test_size=0.3, random_state=7)
Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
Accuracy.append(accuracy[0])

# CASE D
features_d= _features[:,[0,2]]
X_train, X_test, y_train, y_test = train_test_split(features_d, targets, test_size=0.3, random_state=7)
Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
Accuracy.append(accuracy[0])

# CASE E
features_e= _features[:,[1,3]]
X_train, X_test, y_train, y_test = train_test_split(features_e, targets, test_size=0.3, random_state=7)
Knn_N = KNN_Normal(K)
Knn_N.fit(X_train, y_train)
predictions= Knn_N.predict(X_test)
accuracy= Knn_N.accuracy(predictions, y_test)
Accuracy.append(accuracy[0])

_Accuracy= [100*x for x in Accuracy]
List= ["All four parameters", "Only petal parameters", "Only sepal parameters", "Only length parameters", "Only width parameters"]

# plotting Percentage Accuracy vs K
plt.plot(List, _Accuracy, marker= 'o', color= 'g', label= "Accuracy")
plt.xlabel("Parameters Considered")
plt.ylabel("Percentage Accuracy")
plt.xticks(List, rotation= 90)
plt.yticks(_Accuracy)
plt.legend()
plt.title("Curse of Dimensionality")
plt.show()