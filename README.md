# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vishwa K
RegisterNumber:  212223080061
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:
```
data.head()
```
![Screenshot 2024-09-27 140414](https://github.com/user-attachments/assets/a11c9d78-5334-4abb-9405-cbe09c13a186)

```
data.info()
```
![Screenshot 2024-09-27 140446](https://github.com/user-attachments/assets/2770e8d9-47f1-44d7-9caa-4b2e6a14600f)

```
data.isnull().sum()
```
![Screenshot 2024-09-27 140502](https://github.com/user-attachments/assets/17ae8264-8a3b-486b-914c-c4fcc06b2fb9)


```
data["left"].value_counts()
```
![Screenshot 2024-09-27 140520](https://github.com/user-attachments/assets/88e7648b-3cee-4696-893f-c42a1db6a866)


```
Predection
```
![Screenshot 2024-09-27 134928](https://github.com/user-attachments/assets/20ebed51-fc30-42f2-a75e-a63e3e190931)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
