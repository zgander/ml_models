import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("./data/titanic_train_cleaned.csv")

train_df = pd.DataFrame(train_data)

x_train = train_df[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
y_train = train_df["Survived"]

test_data = pd.read_csv("./data/test.csv")

test_df = pd.DataFrame(test_data)
x_test = test_df[["Pclass","Sex","Age","SibSp","Parch","Fare"]]

model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_train)

print(accuracy_score(y_train, predictions))


