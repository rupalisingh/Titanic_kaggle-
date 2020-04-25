import pandas as pd
import sklearn
import numpy as np



#Importing dataset

df_train = pd.read_csv("train.csv")
df_train = df_train.drop(["Name", "Cabin", "Ticket"], axis=1)
df_train_X = df_train.iloc[:, [0,2,3,4,5,6,7,8]]
df_train_Y = df_train.iloc[:, 1]
df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["Name", "Cabin", "Ticket"], axis=1)
id = df_test["PassengerId"]



# Preprocessing

# Missing Data
# Fare
df_test["Fare"] = df_test["Fare"].replace(np.NaN, df_test["Fare"].mean())

# Age
df_train_X["Age"] = df_train_X["Age"].replace(np.NaN, df_train_X["Age"].mean())
df_test["Age"] = df_test["Age"].replace(np.NaN, df_test["Age"].mean())

df_test.info()

# Embarked

df_train_X["Embarked"].describe()
common_value = "S"
df_train_X["Embarked"] = df_train_X["Embarked"].fillna(common_value)
df_test["Embarked"].describe()
df_test["Embarked"] = df_test["Embarked"].fillna(common_value)


# Categorical features

from sklearn.preprocessing import LabelEncoder

label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()
label_encoder4 = LabelEncoder()

df_train_X["Sex"] = label_encoder1.fit_transform(df_train_X["Sex"])
df_train_X["Embarked"] = label_encoder2.fit_transform(df_train_X["Embarked"])
df_test["Sex"] = label_encoder3.fit_transform(df_test["Sex"])
df_test["Embarked"] = label_encoder4.fit_transform(df_test["Embarked"])

# Feature Scaling

from sklearn.preprocessing import StandardScaler

feature_scaling = StandardScaler()
df_train_X = feature_scaling.fit_transform(df_train_X)
df_test = feature_scaling.fit_transform(df_test)


# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(df_train_X, df_train_Y)

Y_pred = random_forest.predict(df_test)

accuracy_random = round(random_forest.score(df_train_X, df_train_Y)* 100, 2)
print(accuracy_random)
print(Y_pred)

submission = pd.DataFrame({"PassengerId": id, "Survived": Y_pred})
submission.to_csv("submissions.csv")




















