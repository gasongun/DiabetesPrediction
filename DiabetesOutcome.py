"""

Diabetes Prediction

The aim of this project is to predict whether women over the age of 21 will have diabetes or not.
Diabetes dataset come from the National Institute of Diabetes and Digestive and Kidney Diseases in United States.

We are going to follow the content below in this project:

1- Import Data and Libraries
    a- Importing Libraries
    b- Importing Data
2- Data Preprocessing
3- Feature Engineering
    a- Missing Value Analysis
    b- Outlier Value Analysis
    c- Feature Extraction
    d- One Hot Encoding
4- Find the optimum model for estimate will have diabetes.
    a-  Random Forest
    b-  Logistic Regression
    c-  SMOTE + Logistic Regression

"""
## Import Data and Libraries

# Importing Libraries

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Importing Data

def load_dataset(data,pathChange = True):
    path = os.getcwd()
    if pathChange == False:
        pathContinue = input("Write the continue of the path")
        os.chdir(path + '/' + pathContinue)
        path = os.getcwd()
    return pd.read_csv(path + '/' + data + ".csv")


df_ = load_dataset("diabetes",False)     # DSMBLC7-Gülbüke\5.Hafta
df = df_.copy()
df.head()

# Dataset has 9 variables and 768 cases
df.shape

## Data Preprocessing

# First we will examine our dataset.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    This function returns the group of variables in the dataset. These are categorical, numerical and cardinal which is defined as object but number of categorical values more than the threshold value.
    PS: Categorical group includes that looks numeric but is categorical variables.


    Parameters
    ------
        dataframe: dataframe
                Dataframe which uses for variable types
        cat_th: int, optional
                Threshold value for a variable that looks numeric but is categorical.
        car_th: int, optional
                Threshold value for a variable that looks categoric but is cardinal.

    Returns
    ------
        cat_cols: list
                List of categorical variables.
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of categorical but cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = All variables in the dataset.
        cat_cols list includes num_but_cat list

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# The dataset has 9 variables and 768 observations. One of them is categorical which is our dependent variable.
# Pregnancies variable looks numerical column but it shouldn't be in numerical variables.
# This variable's unique category count is greater than our cat_th parameter value, this threshold can be changed or variable category can be changed.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(f'cat_cols: {cat_cols}\n')
# cat_cols: ['Outcome']
print(f'num_cols: {num_cols}\n')
# num_cols: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

df["Pregnancies"].nunique() # 17
num_cols.pop(0)
cat_cols.append('Pregnancies')

def cat_summary(dataframe, col_name, plot=False):
    """
    This function returns value counts and ratios of categorical variables in dataset. Also when you choose plot parameter as True, it gives count plot for each categorical variable.

    Parameters
    ----------
    dataframe: Dataframe which uses for analyzing the categorical variables
    col_name: Column which to be analyzed
    plot: Parameter takes True/False option that if its true, it gives a countplot of the variable

   Examples
   ------
       import seaborn as sns
        df = sns.load_dataset("diamonds")
        cat_summary(df, "color", plot=True)

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# We examine our categorical variables with the cat_summary function.
# After examination, dependent variable seems on the limit being a balanced variable. May be it could be a problem but we will see.

for col in cat_cols:
    cat_summary(df, col, plot=True)

"""
   Outcome  Ratio
0      500 65.104
1      268 34.896
##########################################
    Pregnancies  Ratio
1           135 17.578
0           111 14.453
2           103 13.411
3            75  9.766
4            68  8.854
5            57  7.422
6            50  6.510
7            45  5.859
8            38  4.948
9            28  3.646
10           24  3.125
11           11  1.432
13           10  1.302
12            9  1.172
14            2  0.260
15            1  0.130
17            1  0.130
"""

def num_summary(dataframe, numerical_col, plot=False):
    """
    This function returns description of numerical variable in dataset.
    Quartiles are split more frequently to examine the outlier and extreme values.
    Also when you choose plot parameter as True, it gives histrogram plot for numerical variable.

    Parameters
    ----------
    dataframe: Dataframe which uses for analyzing the numerical variables
    numerical_col: Column which to be analyzed
    plot: Parameter takes True/False option that if its true, it gives a histogram of the variable

    Examples
     ------
       import seaborn as sns
        df = sns.load_dataset("diamonds")
        num_summary(df, "carat", plot=True)

    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

# We examine our numerical variables with the num_summary function.
# The Insulin variable has %40 a value of zero, and shows a jump above the %95 values. Also SkinThickness variable has %20 a value of zero and a jump above the %99 values.
# These two variables definitely need an action.

for col in num_cols:
    num_summary(df, col, plot=False)

## Feature Engineering

# Missing Value Analysis

def missing_values_table(dataframe, na_name=False):
    import pandas as pd
    import numpy as np
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

# After first look table of the missing values, looks like there are no missing values in the dataset.
# Some variable contains 0 value like glucose, insulin etc.
# These are missing because some variables can't be zero like blood pressure, insulin level as we know with our business knowledge.

missing_values_table(df, na_name=True)

# Considering this situation, we assign the zero values to the relevant values as NaN and then apply the operations to the missing values.
# When we call the missing value function again, can be seen these variables has missing values.
# Most of the Insulin and SkinThickness variables are missing, we don't prefer the filling methods on these two variables.

for col in num_cols:
    df.loc[df[col] == 0, col] = np.NAN

missing_values_table(df, na_name=True)

# We decided to exclude both variables from the na columns and create a list of na variables.
# There are 3 variables that have missing values.

na_cols = [col for col in df.columns if ((df[col].isnull().sum() / df.shape[0] * 100) < 20 and (df[col].isnull().sum() > 0))]

# We decided to fill in these missing variables using the mean or median by looking at the distributions of the variables.
# Glucose and BloodPressure variables look not skewed, BMI looks skewed.
# If the variable looks skewed, it's better to fill with median and if it doesnt look skewed, can be filled with mean.

df[na_cols].hist(bins=20)

# Filled
skewed = ['BMI']
not_skewed = ['Glucose', 'BloodPressure']
df = df.fillna(df[skewed].median())
df = df.fillna(df[not_skewed].mean())

# Insulin and SkinThickness variables have still missing values. We dont forget on the feature extraction step.

missing_values_table(df, na_name=True)


# Feature Extraction

# *Pregnancies* variable update.
# Pregnancies variable has very diverse. We saw this diversty on the cat_summary function's output.
# People can give birth more than 4 children but frequency will be less than the others.
# With this information we will merge more than 4 birth to same group.

df.loc[(df['Pregnancies'] >= 4), 'Pregnancies'] = '4+'
df['Pregnancies'].value_counts()

# *AgeGroup* variable from Age.
# It's better to look as a grouping variable for age. Because fertility decreases with age.
# Older age pregnancies are less healthier than the younger ones.

df['Age'].hist(bins=20)
plt.show(block=True)

df.loc[(df['Age'] <= 30), 'Age_Group'] = '-30'
df.loc[((df['Age'] >= 31) & (df['Age'] <= 40)), 'Age_Group'] = '31-40'
df.loc[((df['Age'] >= 41 )& (df['Age'] <= 50)), 'Age_Group'] = '41-50'
df.loc[(df['Age'] >= 51), 'Age_Group'] = '50+'

df.pivot_table(df[['Age']], index='Age_Group', columns='Pregnancies', aggfunc='count')

# *GlucoseGroup* variable from Glucose (2 hours after eating).
# When we search information about the blood sugar level during on pregnancy, we learnt blood sugar level has three different period which are before meal, am hour after a meal and two hours after a meal.

# https://www.webmd.com/diabetes/gestational-diabetes#:~:text=The%20American%20Diabetes%20Association%20recommends,120%20mg%2FdL%20or%20less

df.loc[:, "GlucoseGroup"] = np.where((df['Glucose'] <= 120), 'Normal','Not-Normal')
df['GlucoseGroup'].value_counts()

# *InsulinGroup* variable from Insulin (2 hours after eating).
# A document on below shows insulin resistance levels, I've chosen to cut insulin resistance as 4 level and one more group for missing values.
# Because %48 of Insulin variable is missing.

# https://thebloodcode.com/insulin-resistancet2-diabetes-map-test-results/

df.loc[df['Insulin'].isna(), 'InsulinGroup'] = 'Missing'
df.loc[(df['Insulin'] <= 74), 'InsulinGroup'] = 'Low'
df.loc[((df['Insulin'] > 74) & (df['Insulin'] < 96)), 'InsulinGroup'] = 'Normal'
df.loc[((df['Insulin'] > 95) & (df['Insulin'] < 125)), 'InsulinGroup'] = 'High'
df.loc[(df['Insulin'] >= 125), 'InsulinGroup'] = 'VeryHigh'

df['InsulinGroup'].value_counts()
df[["InsulinGroup","Insulin"]].groupby("InsulinGroup").agg({"sum","mean","min","max"})

# *BMIGroup* variable from BMI
# BMI variable can be use as a numerical variable which means BMI index but in general people are grouped according to their BMI index.

# https://en.wikipedia.org/wiki/Body_mass_index

df.loc[(df['BMI'] < 18.5), 'BMIGroup'] = 'Underweight'
df.loc[((df['BMI'] >= 18.5) & (df['BMI'] < 25)), 'BMIGroup'] = 'Normal'
df.loc[((df['BMI'] >= 25) & (df['BMI'] < 30)), 'BMIGroup'] = 'Overweight'
df.loc[((df['BMI'] >= 30) & (df['BMI'] < 35)), 'BMIGroup'] = 'Obese(ClassI)'
df.loc[((df['BMI'] >= 35) & (df['BMI'] < 40)), 'BMIGroup'] = 'Obese(ClassII)'
df.loc[(df['BMI'] >= 40), 'BMIGroup'] = 'Obese(ClassIII)'

df['BMIGroup'].value_counts()

# *SkinThicknessGroup* variable from SkinThickness.
# As a document on wikipedia, skin plays an important immunity role in protecting the body against pathogens and excessive water loss. A skin cell usually ranges from 25 to 40 μm2
# This information can give us a perspective about diabetes.

# https://en.wikipedia.org/wiki/Human_skin

df["SkinThicknessGroup"] = pd.cut(df["SkinThickness"],
                                  [df["SkinThickness"].min(),25,40,df["SkinThickness"].max()],
                                  labels=["Low","Normal","High"],
                                  include_lowest=True)
df['SkinThicknessGroup'].value_counts()


# *BloodPressureGroup* variable from BloodPressure.
# BloodPressure is important for human life, sometimes can be up and down which depends with your activity but when you are resting, it is expected to be stable.
# This variable also can be related with being overweight or not, then we group as being low, optimal or high.

# https://www.bhf.org.uk/informationsupport/risk-factors/high-blood-pressure

df["BloodPressureGroup"] = pd.cut(df["BloodPressure"],
                                  [df["BloodPressure"].min(),60,80,df["BloodPressure"].max()],
                                  labels=["Low","Optimal","High"],
                                  include_lowest=True)

df["BloodPressureGroup"].value_counts()


# Before feature extraction step, dataset had 9 variables and 768 cases, after this step has 15 variables now.
df.shape

# There are 6 variables that we produce using variables themselves. We have to remove these 6 variables.
# We call a grab_col_names function again with new variables

dropVariables = ["Age","Glucose","Insulin","BMI","SkinThickness","BloodPressure"]
df.drop(dropVariables,axis=1,inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

# One Hot Encoding

# This library helps to create a variable for each variable groups.
# We don't want to change binary variables which we have 2 in the dataset, then we choose except from these are.

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.head()

# Label Encoding

# One of the two binary variable has already described as binary label but the other one needs to change to binary label.
df['GlucoseGroup'] = LabelEncoder().fit_transform(df['GlucoseGroup'])

## Modelling

# Dataset had been splitted as train and test before the model was created.
# Test data size was taken as %30 of the whole dataset. y shows the dependent variable which is Outcome.

y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# I'll use 2 different type model but 3 different techniques on modelling part which are random forest, logistic regression with balanced and
# smote before logistic regression and I'll decide to which one is better from others.

def confusionMatrixPlot(y, y_pred, model):
    """
    This function helps to display confusion matrix in a graphic. Confusion matrix doesn't show a class label of the variable and accuracy score.
    In this function helps to see all of the information in the graphic.

    Parameters
    ----------
    y: Dependent variable values on the dataset.
    y_pred: Prediction of the dependent variable from the model.
    model: Saved model after fitting.

    Examples
     ------
       confusionMatrixPlot(y, y_pred)

    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()



# Random Forest

# Original dependent variable does not look too much imbalanced. But recall value which is an important on this project is very low.
# Our model has more accurate prediction on non-diabetic persons. It means, patient can be diabetic but model says non-diabetic and we don't want this situation.

rfModel = RandomForestClassifier(random_state=42).fit(X_train, y_train)
rf_y_pred = rfModel.predict(X_test)

# Accuracy score of the random forest model is % 73
#                precision    recall  f1-score   support
#           0       0.79      0.81      0.80       151
#           1       0.62      0.60      0.61        80
# True label 0  [122,  29],
#            1  [ 32,  48]]
# Predicted label  0   1

accuracy_score(y_test, rf_y_pred) # 0.7359
confusion_matrix(y_test, rf_y_pred)
print('\n',classification_report(y_test, rf_y_pred))
confusionMatrixPlot(y_test, rf_y_pred,rfModel)

# Logistic Regression

# We said that the dependent variable does not seem unbalanced, but the confusion matrix result shows that it should be balanced.
# The logistic regression model has its own weight technique.
# This model's predictions are better on diabetic persons.

lrModel = LogisticRegression(random_state=42, class_weight="balanced", solver='lbfgs').fit(X_train, y_train)
lr_y_pred = lrModel.predict(X_test)

# Accuracy score of the balanced logistic regression model is % 67
#                precision    recall  f1-score   support
#           0       0.82      0.65      0.72       151
#           1       0.52      0.72      0.61        80
# True label 0  [98,  53],
#            1  [22,  58]]
# Predicted label  0   1

confusion_matrix(y_test, lr_y_pred)
accuracy_score(y_test, lr_y_pred) #0.6753
print('\n',classification_report(y_test, lr_y_pred))
confusionMatrixPlot(y_test, lr_y_pred, lrModel)

# SMOTE + Logistic Regression

# SMOTE method helps to oversampling the small category of the variable.
# After the smoothing method, model did more accurate predictions on diabetic patients.
# Precision value less accurate comparing with random forest but recall value is more important than the recall on this problem.

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=42)

X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
np.bincount(y_train_sm) # [349, 349] before smoothing it was [349, 188]

# from collections import Counter    -> I could use this method instead of bincount.
# print('Resampled dataset shape %s' % Counter(y_train_sm))


smoteLrModel = LogisticRegression(random_state=42).fit(X_train_sm, y_train_sm)
smo_lr_pred = smoteLrModel.predict(X_test)

# Accuracy score of the smooth logistic regression model is % 71
#                precision    recall  f1-score   support
#           0       0.84      0.69      0.76       151
#           1       0.56      0.75      0.64        80
# True label 0  [104,  47],
#            1  [20,  60]]
# Predicted label  0   1

confusion_matrix(y_test, smo_lr_pred)
accuracy_score(y_test, smo_lr_pred) #0.7099
print('\n',classification_report(y_test, smo_lr_pred))
confusionMatrixPlot(y_test, smo_lr_pred, smoteLrModel)


