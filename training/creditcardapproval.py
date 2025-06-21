#!/usr/bin/env python
# coding: utf-8

# # Visualizing and Analysing the data

# ## Importing The Libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


# ## Read the dataset

# In[5]:


app = pd.read_csv('Images/application_record.csv')
credit = pd.read_csv('Images/credit_record.csv')


# In[6]:


app.head()


# In[7]:


credit.head()


# ## Univariate Analysis

# In[9]:


print("Number of people working status: ")
print(app['OCCUPATION_TYPE'].value_counts())
sns.set(rc = {'figure.figsize':(18,6)})
sns.countplot(x='OCCUPATION_TYPE', data=app, palette = 'Set2')


# In[10]:


print("Types of house of the peoples :")
print(app['NAME_HOUSING_TYPE'].value_counts())
sns.set(rc = {'figure.figsize':(15,4)})
sns.countplot(x="NAME_HOUSING_TYPE", data = app, palette='Set2')


# In[11]:


print("Income Types of the Person :")
print(app['NAME_INCOME_TYPE'].value_counts())
sns.set(rc = {'figure.figsize':(8,5)})
sns.countplot(x="NAME_INCOME_TYPE", data = app, palette='Set2')


# ## Multivariate Analysis

# In[13]:


# fig,ax = plt.subplots(figsize=(8,6))
# sns.heatmap(app.corr(), annot=True)
#Error in the given cell


# ## Descriptive Analysis

# In[15]:


app.describe()


# # Data Pre-Processing

# ## Drop Unwanted Features

# In[18]:


# dropping duplicate rows
app.drop_duplicates(
    subset=[
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
        'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'
    ],
    keep='first',
    inplace=True
)


# ## Handling Missing Values

# In[20]:


app.isnull().mean()


# ## Data Cleaning and Merging

# In[22]:


def data_cleansing(data):
    data['CNT_FAM_MEMBERS'] = data['CNT_FAM_MEMBERS'] + data['CNT_CHILDREN']

    dropped_cols = ['FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
                    'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_CHILDREN']
    data = data.drop(dropped_cols, axis=1)

    data['DAYS_BIRTH'] = np.abs(data['DAYS_BIRTH']) / 365 
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'] / 365


    housing_type = {
        'House / apartment': 'House / apartment',
        'With parents': 'With parents',
        'Municipal apartment': 'House / apartment',
        'Rented apartment': 'House / apartment',
        'Office apartment': 'House / apartment',
        'Co-op apartment': 'House / apartment'
    }

    income_type = {
        'Commercial associate': 'Working',
        'State servant': 'Working',
        'Working': 'Working',
        'Pensioner': 'Pensioner',
        'Student': 'Student'
    }

    education_type = {
        'Secondary / secondary special': 'secondary',
        'Lower secondary': 'secondary',
        'Higher education': 'Higher education',
        'Incomplete higher': 'Higher education',
        'Academic degree': 'Academic degree'
    }

    family_status = {
        'Single / not married': 'Single',
        'Separated': 'Single',
        'Widow': 'Single',
        'Civil marriage': 'Married',
        'Married': 'Married'
    }

    data['NAME_HOUSING_TYPE'] = data['NAME_HOUSING_TYPE'].map(housing_type)
    data['NAME_INCOME_TYPE'] = data['NAME_INCOME_TYPE'].map(income_type)
    data['NAME_EDUCATION_TYPE'] = data['NAME_EDUCATION_TYPE'].map(education_type)
    data['NAME_FAMILY_STATUS'] = data['NAME_FAMILY_STATUS'].map(family_status)

    return data


# In[23]:


credit.head()


# In[24]:


credit.shape


# In[25]:


credit.info()


# In[26]:


grouped = credit.groupby('ID')

pivot_tb = credit.pivot(index='ID', columns='MONTHS_BALANCE', values='STATUS')
pivot_tb['open_month'] = grouped['MONTHS_BALANCE'].min()
pivot_tb['end_month'] = grouped['MONTHS_BALANCE'].max()
pivot_tb['window'] = pivot_tb['end_month'] - pivot_tb['open_month']
pivot_tb['window'] = pivot_tb['window'] + 1 

pivot_tb['paid_off'] = pivot_tb[pivot_tb.iloc[:, 0:61] == 'C'].count(axis=1)
pivot_tb['pastdue_1–29'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '0'].count(axis=1)
pivot_tb['pastdue_30–59'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '1'].count(axis=1)
pivot_tb['pastdue_60–89'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '2'].count(axis=1)
pivot_tb['pastdue_90–119'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '3'].count(axis=1)
pivot_tb['pastdue_120–149'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '4'].count(axis=1)
pivot_tb['pastdue_over_150'] = pivot_tb[pivot_tb.iloc[:, 0:61] == '5'].count(axis=1)
pivot_tb['no_loan'] = pivot_tb[pivot_tb.iloc[:, 0:61] == 'X'].count(axis=1)

pivot_tb['ID'] = pivot_tb.index


# In[27]:


pivot_tb.head(10)


# ## Feature Engineering

# In[29]:


def feature_engineering_target(data):
    good_or_bad = []
    for index, row in data.iterrows():
        paid_off = row['paid_off']
        over_1 = row['pastdue_1–29']
        over_30 = row['pastdue_30–59']
        over_60 = row['pastdue_60–89']
        over_90 = row['pastdue_90–119']
        over_120 = row['pastdue_120–149'] + row['pastdue_over_150']
        no_loan = row['no_loan']

        overall_pastdues = over_1 + over_30 + over_60 + over_90 + over_120

        if overall_pastdues == 0:
            if paid_off >= no_loan or paid_off <= no_loan:
                good_or_bad.append(1)
            elif paid_off == 0 and no_loan == 1:
                good_or_bad.append(1)

        elif overall_pastdues != 0:
            if paid_off > overall_pastdues:
                good_or_bad.append(1)
            elif paid_off <= overall_pastdues:
                good_or_bad.append(0)

        elif paid_off == 0 and no_loan != 0:
            if overall_pastdues <= no_loan or overall_pastdues >= no_loan:
                good_or_bad.append(0)

        else:
            good_or_bad.append(1)

    return good_or_bad


# In[30]:


cleansed_app = data_cleansing(app)
target = pd.DataFrame()
target['ID'] = pivot_tb.index
target['paid_off'] = pivot_tb['paid_off'].values
target['#_of_pastdues'] = pivot_tb['pastdue_1–29'].values + pivot_tb['pastdue_30–59'].values \
                        + pivot_tb['pastdue_60–89'].values + pivot_tb['pastdue_90–119'].values \
                        + pivot_tb['pastdue_120–149'].values + pivot_tb['pastdue_over_150'].values

target['no_loan'] = pivot_tb['no_loan'].values
target['target'] = feature_engineering_target(pivot_tb)

credit_app = cleansed_app.merge(target, how='inner', on='ID')
credit_app.drop('ID', axis=1, inplace=True)


# ## Handling Categorical Values

# In[32]:


from sklearn.preprocessing import LabelEncoder

cg = LabelEncoder()
oc = LabelEncoder()
own_r = LabelEncoder()
it = LabelEncoder()
et = LabelEncoder()
fs = LabelEncoder()
ht = LabelEncoder()

credit_app['CODE_GENDER'] = cg.fit_transform(credit_app['CODE_GENDER'])
credit_app['FLAG_OWN_CAR'] = oc.fit_transform(credit_app['FLAG_OWN_CAR'])
credit_app['FLAG_OWN_REALTY'] = own_r.fit_transform(credit_app['FLAG_OWN_REALTY'])
credit_app['NAME_INCOME_TYPE'] = it.fit_transform(credit_app['NAME_INCOME_TYPE'])
credit_app['NAME_EDUCATION_TYPE'] = et.fit_transform(credit_app['NAME_EDUCATION_TYPE'])
credit_app['NAME_FAMILY_STATUS'] = fs.fit_transform(credit_app['NAME_FAMILY_STATUS'])
credit_app['NAME_HOUSING_TYPE'] = ht.fit_transform(credit_app['NAME_HOUSING_TYPE'])


# In[33]:


print("CODE_GENDER", credit_app['CODE_GENDER'].unique())
print(cg.inverse_transform(list(credit_app['CODE_GENDER'].unique())))
print()

print("FLAG_OWN_CAR:", credit_app['FLAG_OWN_CAR'].unique())
print(oc.inverse_transform(list(credit_app['FLAG_OWN_CAR'].unique())))
print()

print("FLAG_OWN_REALTY", credit_app['FLAG_OWN_REALTY'].unique())
print(own_r.inverse_transform(list(credit_app['FLAG_OWN_REALTY'].unique())))
print()

print("NAME_INCOME_TYPE", credit_app['NAME_INCOME_TYPE'].unique())
print(it.inverse_transform(list(credit_app['NAME_INCOME_TYPE'].unique())))
print()

print("NAME_EDUCATION_TYPE", credit_app['NAME_EDUCATION_TYPE'].unique())
print(et.inverse_transform(list(credit_app['NAME_EDUCATION_TYPE'].unique())))
print()

print("NAME_FAMILY_STATUS", credit_app['NAME_FAMILY_STATUS'].unique())
print(fs.inverse_transform(list(credit_app['NAME_FAMILY_STATUS'].unique())))
print()

print("NAME_HOUSING_TYPE", credit_app['NAME_HOUSING_TYPE'].unique())
print(ht.inverse_transform(list(credit_app['NAME_HOUSING_TYPE'].unique())))
print()


# In[34]:


credit_app


# ## Splitting Data into Train and Test

# In[36]:


x = credit_app[credit_app.drop('target', axis=1).columns]
y=credit_app['target']
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8, random_state = 42)


# # Model Building

# ## Logistic Regression Model

# In[39]:


def logistic_reg(xtrain, xtest, ytrain, ytest):
    lr = LogisticRegression(solver='liblinear')
    lr.fit(xtrain, ytrain)
    ypred = lr.predict(xtest)
    print('***LogisticRegression***')
    print('Confusion Matrix')
    print(confusion_matrix(ytest, ypred))
    print('Classification report')
    print(classification_report(ytest, ypred))


# ## Random Forest Classifier

# In[41]:


def random_forest(xtrain, xtest, ytrain, ytest):
    rf = RandomForestClassifier()
    rf.fit(xtrain, ytrain)
    ypred = rf.predict(xtest)
    print('***RandomForestClassifier***')
    print('Confusion Matrix')
    print(confusion_matrix(ytest, ypred))
    print('Classification report')
    print(classification_report(ytest, ypred))


# ## Xgboost Model

# In[43]:


def g_boosting(xtrain, xtest, ytrain, ytest):
    gb=GradientBoostingClassifier()
    gb.fit(xtrain, ytrain)
    ypred = gb.predict(xtest)
    print('***GradientBoostingClassifier***')
    print('Confusion Matrix')
    print(confusion_matrix(ytest, ypred))
    print('Classification report')
    print(classification_report(ytest, ypred))


# ## Decision Tree Model

# In[45]:


def d_tree(xtrain, xtest, ytrain, ytest):
    dt = DecisionTreeClassifier()
    dt.fit(xtrain, ytrain)
    ypred = dt.predict(xtest)
    print('***DecisionTreeClassifier***')
    print('Confusion Matrix')
    print(confusion_matrix(ytest, ypred))
    print('Classification report')
    print(classification_report(ytest, ypred))


# ## Comparing the models

# In[47]:


def compare_model(xtrain, xtest, ytrain, ytest):
    logistic_reg(xtrain, xtest, ytrain, ytest)
    print('-'*100)
    random_forest(xtrain, xtest, ytrain, ytest)
    print('-'*100)
    g_boosting(xtrain, xtest, ytrain, ytest)
    print('-'*100)
    d_tree(xtrain, xtest, ytrain, ytest)


# In[48]:


compare_model(xtrain, xtest, ytrain, ytest)


# In[49]:


#Using decision tree classifier as per the score


# In[50]:


dt = DecisionTreeClassifier()
dt.fit(xtrain, ytrain)
ypred = dt.predict(xtest)


# In[51]:


import pickle


# In[52]:


pickle.dump(dt, open("model.pkl", "wb"))


# In[ ]:




