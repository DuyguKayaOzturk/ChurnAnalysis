

# Problem:

# **Can you develop a machine learning model that can predict customers leaving the company??**

# - The objective is to predict whether a bank's customers will leave the bank or not.

# - The defining event of customer churn is the closing of the customer's bank account..

# **Dataset Story:**

# - Consists of 10000 observations and 12 variables.
# - Independent variables contain information about customers.
# - The dependent variable represents the customer abandonment status..

# **Variables:**

# - Surname
# - CreditScore
# - Geography : Country (Germany/France/Spain)
# - Gender  (Female/Male)
# - Age
# - Tenure : KHow many years of customers
# - Balance : Account balance
# - NumOfProducts : Number of bank products used
# - HasCrCard : Credit card status (0=No,1=Yes)
# - IsActiveMember : Active membership status (0=No,1=Yes)
# - EstimatedSalary : Estimated salary
# - Exited : Churn or Not? (0=No,1=Yes)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

%config InlineBackend.figure_format = 'retina'

churn = pd.read_csv("ChurnProject/DataSets/churn.csv", index_col=0)
df=churn.copy()

def general(dataframe):
    print(dataframe.head())
    print(dataframe.shape)
    print(dataframe.info())
    print(dataframe.columns)
    print(dataframe.isnull().values.any())

general(df)

df.nunique()

df.drop(["Surname","CustomerId"],inplace = True, axis =1) # (I dropped unnecessary variables)
df=df.reset_index(drop=True)
df.describe().T

# Target Analyze

df.groupby("Exited").size()
df["Exited"].value_counts()*100/len(df)

sns.countplot(df["Exited"], label="Count")
plt.show()


plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# VARIABLES
obj_list = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Tenure']
def cat_summary (data ,categorical_cols ,target ,number_of_classes=10):
    var_count =0
    vars_more_classes =[]
    for var in categorical_cols:
        if len(df[var].value_counts() )<=number_of_classes:
            print(pd.DataFrame({var:data[var].value_counts(),
                                "Ratio": 100 *(data[var].value_counts( ) /len(data)),
                                "Target_Median": data.groupby(var)[target].median()}) ,end="\n\n\n")
            var_count +=1
        else:
            vars_more_classes.append(data[var].name)

    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cat_summary(df ,obj_list ,"Exited")



for col in obj_list:
    df[col] = df[col].astype('category')

num_list = df.columns.difference(obj_list)


# Categorical Variables

obj_list

df.groupby("IsActiveMember").median()
df['IsActiveMember'].corr(df['HasCrCard']) # There is no meaningful relationship between these two.

df.groupby("NumOfProducts").median()
df["NumOfProducts"].value_counts()
df[df["NumOfProducts"]==1]["Exited"].mean() # 0.27
df[df["NumOfProducts"]==2]["Exited"].mean() # 0.07
df[df["NumOfProducts"]==3]["Exited"].mean() # 0.82
df[df["NumOfProducts"]==4]["Exited"].mean() # 1.0

# Customers with more than two products are likely to leave. That's why I created a new var and grouped.
df["NOP"]=pd.Categorical(df["NumOfProducts"], ordered=True)
df.loc[(df["NOP"]==1), "NOP"]= 2
df.loc[(df["NOP"]==2), "NOP"]= 1
df.loc[(df["NOP"]>2), "NOP"]= 3
df.head(3)


df.groupby("Geography").median()
df.groupby("Gender").median()
df.head(5)

df.groupby("HasCrCard").median()
df["HasCrCard"].value_counts()*100/len(df)
sns.countplot(df['HasCrCard'],label="Count")
plt.show()

# Numerical Variables

num_list

# Age

df["Age"].describe().T
sns.boxplot(df["Age"])
plt.show()

df.loc[(df['Age'] <= 25), "NEW_Age"] = 'Young'
df.loc[((df['Age'] > 25) & (df['Age'] <= 45)), "NEW_Age"] = 'Adult1'
df.loc[((df['Age'] > 45) & (df['Age'] <= 60)), "NEW_Age"] = 'Adult2'
df.loc[(df['Age'] > 60), "NEW_Age"] = 'Honored'

# Balance

df["Balance"].describe([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
sns.boxplot(df["Balance"])
plt.show()

df[df["Balance"]==0]["Exited"].mean()
df[df["Balance"]!=0]["Exited"].mean()

df["NewBalance"] = df["Balance"]
df.loc[(df["Balance"]==0), ["NewBalance"]] = "Zero"
df.loc[(df["Balance"]>0) & (df["Balance"]<=60000), ["NewBalance"]] = "0-60K"
df.loc[(df["Balance"]>60000) & (df["Balance"]<=100000), ["NewBalance"]] = "60K-100K"
df.loc[(df["Balance"]>100000) & (df["Balance"]<=150000), ["NewBalance"]] = "100K-150K"
df.loc[(df["Balance"]>150000) & (df["Balance"]<=200000), ["NewBalance"]] = "150K-200K"
df.loc[(df["Balance"]>200000) , ["NewBalance"]] = "200+"

# CreditScore

df["CreditScore"].describe([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
sns.boxplot(df["CreditScore"])
plt.show()

(df["CreditScore"]<400).sum()
df[df["CreditScore"]<400].apply({"Exited": "value_counts"}) # CreditScore<400 are all churn

df["NewCreditScore"]=df["CreditScore"]
df.loc[(df["CreditScore"]<700),["NewCreditScore"]] = "HighRisk"
df.loc[(df["CreditScore"]>=700)&(df["CreditScore"]<1100),["NewCreditScore"]] = "MidRisk"
df.loc[(df["CreditScore"]>=1100)&(df["CreditScore"]<1500),["NewCreditScore"]] = "LowRisk"
df.loc[(df["CreditScore"]>=1500)&(df["CreditScore"]<1700),["NewCreditScore"]] = "Good"
df.loc[(df["CreditScore"]>=1700)&(df["CreditScore"]<1900),["NewCreditScore"]] = "VeryGood"


# PossOfUsingLoan (New Var)
PossOfUsingLoan=pd.Series(["VeryLow", "Low","DependsSalary","High","VeryHigh"], dtype = "category")
df["PossOfUsingLoan"]=PossOfUsingLoan
df.loc[(df["NewCreditScore"]=="HighRisk"),"PossOfUsingLoan"] = PossOfUsingLoan[0]
df.loc[(df["NewCreditScore"]=="MidRisk"),"PossOfUsingLoan"] = PossOfUsingLoan[1]
df.loc[(df["NewCreditScore"]=="LowRisk"),"PossOfUsingLoan"] = PossOfUsingLoan[2]
df.loc[(df["NewCreditScore"]=="Good"),"PossOfUsingLoan"] = PossOfUsingLoan[3]
df.loc[(df["NewCreditScore"]=="VeryGood"),"PossOfUsingLoan"] = PossOfUsingLoan[4]

# EstimatedSalary

df["EstimatedSalary"].describe([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
sns.boxplot(df["EstimatedSalary"])
plt.show()

num_list

# Drop Feat.
df.head(2)
df.drop(["Age","CreditScore","Balance"],inplace = True, axis =1)



# ONE-HOT ENCODING

def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    data = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [col for col in data.columns if col not in original_columns]
    return data, new_columns

df, new_columns_ohe = one_hot_encoder(dataframe=df, categorical_cols=obj_list, nan_as_category=False)
df.head()

df=pd.get_dummies(df,columns =["NEW_Age","NOP", "NewCreditScore","PossOfUsingLoan","NewBalance"], drop_first = True)
df.head(2)

# Modelling

X = df.drop(["Exited"],axis=1)
y = df[["Exited"]]

# Train and test data separation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12345)


models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=100, random_state=12345)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), np.std(cv_results))
    print(msg)

fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
ax.set_xticklabels(names)
plt.boxplot(results)
plt.show()

# Random Forest(RF)


rf_model = RandomForestClassifier().fit(X_train,y_train)
print('RF accuracy: {:.3f}'.format(accuracy_score(y_test, rf_model.predict(X_test))), '\n')
print(classification_report(y_test, rf_model.predict(X_test)))
rf_model_y_pred = rf_model.predict(X_test)
rf_model_cv = cross_val_score(rf_model,X,y,cv=10).mean()


rf_params = {"n_estimators": [200,500],
             "max_features": [3,5],
             "min_samples_split": [5,10],
             "max_depth": [3,5, None]}

rf_model=RandomForestClassifier(random_state=12345)

gs_cv = GridSearchCV(rf_model,
                     rf_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

gs_cv.best_params_

rf_tuned = RandomForestClassifier(**gs_cv.best_params_).fit(X, y)
print(cross_val_score(rf_tuned, X, y, cv=10).mean())
print('RF accuracy: {:.3f}'.format(accuracy_score(y_test, rf_tuned.predict(X_test))), '\n')
print(classification_report(y_test,rf_tuned.predict(X_test)))

# LGBM

lgbm_model = LGBMClassifier().fit(X_train,y_train)
lgbm_model_y_pred = lgbm_model.predict(X_test)
lgbm_model_cv=cross_val_score(lgbm_model, X, y, cv=10).mean()
print('RF accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm_model.predict(X_test))), '\n')
print(classification_report(y_test, lgbm_model.predict(X_test)))



lgbm_params = {"learning_rate": [0.04],
               "max_depth": [3],
               "n_estimators": [438]
               }

lgbm_model=LGBMClassifier(random_state=12345)

gs_cv = GridSearchCV(lgbm_model,
                     lgbm_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

gs_cv.best_params_

lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X, y)
print(cross_val_score(lgbm_tuned, X, y, cv=10).mean())
print('LGBM accuracy: {:.3f}'.format(accuracy_score(y_test, lgbm_tuned.predict(X_test))), '\n')
print(classification_report(y_test,lgbm_tuned.predict(X_test)))


# XGB

XGB_model = XGBClassifier().fit(X_train,y_train)
XGB_model_y_pred = XGB_model.predict(X_test)
XGB_model_cv=cross_val_score(XGB_model, X, y, cv=10).mean()
print('XGB accuracy: {:.3f}'.format(accuracy_score(y_test, XGB_model.predict(X_test))), '\n')
print(classification_report(y_test, XGB_model.predict(X_test)))

XGB_params = {"learning_rate": [0.06],
               "max_depth": [6],
               "min_split_loss": [10],
               "min_child_weight": [3]
               }

XGB_model=XGBClassifier(random_state=12345)

gs_cv = GridSearchCV(XGB_model,
                     XGB_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

gs_cv.best_params_

XGB_tuned = XGBClassifier(**gs_cv.best_params_).fit(X, y)
print(cross_val_score(XGB_tuned, X, y, cv=10).mean())
print('XGB accuracy: {:.3f}'.format(accuracy_score(y_test, XGB_tuned.predict(X_test))), '\n')
print(classification_report(y_test,XGB_tuned.predict(X_test)))

