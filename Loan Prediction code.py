#Factors influencing loan prediction
#Salary                     Loan term
#Previous history           EMI
#Loan amount


#LOADING LIBRARIES
import pandas as pd                    # for working with data
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs
import warnings                        # for throwing exceptions       
warnings.filterwarnings("ignore")      # To ignore any warnings
from sklearn.model_selection import train_test_split    #To split training data into train and validate
from sklearn.linear_model import LogisticRegression     #To build Logistic regression model
from sklearn import tree                                #To build Decision trees
from sklearn.ensemble import RandomForestClassifier     #To build Random forest
from xgboost import XGBClassifier                       #To build XGB Classifier
from sklearn.metrics import accuracy_score              #Accuracy metric
from sklearn.model_selection import StratifiedKFold     #To perform Kfold cross validation


##########################################################################################################################################################

#Loading Dataset
train=pd.read_csv("Training_dataset.txt") 

#Making copies of dataset
train_original=train.copy() 

#Data types
print("DATA TYPES OF EACH FEATURE INVOLVED IN DATASET:")
print(train.dtypes)
print("\nLoan_Status is our target variable")
print("\n\nDESCRIPTION ABOUT DATA TYPES INVOLLVED IN DATASET:")
print("object: Instance of a class")
print("int64: It represents the integer variables")
print("float64: It represents the variable which have some decimal values involved")

#Dimensionality of train dataset
print("\n\nDimensionality of train dataset")
print(train.shape)


##########################################################################################################################################################

#UNIVARIATE ANALYSIS- Summarizes individual data; No comparison
print("\n\nUNIVARIATE ANALYSIS")

#Visualizing dependent variable - Loan_Status
print("\nLOAN_STATUS - FREQUENCY")
print(train['Loan_Status'].value_counts())
#To print frequency in probability
print("LOAN_STATUS - PROPORTION")
print(train['Loan_Status'].value_counts(normalize=True))
#Plotting Loan_Status frequency
train['Loan_Status'].value_counts().plot.bar(title = 'Loan_Status')
plt.show()



#Visualizing Categorical independent features
#2X2 graph plot of all categorical features
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender') 
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
#Displaying graph
plt.show()



#Visualizing Ordinal independent features
    #ORDINAL FEATURES-> Categorial features that can be arranged in some order/hierarchy
#1X3 graph plot of all categorical features
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(title= 'Dependents') 
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education') 
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')
#Displaying graph
plt.show()



#Visualizing Numerical independent features
#1X2 graph plot of Applicant Income
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train.boxplot(column = 'ApplicantIncome') 
plt.show()
    #Most of the data in the distribution of applicant income is towards left - Not normally distributed
    #Boxplot confirms the presence of a lot of outliers/extreme values - Income disparity among social classes
#Boxplot of Applicant Income segregated based on "Education"
train.boxplot(column = 'ApplicantIncome' , by = 'Education') 
plt.show()

#1X2 graph plot of Coapplicant Income
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train.boxplot(column = 'CoapplicantIncome') 
plt.show()
    #Most of the data in the distribution of applicant income is towards left - Not normally distributed
    #Boxplot confirms the presence of a lot of outliers/extreme values
    #Since we don't know education status of coapplicant, we stop here

#1X2 graph plot of Loan Amount
plt.figure(1)
plt.subplot(121)
#Removing NaN values
df = train.dropna()
sns.distplot(df['LoanAmount'])
plt.subplot(122)
train.boxplot(column = 'LoanAmount') 
plt.show()
    #Distribution is fairly normal
    #Lots of outliers here too
#OUTLIERS-> GREATER THAN UPPER INNNER FENCE /SMALLR THAN LOWER INNER FENCE


##############################################################################################################################################

#BIVARIATE ANALYSIS-> Comparison between 2 variables; dependent and non-independent
print("\n\n\n\nBIVARIATE ANALYSIS")



#Categorical Independent Variable vs Target Variable
print("\n\nCategorical Independent Variable vs Target Variable")
    #Split up of Loan_Status based on Gender
Gender_crosstab = pd.crosstab(train['Gender'],train['Loan_Status'])
print("\nSplit up of Loan_Status based on Gender")
print(Gender_crosstab)
#Finding probability split up of Loan_Status based on Gender 
print("\nSplit up of Loan_Status based on Gender")
    #Axis 1 => Y/N ; Axis 0 => Male/Female
    #Getting probability of each loan status for each gender 
Gender_prob = Gender_crosstab.div(Gender_crosstab.sum(axis = 1).astype(float), axis=0)
print(Gender_prob)
#Plotting Graph
Gender_crosstab.div(Gender_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Gender Status vs Loan Status')
plt.show()

    #Split up of Loan_Status based on Married or not
Married_crosstab = pd.crosstab(train['Married'],train['Loan_Status'])
print("\n\nSplit up of Loan_Status based on Married or not")
print(Married_crosstab)
#Finding probability split up of Loan_Status based on Married or not
print("\nSplit up of Loan_Status based on Married or not - Probabillity")
    #Axis 1 => Y/N ; Axis 0 => Married/Unmarried
    #Getting probability of each loan status for each marriage status 
Married_prob = Married_crosstab.div(Married_crosstab.sum(axis = 1).astype(float), axis=0)
print(Married_prob)
#Plotting Graph
Married_crosstab.div(Married_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Married Status vs Loan Status')
plt.show()

    #Split up of Loan_Status based on Education
Education_crosstab = pd.crosstab(train['Education'],train['Loan_Status'])
print("\n\nSplit up of Loan_Status based on Education")
print(Education_crosstab)
#Finding probability split up of Loan_Status based on Education
print("\nSplit up of Loan_Status based on Education - Probabillity")
    #Axis 1 => Y/N ; Axis 0 => Graduated/Ungraduated
    #Getting probability of each loan status for each education status
Education_prob = Education_crosstab.div(Education_crosstab.sum(axis = 1).astype(float), axis=0)
print(Education_prob)
#Plotting Graph
Education_crosstab.div(Education_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Education Status vs Loan Status')
plt.show()

    #Split up of Loan_Status based on Selfemployed or not
Selfemployed_crosstab = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
print("\n\nSplit up of Loan_Status based on Selfemployed or not")
print(Selfemployed_crosstab)
#Finding probability split up of Loan_Status based on Selfemployed or not
print("\nSplit up of Loan_Status based on Self employed or Not - Probabillity")
    #Axis 1 => Y/N ; Axis 0 => Selfemployed/Not Selfemployed
    #Getting probability of each loan status for each self employment status 
Selfemployed_prob = Selfemployed_crosstab.div(Selfemployed_crosstab.sum(axis = 1).astype(float), axis=0)
print(Selfemployed_prob)
#Plotting Graph
Selfemployed_crosstab.div(Selfemployed_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Selfemployed Status vs Loan Status')
plt.show()

    #Split up of Loan_Status based on Number of dependents
Dependents_crosstab = pd.crosstab(train['Dependents'],train['Loan_Status'])
print("\n\nSplit up of Loan_Status based on number of Dependents")
print(Dependents_crosstab)
#Finding probability split up of Loan_Status based on number of Dependents
print("\nSplit up of Loan_Status based on number of Dependents - Probabillity")
    #Axis 1 => Y/N ; Axis 0 => Number of Dependents
    #Getting probability of each loan status for each number of dependents
Dependents_prob = Dependents_crosstab.div(Dependents_crosstab.sum(axis = 1).astype(float), axis=0)
print(Dependents_prob)
#Plotting Graph
Dependents_crosstab.div(Dependents_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Dependents Status vs Loan Status')
plt.show()

    #Split up of Loan_Status based on Credit History
Credit_History_crosstab = pd.crosstab(train['Credit_History'],train['Loan_Status'])
print("\n\nSplit up of Loan_Status based on Credit History")
print(Credit_History_crosstab)
#Finding probability split up of Loan_Status based on Credit History
print("\nSplit up of Loan_Status based on Credit History - Probabillity")
    #Axis 1 => Y/N ; Axis 0 => Credit History
    #Getting probability of each loan status for each credit history status
Credit_History_prob = Credit_History_crosstab.div(Credit_History_crosstab.sum(axis = 1).astype(float), axis=0)
print(Credit_History_prob)
#Plotting Graph
Credit_History_crosstab.div(Credit_History_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Credit History Status vs Loan Status')
plt.show()

    #Split up of Loan_Status based on Property Area
Property_Area_crosstab = pd.crosstab(train['Property_Area'],train['Loan_Status'])
print("\n\nSplit up of Loan_Status based on Property Area")
print(Property_Area_crosstab)
#Finding probability split up of Loan_Status based on Property Area
print("\nSplit up of Loan_Status based on Property Area - Probabillity")
    #Axis 1 => Y/N ; Axis 0 => Property_Area
    #Getting probability of each loan status for each proprty area category 
Property_Area_prob = Property_Area_crosstab.div(Property_Area_crosstab.sum(axis = 1).astype(float), axis=0)
print(Property_Area_prob)
#Plotting Graph
Property_Area_crosstab.div(Property_Area_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Property Area Status vs Loan Status')
plt.show()



#Numerical Independent Variable vs Target Variable
print("\n\nNumerical Independent Variable vs Target Variable")
    #Split up of Loan_Status based on Income
#Plotting mean income of acccepted and non-accepted loans
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar(title = "Applicant's Income")
plt.show()
print("\nFrom Appliant Income vs Loan Status, we cannot come to a conclusion")
print("So, we create several bins and try comparing the loan status")
#Specifying limits for bins
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
#Segments data into bins
print("Samples of Income after segregating into different bins")
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
print(train['Income_bin'].head(10))
#Split up of Loan_Status based on Income(Bins)
Income_bin_crosstab=pd.crosstab(train['Income_bin'],train['Loan_Status'])
print("\nSplit up of Loan_Status based on Income")
print(Income_bin_crosstab)
#Finding probability split up of Loan_Status based on Income
print("\nSplit up of Loan_Status based on Income - Probability")
    #Axis 1 => Y/N ; Axis 0 => Income_bin
    #Getting probability of each loan status for each Income bin category
Income_prob = Income_bin_crosstab.div(Income_bin_crosstab.sum(axis = 1).astype(float), axis=0)
print(Income_prob)
#Plotting graph
Income_bin_crosstab.div(Income_bin_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Income Status vs Loan Status')
plt.show()
print("It can be inferred that Applicant income does not affect the chances of loan approval")


#Split up of Loan_Status based on Co-Applicant Income
#Specifying limits for bins
bins=[0,1000,3000,42000]
group=['Low','Average','High']
#Segments data into bins
print("Samples of Co-Applicant Income after segregating into different bins")
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
print(train['Coapplicant_Income_bin'].head(10))
#Split up of Loan_Status based on Co-Applicant Income(Bins)
Coapplicant_Income_bin_crosstab=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
print("\nSplit up of Loan_Status based on Coapplicant Income")
print(Coapplicant_Income_bin_crosstab)
#Finding probability split up of Loan_Status based on Income
print("\nSplit up of Loan_Status based on Coapplicant Income - Probability")
    #Axis 1 => Y/N ; Axis 0 => Coapplicant Income
    #Getting probability of each loan status for each Co-applicant bin category
Coapplicant_Income_prob = Coapplicant_Income_bin_crosstab.div(Coapplicant_Income_bin_crosstab.sum(axis = 1).astype(float), axis=0)
print(Coapplicant_Income_prob)
#Plotting graph
Coapplicant_Income_bin_crosstab.div(Coapplicant_Income_bin_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Coapplicant Income Status vs Loan Status')
plt.show()
print("It can be inferred that if coapplicant’s income is less the chances of loan approval are high")
print("""But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant.
So, the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it.
So, we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval
(Nan for some records represents that those applicants don’t have any coapplicant).""")


#Split up of Loan_Status based on Total Income (Applicant and Co-Applicant)
    #Combining Applicant and Co-Applicant's income
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
#Specifying limits for bins
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high'] 
#Segments data into bins
print("Samples of Total Income (Applicant and Co-Applicant) after segregating into different bins")
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
print(train['Total_Income_bin'].head(10))
#Split up of Loan_Status based on Total Income(Bins)
Total_Income_bin_crosstab=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
print("\nSplit up of Loan_Status based on Total Income (Applicant and Co-Applicant)")
print(Total_Income_bin_crosstab)
#Finding probability split up of Loan_Status based on Total Income
print("\nSplit up of Loan_Status based on Coapplicant Income (Applicant and Co-Applicant) - Probability")
    #Axis 1 => Y/N ; Axis 0 => Total Income
    #Getting probability of each loan status for each total applicant bin category
Total_Income_prob = Total_Income_bin_crosstab.div(Total_Income_bin_crosstab.sum(axis = 1).astype(float), axis=0)
print(Total_Income_prob)
#Plotting graph
Total_Income_bin_crosstab.div(Total_Income_bin_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='Total Income Status vs Loan Status')
plt.show()
print("""It can be inferred that proportion of loans getting approved for applicants having low Total Income is the least.""")


#Split up of Loan_Status based on Loan Amount
#Specifying limits for bins
bins=[0,100,200,700]
group=['Low','Average','High']
#Segments data into bins
print("Samples of Loan amount after segregating into different bins")
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
print(train['LoanAmount_bin'].head(10))
#Split up of Loan_Status based on LoanAmount(Bins)
LoanAmount_bin_crosstab=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
print("\nSplit up of Loan_Status based on LoanAmount")
print(LoanAmount_bin_crosstab)
#Finding probability split up of Loan_Status based on LoanAmount
print("\nSplit up of Loan_Status based on LoanAmount - Probability")
    #Axis 1 => Y/N ; Axis 0 => LoanAmount
    #Getting probability of each loan status for each Loan Amount category
LoanAmount_prob = LoanAmount_bin_crosstab.div(LoanAmount_bin_crosstab.sum(axis = 1).astype(float), axis=0)
print(LoanAmount_prob)
#Plotting graph
LoanAmount_bin_crosstab.div(LoanAmount_bin_crosstab.sum(axis = 1).astype(float), axis=0).plot.bar(stacked = True , title='LoanAmount Status vs Loan Status')
plt.show()
print("It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount as compared to that of High Loan Amount")


#Heat Map visulaization
train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
#Changing 3+ dependents to 3
train['Dependents'].replace('3+', 3,inplace=True)
#Changing categorical dependent variable into numerical
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)
#Correlation indicates the extent to which two or more variables fluctuate together.
    #Correlation found only between numerical columns in training dataset
matrix = train.corr()
#f, ax = plt.subplots(figsize=(9, 6))
#Plotting heatmap for the correlation matrix found
    #Anything above 0.8 will be given max colour depth
sns.heatmap(matrix, vmax=.8, square=True, cmap="Blues");
plt.show()
print("""We see that the most correlated variables are
1. ApplicantIncome - LoanAmount
2. Credit_History - Loan_Status""")

##############################################################################################################################################


#Imputing missing values
    #Finding number of missing values in each feature
print("\nNumber of null values in each field")
print(train.isnull().sum())

#Computing missing values for CATEGORICAL FEATURES by finding mode
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

#Computing missing values for CONTINUOUS FEATURES by finding median
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#Removing skewness by taking log transformation
    #Log-> has more effect on larger values
train['LoanAmount_log'] = np.log(train['LoanAmount'])

##############################################################################################################################################

#Building Logistic regression model
    #Dropping ID
train=train.drop('Loan_ID', axis=1) 

#Splitting train dataset into dependent and independent features
X = train.drop('Loan_Status', axis=1) 
y = train.Loan_Status

#Performing One-hot encoding for CATEGORICAL features
X=pd.get_dummies(X)
print("\n\nTraining data after performing One-hot encoding")
train=pd.get_dummies(train)
print(train.head(5))
print("\n\nDummy columns that are automatically generated for the purpose of One hot encoding")
print(train.columns)

#Splitting training data for training and validation
    #70% data for training ; 30% data for validation
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

#Building model 
model = LogisticRegression(random_state = 1)
#Fitting data
model.fit(x_train, y_train)
#Predicting validation data's output using model
pred_cv = model.predict(x_cv)
#Calculating accuracy of predictions made on validation data
print("\n\nAccuracy of the Logistic Regression model built is")
print(accuracy_score(y_cv,pred_cv))

##############################################################################################################################################
#Validating models using Stratified K-fold validation method

#Logistic Regression - Stratified k-folds cross Validation
print("\n\nLogistic Regression - Stratified k-folds cross Validation")
tot_acc = 0
#To make count of iterations
i=1
#Data is split into 5 folds
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
# Indices segregated for training and validation are retrieved using variable as lists
for train_indices,test_indices in kf.split(X,y):
    print('\nIteration {} of kfold {}'.format(i,kf.n_splits))
    #Splitting training data for training and validation based on indices segregated by KF
    xtr,xvl = X.iloc[train_indices],X.iloc[test_indices]     
    ytr,yvl = y.iloc[train_indices],y.iloc[test_indices]
    #Building Logistic Regression model 
    model = LogisticRegression(random_state=1)
    #Fitting data
    model.fit(xtr, ytr)
    #Predicting validation data's output using model
    pred_test = model.predict(xvl)
    #Calculating accuracy of predictions made on validation data
    score = accuracy_score(yvl,pred_test)
    print('accuracy_score',score)     
    i+=1
    #For the purpose of finding mean Accuracy
    tot_acc = tot_acc + score
LR_mean_acc = tot_acc/5
print("\nMean validation accuracy for Logistic Regression - Stratified k-folds cross Validation model is {}" .format(LR_mean_acc))





#Decision Tree - Stratified k-folds cross Validation
print("\n\nDecision Tree - Stratified k-folds cross Validation")
tot_acc = 0
#To make count of iterations
i=1
#Data is split into 5 folds
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
# Indices segregated for training and validation are retrieved using variable as lists
for train_indices,test_indices in kf.split(X,y):
    print('\nIteration {} of kfold {}'.format(i,kf.n_splits))
    #Splitting training data for training and validation based on indices segregated by KF
    xtr,xvl = X.loc[train_indices],X.loc[test_indices]     
    ytr,yvl = y[train_indices],y[test_indices]         
    #Building Decision tree model 
    model = tree.DecisionTreeClassifier(random_state=1)
    #Fitting data
    model.fit(xtr, ytr)
    #Predicting validation data's output using model
    pred_test = model.predict(xvl)
    #Calculating accuracy of predictions made on validation data
    score = accuracy_score(yvl,pred_test)
    print('accuracy_score',score)     
    i+=1
    #For the purpose of finding mean Accuracy
    tot_acc = tot_acc + score
DT_mean_acc = tot_acc/5
print("\nMean validation accuracy for Decision Tree - Stratified k-folds cross Validation model is {}" .format(DT_mean_acc))





#Random Forest - Stratified k-folds cross Validation
print("\n\nRandom Forest - Stratified k-folds cross Validation")
tot_acc = 0
#To make count of iterations
i=1
#Data is split into 5 folds
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
# Indices segregated for training and validation are retrieved using variable as lists
for train_indices,test_indices in kf.split(X,y):
    print('\nIteration {} of kfold {}'.format(i,kf.n_splits))
    #Splitting training data for training and validation based on indices segregated by KF
    xtr,xvl = X.loc[train_indices],X.loc[test_indices]     
    ytr,yvl = y[train_indices],y[test_indices]         
    #Building Decision tree model 
    model = RandomForestClassifier(random_state=1, max_depth=10)
    #Fitting data
    model.fit(xtr, ytr)
    #Predicting validation data's output using model
    pred_test = model.predict(xvl)
    #Calculating accuracy of predictions made on validation data
    score = accuracy_score(yvl,pred_test)
    print('accuracy_score',score)     
    i+=1
    #For the purpose of finding mean Accuracy
    tot_acc = tot_acc + score
RF_mean_acc = tot_acc/5
print("\nMean validation accuracy for Random Forest - Stratified k-folds cross Validation model is {}" .format(RF_mean_acc))





#XGBoost - Stratified k-folds cross Validation
print("\n\nXGBoost - Stratified k-folds cross Validation")
tot_acc = 0
#To make count of iterations
i=1
#Data is split into 5 folds
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
# Indices segregated for training and validation are retrieved using variable as lists
for train_indices,test_indices in kf.split(X,y):
    print('\nIteration {} of kfold {}'.format(i,kf.n_splits))
    #Splitting training data for training and validation based on indices segregated by KF
    xtr,xvl = X.loc[train_indices],X.loc[test_indices]     
    ytr,yvl = y[train_indices],y[test_indices]         
    #Building Decision tree model 
    model = XGBClassifier(random_state=1, n_estimators=50)
    #Fitting data
    model.fit(xtr, ytr)
    #Predicting validation data's output using model
    pred_test = model.predict(xvl)
    #Calculating accuracy of predictions made on validation data
    score = accuracy_score(yvl,pred_test)
    print('accuracy_score',score)     
    i+=1
    #For the purpose of finding mean Accuracy
    tot_acc = tot_acc + score
XGB_mean_acc = tot_acc/5
print("\nMean validation accuracy for XGBoost - Stratified k-folds cross Validation model is {}" .format(XGB_mean_acc))

##############################################################################################################################################

#Comparison of accuracies of four algorithms
print("\n\nMean validation accuracies of four algorithms are listed below")
print("Logistic Regression : " +str(LR_mean_acc))
print("Decision Trees : " +str(DT_mean_acc))
print("Random Forest : " +str(RF_mean_acc))
print("XGBoosting : " +str(XGB_mean_acc))

#Plotting graph for the comparison
Accuracies = {'Logistic Regression' : LR_mean_acc, 'Decision Trees' : DT_mean_acc, 'Random Forest' : RF_mean_acc, 'XGBoosting' : XGB_mean_acc}
plt.bar(x = range(len(Accuracies)), height = list(Accuracies.values()), align = 'center')
plt.xticks(range(len(Accuracies)), list(Accuracies.keys()))
plt.yticks(np.arange(0, 1, 0.05))
plt.title('Accuracy Comparison')
plt.show()
