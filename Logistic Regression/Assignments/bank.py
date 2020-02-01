import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# reading a csv file using pandas library
bank=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Logistic Regression\\Assignments\\bank-full.csv")
##bank.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
bank.columns
##bank.drop(["a"],axis=1,inplace=True)
##bank.columns
# To get the count of null values in the data 
bank.isnull().sum()
bank.shape # 45211 17 => Before dropping null values
# To drop null values ( dropping rows)
bank.dropna().shape ### 45211 17 => After dropping null values
#####Exploratory Data Analysis#########################################################
bank.mean() ## age -40.936210, balance - 1362.272058, day - 15.806419,duration - 258.163080
  ################ campaign - 2.763841, pdays -40.197828, previous -  0.580323
bank.median() ####  age - 39.0, balance - 448.0, day - 16.0,duration - 180.0
  ################ campaign - 2.0, pdays - -1.0, previous -  0.0
bank.mode() 
####Measures of Dispersion
bank.var() 
bank.std() ##  age - 10.618762, balance - 3044.765829, day -  8.322476,duration - 257.527812
  ################ campaign - 3.098021, pdays - 100.128746, previous -  2.303441
#### Calculate the range value
range1 = max(bank['age'])-min(bank['age'])  ### 77
range2 = max(bank['balance'])-min(bank['balance']) ### 110146
range3 = max(bank['day'])-min(bank['day']) ### 30
range4 = max(bank['duration'])-min(bank['duration']) ### 4918
range5 = max(bank['campaign'])-min(bank['campaign'])  ##  62
range6 = max(bank['pdays'])-min(bank['pdays']) ### 872
range7 = max(bank['previous'])-min(bank['previous']) #### 275
### Calculate skewness and Kurtosis
bank.skew() ## age - 0.684818, balance -  8.360308, day - 0.093079,duration - 3.144318
  ################ campaign - 4.898650, pdays - 2.615715, previous -41.846454
bank.kurt() ## age - 0.319570, balance -  140.751547, day - -1.059897,duration - 18.153915
  ################ campaign - 39.249651, pdays - 6.935195, previous - 4506.860660
plt.hist(bank["age"])
plt.hist(bank["balance"])
plt.hist(bank["day"])
plt.hist(bank["duration"])
plt.hist(bank["campaign"])
plt.hist(bank["pdays"])
plt.hist(bank["previous"])
plt.boxplot(bank["age"],0,"rs",0)
plt.boxplot(bank["balance"],0,"rs",0)
plt.boxplot(bank["day"],0,"rs",0)
plt.boxplot(bank["duration"],0,"rs",0)
plt.boxplot(bank["campaign"],0,"rs",0)
plt.boxplot(bank["pdays"],0,"rs",0)
plt.boxplot(bank["previous"],0,"rs",0)
plt.plot(bank["age"],bank["y"],"bo");plt.xlabel("age");plt.ylabel("y")
plt.plot(bank["balance"],bank["y"],"bo");plt.xlabel("balance");plt.ylabel("y")
plt.plot(bank["day"],bank["y"],"bo");plt.xlabel("day");plt.ylabel("y")
plt.plot(bank["duration"],bank["y"],"bo");plt.xlabel("duration");plt.ylabel("y")
plt.plot(bank["campaign"],bank["y"],"bo");plt.xlabel("campaign");plt.ylabel("y")
plt.plot(bank["pdays"],bank["y"],"bo");plt.xlabel("pdays");plt.ylabel("y")
plt.plot(bank["previous"],bank["y"],"bo");plt.xlabel("previous");plt.ylabel("y")
# table 
pd.crosstab(bank["age"],bank["y"])
pd.crosstab(bank["balance"],bank["y"])
pd.crosstab(bank["day"],bank["y"])
pd.crosstab(bank["duration"],bank["y"])
pd.crosstab(bank["campaign"],bank["y"])
pd.crosstab(bank["pdays"],bank["y"])
pd.crosstab(bank["previous"],bank["y"])
## Barplot
pd.crosstab(bank["age"],bank["y"]).plot(kind = "bar", width = 1.85)
pd.crosstab(bank["balance"],bank["y"]).plot(kind = "bar", width = 1.85)
pd.crosstab(bank["day"],bank["y"]).plot(kind = "bar", width = 1.85)
pd.crosstab(bank["duration"],bank["y"]).plot(kind = "bar", width = 1.85)
pd.crosstab(bank["campaign"],bank["y"]).plot(kind = "bar", width = 1.85)
pd.crosstab(bank["pdays"],bank["y"]).plot(kind = "bar", width = 1.85)
pd.crosstab(bank["previous"],bank["y"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="age",data=bank,palette="hls")
sns.countplot(x="balance",data=bank,palette="hls")
sns.countplot(x="day",data=bank,palette="hls")
sns.countplot(x="duration",data=bank,palette="hls")
sns.countplot(x="campaign",data=bank,palette="hls")
sns.countplot(x="pdays",data=bank,palette="hls")
sns.countplot(x="previous",data=bank,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="age",y="y",data=bank,palette="hls")
sns.boxplot(x="balance",y="y",data=bank,palette="hls")
sns.boxplot(x="day",y="y",data=bank,palette="hls")
sns.boxplot(x="duration",y="y",data=bank,palette="hls")
sns.boxplot(x="campaign",y="y",data=bank,palette="hls")
sns.boxplot(x="pdays",y="y",data=bank,palette="hls")
sns.boxplot(x="previous",y="y",data=bank,palette="hls")
sns.pairplot(bank.iloc[:,0:6]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(bank,hue="y",size=2)
bank["age"].value_counts()
bank["balance"].value_counts()
bank["day"].value_counts()
bank["duration"].value_counts()
bank["campaign"].value_counts()
bank["pdays"].value_counts()
bank["previous"].value_counts()
bank["age"].value_counts().plot(kind = "pie")
bank["balance"].value_counts().plot(kind = "pie")
bank["day"].value_counts().plot(kind = "pie")
bank["duration"].value_counts().plot(kind = "pie")
bank["campaign"].value_counts().plot(kind = "pie")
bank["pdays"].value_counts().plot(kind = "pie")
bank["previous"].value_counts().plot(kind = "pie")
sns.pairplot(bank,hue="y",size=4,diag_kind = "kde")
sns.FacetGrid(bank,hue="y").map(plt.scatter,"age","y").add_legend()
sns.FacetGrid(bank,hue="y").map(plt.scatter,"balance","y").add_legend()
sns.FacetGrid(bank,hue="y").map(plt.scatter,"day","y").add_legend()
sns.FacetGrid(bank,hue="y").map(plt.scatter,"duration","y").add_legend()
sns.FacetGrid(bank,hue="y").map(plt.scatter,"campaign","y").add_legend()
sns.FacetGrid(bank,hue="y").map(plt.scatter,"pdays","y").add_legend()
sns.FacetGrid(bank,hue="y").map(plt.scatter,"previous","y").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"age").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"balance").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"day").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"duration").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"campaign").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"pdays").add_legend()
sns.FacetGrid(bank,hue="y").map(sns.kdeplot,"previous").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(bank['age'], dist="norm",plot=pylab)
stats.probplot(np.log(bank['age']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['age']),dist="norm",plot=pylab)
stats.probplot((bank['age'] * bank['age']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['age']),dist="norm",plot=pylab)
reci_1=1/bank['age']
stats.probplot(reci_1,dist="norm",plot=pylab)
stats.probplot(((bank['age'] * bank['age'])+bank['age']),dist="norm",plot=pylab)
stats.probplot(bank['balance'],dist="norm",plot=pylab)
stats.probplot(np.log(bank['balance']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['balance']),dist="norm",plot=pylab)
stats.probplot((bank['balance'] * bank['balance']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['balance']),dist="norm",plot=pylab)
reci_2=1/bank['balance']
stats.probplot(reci_2,dist="norm",plot=pylab)
stats.probplot(((bank['balance'] * bank['balance'])+bank['balance']),dist="norm",plot=pylab)
stats.probplot(bank['day'],dist="norm",plot=pylab)
stats.probplot(np.log(bank['day']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['day']),dist="norm",plot=pylab)
stats.probplot((bank['day'] * bank['day']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['day']),dist="norm",plot=pylab)
reci_3=1/bank['day']
stats.probplot(reci_3,dist="norm",plot=pylab)
stats.probplot(((bank['day'] * bank['day'])+bank['day']),dist="norm",plot=pylab)
stats.probplot(bank['duration'],dist="norm",plot=pylab)
stats.probplot(np.log(bank['duration']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['duration']),dist="norm",plot=pylab)
stats.probplot((bank['duration'] * bank['duration']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['duration']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['duration'])*np.exp(bank['duration']),dist="norm",plot=pylab)
reci_4=1/bank['duration']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((bank['duration'] * bank['duration'])+bank['duration']),dist="norm",plot=pylab)
stats.probplot(bank['campaign'],dist="norm",plot=pylab)
stats.probplot(np.log(bank['campaign']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['campaign']),dist="norm",plot=pylab)
stats.probplot((bank['campaign'] * bank['campaign']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['campaign']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['campaign'])*np.exp(bank['campaign']),dist="norm",plot=pylab)
reci_5=1/bank['campaign']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((bank['campaign'] * bank['campaign'])+bank['campaign']),dist="norm",plot=pylab)
stats.probplot(bank['pdays'],dist="norm",plot=pylab)
stats.probplot(np.log(bank['pdays']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['pdays']),dist="norm",plot=pylab)
stats.probplot((bank['pdays'] * bank['pdays']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['pdays']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['pdays'])*np.exp(bank['pdays']),dist="norm",plot=pylab)
reci_6=1/bank['pdays']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((bank['pdays'] * bank['pdays'])+bank['pdays']),dist="norm",plot=pylab)
stats.probplot(bank['previous'],dist="norm",plot=pylab)
stats.probplot(np.log(bank['previous']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(bank['previous']),dist="norm",plot=pylab)
stats.probplot((bank['previous'] * bank['previous']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['previous']),dist="norm",plot=pylab)
stats.probplot(np.exp(bank['previous'])*np.exp(bank['previous']),dist="norm",plot=pylab)
reci_7=1/bank['previous']
reci_7_2=reci_7 * reci_7
reci_7_4=reci_7_2 * reci_7_2
stats.probplot(reci_7*reci_7,dist="norm",plot=pylab)
stats.probplot(reci_7_2,dist="norm",plot=pylab)
stats.probplot(reci_7_4,dist="norm",plot=pylab)
stats.probplot(reci_7_4*reci_7_4,dist="norm",plot=pylab)
stats.probplot((reci_7_4*reci_7_4)*(reci_7_4*reci_7_4),dist="norm",plot=pylab)
stats.probplot(((bank['previous'] * bank['previous'])+bank['previous']),dist="norm",plot=pylab)

# ppf => Percent point function 
#### age
stats.norm.ppf(0.975,40.936210,10.618762)# similar to qnorm in R ---- 61.748
# cdf => cumulative distributive function 
stats.norm.cdf(bank["age"],40.936210,10.618762) # similar to pnorm in R 
#### balance
stats.norm.ppf(0.975,1362.272058,3044.765829)# similar to qnorm in R ---- 7329.903
# cdf => cumulative distributive function 
stats.norm.cdf(bank["balance"],1362.272058,3044.765829) # similar to pnorm in R 
#### day
stats.norm.ppf(0.975,15.806419,8.322476)# similar to qnorm in R ---- 32.118
# cdf => cumulative distributive function 
stats.norm.cdf(bank["day"],15.806419,8.322476) # similar to pnorm in R 
#### duration
stats.norm.ppf(0.975, 258.163080, 257.527812)# similar to qnorm in R ---- 762.908
# cdf => cumulative distributive function 
stats.norm.cdf(bank["duration"], 258.163080, 257.527812) # similar to pnorm in R 
#### campaign
stats.norm.ppf(0.975, 2.763841, 3.098021)# similar to qnorm in R ---- 8.835
# cdf => cumulative distributive function 
stats.norm.cdf(bank["campaign"], 2.763841, 3.098021) # similar to pnorm in R 
#### pdays
stats.norm.ppf(0.975, 40.197828, 100.128746)# similar to qnorm in R ---- 236.446
# cdf => cumulative distributive function 
stats.norm.cdf(bank["pdays"], 40.197828, 100.128746) # similar to pnorm in R 
#### previous
stats.norm.ppf(0.975,  0.580323, 2.303441)# similar to qnorm in R ---- 5.095
# cdf => cumulative distributive function 
stats.norm.cdf(bank["previous"],  0.580323, 2.303441) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
bank.corr(method = "pearson")
bank.corr(method = "kendall")
bank["balance"].corr(bank["age"]) # # correlation value between X and Y -- 0.0977
bank["balance"].corr(bank["day"])  ### 0.004
bank["balance"].corr(bank["duration"]) ### 0.0215
bank["balance"].corr(bank["campaign"]) ### -0.0145
bank["balance"].corr(bank["pdays"]) #### 0.003
bank["balance"].corr(bank["previous"]) #### 0.0167
bank["age"].corr(bank["day"])  ### -0.009
bank["age"].corr(bank["duration"]) ### -0.004
bank["age"].corr(bank["campaign"]) ### -0.0047
bank["age"].corr(bank["pdays"]) #### -0.023
bank["age"].corr(bank["previous"]) #### 0.0012
np.corrcoef(bank["duration"],bank["day"]) 
np.corrcoef(bank["duration"],bank["campaign"])
np.corrcoef(bank["duration"],bank["pdays"])
np.corrcoef(bank["duration"],bank["previous"])
np.corrcoef(bank["campaign"],bank["day"])
np.corrcoef(bank["campaign"],bank["pdays"])
np.corrcoef(bank["campaign"],bank["previous"])
np.corrcoef(bank["day"],bank["pdays"])
np.corrcoef(bank["day"],bank["previous"])
np.corrcoef(bank["pdays"],bank["previous"])
###### Lets do normalization
a_array = np.array(bank['age'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(bank['balance'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(bank['day'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(bank['duration'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(bank['campaign'])
normalized_E = preprocessing.normalize([e_array])
f_array = np.array(bank['pdays'])
normalized_F = preprocessing.normalize([f_array])
g_array = np.array(bank['previous'])
normalized_G = preprocessing.normalize([g_array])
# to get top 6 rows
bank.head(40) # to get top n rows use cars.head(10)
bank.tail(10)
# Correlation matrix 
bank.corr()
# Scatter plot between the variables along with histograms
sns.pairplot(bank)
pd.tools.plotting.scatter_matrix(bank) ##-> also used for plotting all in one graph
# creating dummy columns for the categorical columns 
bank.columns
bank=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Logistic Regression\\Assignments\\bank-full.csv")
bank_dummies=pd.get_dummies(bank[["job","marital","education","default","housing","loan","contact","month","poutcome","y"]])
bank = pd.concat([bank,bank_dummies],axis=1)
bank.y.value_counts()  ### no-39922, yes-5289
bank.drop(["job"],inplace=True,axis = 1)
bank.drop(["marital"],inplace=True,axis = 1)
bank.drop(["education"],inplace=True,axis = 1)
bank.drop(["default"],inplace=True,axis = 1)
bank.drop(["housing"],inplace=True,axis = 1)
bank.drop(["loan"],inplace=True,axis = 1)
bank.drop(["contact"],inplace=True,axis = 1)
bank.drop(["month"],inplace=True,axis = 1)
bank.drop(["poutcome"],inplace=True,axis = 1)
bank.drop(["y"],inplace=True,axis = 1)
bank.shape
bank.columns
X = bank.iloc[:,[1,2,3,4,5,6]]
Y = bank.iloc[:,0]
classifier = LogisticRegression()
classifier.fit(X,Y)
classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 
y_pred = classifier.predict(X)
bank["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
type(y_pred)
y_pred.shape
accuracy = sum(Y==y_pred)/bank.shape[0] 
pd.crosstab(y_pred,Y)
####Model Building
bank['age']=np.divide(bank['age']-bank['age'].min(),bank['age'].max()-bank['age'].min())
##bank[1]=np.divide(bank[1]-bank[1].min(),bank[1].max()-bank[1].min())
bank['balance']=np.divide(bank['balance']-bank['balance'].min(),bank['balance'].max()-bank['balance'].min())
bank['day']=np.divide(bank['day']-bank['day'].min(),bank['day'].max()-bank['day'].min())
bank['duration']=np.divide(bank['duration']-bank['duration'].min(),bank['duration'].max()-bank['duration'].min())
bank['campaign']=np.divide(bank['campaign']-bank['campaign'].min(),bank['campaign'].max()-bank['campaign'].min())
bank['pdays']=np.divide(bank['pdays']-bank['pdays'].min(),bank['pdays'].max()-bank['pdays'].min())
bank['previous']=np.divide(bank['previous']-bank['previous'].min(),bank['previous'].max()-bank['previous'].min())
import statsmodels.formula.api as sm
bank['y["no"]']=1
bank['y["yes"]']=0
bank_dummies_y=pd.get_dummies(bank[["y"]])
bank_up = pd.concat([bank,bank_dummies_y],axis=1)
bank_up['y_no']=np.divide(bank_up['y_no']-bank_up['y_no'].min(),bank_up['y_no'].max()-bank_up['y_no'].min())
logit_model = sm.logit('y_no~(age+balance+day+duration+campaign+pdays+previous)',data = bank_up).fit()
st.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#summary
logit_model.summary()
y_pred1 = logit_model.predict(bank_up)
bank_up["pred_prob"] = y_pred1
# filling all the cells with zeroes
bank_up["Att_val"] = np.zeros(45211)
bank_up["y_no"]
# Taking threshold value as 0.5 and above the prob value will be treated as correct value 
bank_up.loc[y_pred1>=0.5,"Att_val"] = 1
bank.Att_val
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
bank_up["y_no"]  ### As banks is in continuous form we need to convert it back to original form
classification_report(bank.Att_val,bank_up['y_no'].round())
# confusion matrix 
results = confusion_matrix(bank_up['y_no'].round(),bank_up.Att_val)
# Model Accuracy 
Accuracy = accuracy_score(bank_up['y_no'].round(),bank.Att_val) 
### ROC Curve
fpr, tpr, threshold = metrics.roc_curve(bank_up.y_no.round(), y_pred1)
# the above function is applicable for binary classification class 
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve ---    0.83
### Dividing data into train and test data sets
bank_up.drop("Att_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split
train,test = train_test_split(bank_up,test_size=0.3)
# checking na values 
train.isnull().sum();test.isnull().sum()
# Building a model on train data set 
train_model = sm.logit('y_no~(age+balance+day+duration+campaign+pdays+previous)',data = train).fit()
#summary
train_model.summary()
bank_up['age']=np.divide(bank_up['age']-bank_up['age'].min(),bank_up['age'].max()-bank_up['age'].min())
train_pred = train_model.predict(train.iloc[:,0:])
# Creating new column for storing predicted class of banks
# filling all the cells with zeroes
train["train_pred"] = np.zeros(31647)
# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1
# confusion matrix 
confusion_matrix1 = pd.crosstab(train['y_no'],train.train_pred)
confusion_matrix1
accuracy_train = (27438+666)/(31647) # 88.8%
accuracy_train
# Prediction on Test data set
test_pred = train_model.predict(test)
# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test["test_pred"] = np.zeros(13564)
# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1
# confusion matrix 
confusion_matrix2 = pd.crosstab(test['y_no'],test.test_pred)
confusion_matrix2
accuracy_test = (11832+248)/(13564) #### 89.06%
accuracy_test
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
bank = bank[bank.columns[bank.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
bank = bank.loc[bank.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
bank = bank.fillna(bank.median())
bank['job'].fillna(bank['job'].value_counts().idxmax(), inplace=True)
bank['marital'].fillna(bank['marital'].value_counts().idxmax(), inplace=True)
bank['education'].fillna(bank['education'].value_counts().idxmax(), inplace=True)
bank['default'].fillna(bank['default'].value_counts().idxmax(), inplace=True)
bank['housing'].fillna(bank['housing'].value_counts().idxmax(), inplace=True)
bank['loan'].fillna(bank['loan'].value_counts().idxmax(), inplace=True)
bank['contact'].fillna(bank['contact'].value_counts().idxmax(), inplace=True)
bank['month'].fillna(bank['month'].value_counts().idxmax(), inplace=True)
bank['poutcome'].fillna(bank['poutcome'].value_counts().idxmax(), inplace=True)
bank['y'].fillna(bank['y'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##bank['column_name'].fillna(bank['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = bank['age'].mean () + bank['age'].std () * factor   ### 72.79
lower_lim1= bank['age'].mean () - bank['age'].std () * factor   ### 9.08
bank1 = bank[(bank['age'] < upper_lim1) & (bank['age'] > lower_lim1)]
upper_lim2 = bank['balance'].mean () + bank['balance'].std () * factor  #### 10496.57
lower_lim2 = bank['balance'].mean () - bank['balance'].std () * factor  ## -7772.03
bank2 = bank[(bank['balance'] < upper_lim2) & (bank['balance'] > lower_lim2)]
upper_lim3 = bank['day'].mean () + bank['day'].std () * factor  ### 40.77
lower_lim3 = bank['day'].mean () - bank['day'].std () * factor  ### -9.16
bank3 = bank[(bank['day'] < upper_lim3) & (bank['day'] > lower_lim3)]
upper_lim4 = bank['duration'].mean () + bank['duration'].std () * factor ### 1030.75
lower_lim4 = bank['duration'].mean () - bank['duration'].std () * factor  #####  -514.42
bank4 = bank[(bank['duration'] < upper_lim4) & (bank['duration'] > lower_lim4)]
upper_lim5 = bank['campaign'].mean () + bank['campaign'].std () * factor   #### 12.05
lower_lim5 = bank['campaign'].mean () - bank['campaign'].std () * factor  ### -6.53
bank5 = bank[(bank['campaign'] < upper_lim5) & (bank['campaign'] > lower_lim5)]
upper_lim6 = bank['pdays'].mean () + bank['pdays'].std () * factor    #### 340.58
lower_lim6 = bank['pdays'].mean () - bank['pdays'].std () * factor     ##### -260.18
bank6 = bank[(bank['pdays'] < upper_lim6) & (bank['pdays'] > lower_lim6)]
upper_lim7 = bank['previous'].mean () + bank['previous'].std () * factor   #### 7.49
lower_lim7 = bank['previous'].mean () - bank['previous'].std () * factor  #### -6.33
bank = bank[(bank['previous'] < upper_lim7) & (bank['previous'] > lower_lim7)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim1 = bank['age'].quantile(.95)
lower_lim1 = bank['age'].quantile(.05)
bank1 = bank[(bank['age'] < upper_lim1) & (bank['age'] > lower_lim1)]
upper_lim2 = bank['balance'].quantile(.95)
lower_lim2 = bank['balance'].quantile(.05)
bank2 = bank[(bank['balance'] < upper_lim2) & (bank['balance'] > lower_lim2)]
upper_lim3 = bank['day'].quantile(.95)
lower_lim3 = bank['day'].quantile(.05)
bank3 = bank[(bank['day'] < upper_lim3) & (bank['day'] > lower_lim3)]
upper_lim4 = bank['duration'].quantile(.95)
lower_lim4 = bank['duration'].quantile(.05)
bank4 = bank[(bank['duration'] < upper_lim4) & (bank['duration'] > lower_lim4)]
upper_lim5 = bank['campaign'].quantile(.95)
lower_lim5 = bank['campaign'].quantile(.05)
bank5 = bank[(bank['campaign'] < upper_lim5) & (bank['campaign'] > lower_lim5)]
upper_lim6 = bank['pdays'].quantile(.95)
lower_lim6 = bank['pdays'].quantile(.05)
bank6 = bank[(bank['pdays'] < upper_lim6) & (bank['pdays'] > lower_lim6)]
upper_lim7 = bank['previous'].quantile(.95)
lower_lim7 = bank['previous'].quantile(.05)
bank7 = bank[(bank['previous'] < upper_lim7) & (bank['previous'] > lower_lim7)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
bank.loc[(bank['age'] > upper_lim1)] = upper_lim1
bank.loc[(bank['age'] < lower_lim1)] = lower_lim1
bank.loc[(bank['balance'] > upper_lim2)] = upper_lim2
bank.loc[(bank['balance'] < lower_lim2)] = lower_lim2
bank.loc[(bank['day'] > upper_lim3)] = upper_lim3
bank.loc[(bank['day'] < lower_lim3)] = lower_lim3
bank.loc[(bank['duration'] > upper_lim4)] = upper_lim4
bank.loc[(bank['duration'] < lower_lim4)] = lower_lim4
bank.loc[(bank['campaign'] > upper_lim5)] = upper_lim5
bank.loc[(bank['campaign'] < lower_lim5)] = lower_lim5
bank.loc[(bank['pdays'] > upper_lim6)] = upper_lim6
bank.loc[(bank['pdays'] < lower_lim6)] = lower_lim6
bank.loc[(bank['previous'] > upper_lim7)] = upper_lim7
bank.loc[(bank['previous'] < lower_lim7)] = lower_lim7
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
bank['bin1'] = pd.cut(bank['age'], bins=[18,43,70,95], labels=["Low", "Mid", "High"])
bank['bin2'] = pd.cut(bank['balance'], bins=[-8019,35000,70000,102127], labels=["Low", "Mid", "High"])
bank['bin3'] = pd.cut(bank['day'], bins=[1,11,21,31], labels=["Low", "Mid", "High"])
bank['bin4'] = pd.cut(bank['duration'],bins=[0,1650,3300,4918],labels=["Low", "Mid", "High"])
bank['bin5'] = pd.cut(bank['campaign'],bins=[1,21,42,63],labels=["Low", "Mid", "High"])
bank['bin6'] = pd.cut(bank['pdays'],bins=[-1,290,580,871],labels=["Low", "Mid", "High"])
bank['bin7'] = pd.cut(bank['previous'],bins=[0,20,100,275],labels=["Low", "Mid", "High"])
conditions1 = [
    bank['job'].str.contains('admin.'),
    bank['job'].str.contains('blue-collar'),
    bank['job'].str.contains('entrepreneur'),
    bank['job'].str.contains('housemaid'),
    bank['job'].str.contains('management'),
    bank['job'].str.contains('retired'),
    bank['job'].str.contains('self-employed'),
    bank['job'].str.contains('services'),
    bank['job'].str.contains('student'),
    bank['job'].str.contains('technician'),
    bank['job'].str.contains('unemployed'),
    bank['job'].str.contains('unknown')]
choices1=['1','2','3','4','5','6','7','8','9','10','11','12']
bank['choices1']=np.select(conditions1,choices1,default='Other')
conditions2 = [
    bank['marital'].str.contains('divorced'),
    bank['marital'].str.contains('married'),
    bank['marital'].str.contains('single')]
choices2= ['1','2','3']
bank['choices2']=np.select(conditions2,choices2,default='Other')
conditions3 = [
    bank['education'].str.contains('primary'),
    bank['education'].str.contains('secondary'),
    bank['education'].str.contains('tertiary'),
    bank['education'].str.contains('unknown')]
choices3= ['1','2','3','4']
bank['choices3']=np.select(conditions3,choices3,default='Other')
conditions4 = [
    bank['default'].str.contains('no'),
    bank['default'].str.contains('yes')]
choices4= ['1','2']
bank['choices4']=np.select(conditions4,choices4,default='Other')
conditions5 = [
    bank['housing'].str.contains('no'),
    bank['housing'].str.contains('yes')]
choices5= ['1','2']
bank['choices5']=np.select(conditions5,choices5,default='Other')
conditions6 = [
    bank['loan'].str.contains('no'),
    bank['loan'].str.contains('yes')]
choices6= ['1','2']
bank['choices6']=np.select(conditions6,choices6,default='Other')
conditions7 = [
    bank['contact'].str.contains('cellular'),
    bank['contact'].str.contains('telephone'),
    bank['contact'].str.contains('unknown')]
choices7= ['1','2','3']
bank['choices7']=np.select(conditions7,choices7,default='Other')
conditions8 = [
    bank['month'].str.contains('jan'),
    bank['month'].str.contains('feb'),
    bank['month'].str.contains('mar'),
    bank['month'].str.contains('apr'),
    bank['month'].str.contains('may'),
    bank['month'].str.contains('jun'),
    bank['month'].str.contains('jul'),
    bank['month'].str.contains('aug'),
    bank['month'].str.contains('sep'),
    bank['month'].str.contains('oct'),
    bank['month'].str.contains('nov'),
    bank['month'].str.contains('dec')]
choices8= ['1','2','3','4','5','6','7','8','9','10','11','12']
bank['choices8']=np.select(conditions8,choices8,default='Other')
conditions9 = [
    bank['poutcome'].str.contains('failure'),
    bank['poutcome'].str.contains('other'),
    bank['poutcome'].str.contains('success'),
    bank['poutcome'].str.contains('unknown')]
choices9= ['1','2','3','4']
bank['choices9']=np.select(conditions9,choices9,default='Other')
conditions10 = [
    bank['y'].str.contains('no'),
    bank['y'].str.contains('yes')]
choices10= ['1','2']
bank['choices10']=np.select(conditions10,choices10,default='Other')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
bank = pd.DataFrame({'age':bank.iloc[:,0]})
bank['log+1'] = (bank['age']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['age']-bank['age'].min()+1).transform(np.log)
bank = pd.DataFrame({'balance':bank.iloc[:,5]})
bank['log+1'] = (bank['balance']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['balance']-bank['balance'].min()+1).transform(np.log)
bank = pd.DataFrame({'day':bank.iloc[:,9]})
bank['log+1'] = (bank['day']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['day']-bank['day'].min()+1).transform(np.log)
bank = pd.DataFrame({'duration':bank.iloc[:,11]})
bank['log+1'] = (bank['duration']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['duration']-bank['duration'].min()+1).transform(np.log)
bank = pd.DataFrame({'campaign':bank.iloc[:,12]})
bank['log+1'] = (bank['campaign']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['campaign']-bank['campaign'].min()+1).transform(np.log)
bank = pd.DataFrame({'pdays':bank.iloc[:,13]})
bank['log+1'] = (bank['pdays']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['pdays']-bank['pdays'].min()+1).transform(np.log)
bank = pd.DataFrame({'previous':bank.iloc[:,14]})
bank['log+1'] = (bank['previous']+1).transform(np.log)
#Negative Values Handling
bank['log'] = (bank['previous']-bank['previous'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(bank['job'])
bank = bank.join(encoded_columns.add_suffix('_job')).drop('job', axis=1) 
encoded_columns_1 = pd.get_dummies(bank['marital'])
bank = bank.join(encoded_columns_1.add_suffix('_marital')).drop('marital', axis=1)    
encoded_columns_2 = pd.get_dummies(bank['education'])
bank = bank.join(encoded_columns_2.add_suffix('_education')).drop('education', axis=1)   
encoded_columns_3 = pd.get_dummies(bank['default'])
bank = bank.join(encoded_columns_3.add_suffix('_default')).drop('default', axis=1)   
encoded_columns_4 = pd.get_dummies(bank['housing'])
bank = bank.join(encoded_columns_4.add_suffix('_housing')).drop('housing', axis=1) 
encoded_columns_5 = pd.get_dummies(bank['loan'])
bank = bank.join(encoded_columns_5.add_suffix('_loan')).drop('loan', axis=1) 
encoded_columns_6 = pd.get_dummies(bank['contact'])
bank = bank.join(encoded_columns_6.add_suffix('_contact')).drop('contact', axis=1)   
encoded_columns_7 = pd.get_dummies(bank['month'])
bank = bank.join(encoded_columns_7.add_suffix('_month')).drop('month', axis=1) 
encoded_columns_8 = pd.get_dummies(bank['poutcome'])
bank = bank.join(encoded_columns_8.add_suffix('_poutcome')).drop('poutcome', axis=1)  
encoded_columns_9 = pd.get_dummies(bank['y'])
bank = bank.join(encoded_columns_9.add_suffix('_y')).drop('y', axis=1)                         
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = bank.groupby('y')
sums = grouped['y'].sum().add_suffix('_sum')
avgs = grouped['y'].mean().add_suffix('_avg')
####Categorical Column grouping
bank.groupby('y').agg(lambda x: x.value_counts().index[0])
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(bank.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(bank.iloc[:,0:9])
##### Feature Extraction
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X1 = bank.iloc[:,0]
##X1.astype('int64')
X2 = bank.iloc[:,5]
X3 = bank.iloc[:,9]
X4 = bank.iloc[:,11:14]
##X = pd.get_dummies(X, prefix_sep='_')
X=pd.concat([X1,X2,X3,X4],axis =1)
Y = bank['y']
##X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 45211)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=100).fit(X_Train,Y_Train)
    print(time.process_time() - start)   #### 4.28125
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))  ##### [[11538 414]
    ###                                                       [1066   546]]
    print(classification_report(Y_Test,predictionforest))  ####0.89
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, bank['y']], axis = 1)
PCA_df['y'] = LabelEncoder().fit_transform(PCA_df['y'])
PCA_df.head()  
#### Dont miss steps from here  
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
y = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for y, color in zip(y, colors):
    plt.scatter(PCA_df.loc[PCA_df['y'] == y, 'PC1'], 
                PCA_df.loc[PCA_df['y'] == y, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 20)
plt.legend(['Less', 'More'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)   ######  [9270631.20561717   66289.69159788   10026.36236128]
forest_test(X_pca, Y)
###Additionally, using our two-dimensional dataset, we can now also visualize the decision boundary used by our Random Forest in order to classify each of the different data points.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 45211)
trainedforest = RandomForestClassifier(n_estimators=100).fit(X_Reduced,Y_Reduced)
x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('Random Forest', fontsize = 20)
plt.show()
####Independent Component Analysis
from sklearn.decomposition import FastICA
ica = FastICA(n_components=3)
X_ica = ica.fit_transform(X)
forest_test(X_ica, Y)   ###accuracy 0.89
####Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
# run an LDA and use it to transform the features
labels=np.unique(Y)
X_lda = lda.fit(X,Y).transform(X)
print('Original number of features:', X.shape[1])   ### 6
print('Reduced number of features:', X_lda.shape[1])  ###1
forest_test(X_lda, Y)   #### accuracy 0.83
#####LDA can also be used as a classifier. Therefore, we can now test how an LDA Classifier can perform in this situation.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.30,random_state = 101) 
start = time.process_time()
lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)
print(time.process_time() - start)
predictionlda = lda.predict(X_Test_Reduced)
print(confusion_matrix(Y_Test_Reduced,predictionlda))
print(classification_report(Y_Test_Reduced,predictionlda))  ### accuracy 0.89
#####Locally Linear Embedding is a dimensionality reduction technique based on Manifold 
##Learning. A Manifold is an object of D dimensions which is embedded in an higher-dimensional space.
## Manifold Learning aims then to make this object representable in its original D dimensions instead of being represented in an unnecessary greater space.
from sklearn.manifold import LocallyLinearEmbedding
embedding = LocallyLinearEmbedding(n_components=3)
X_lle = embedding.fit_transform(X) 
forest_test(X_lle, Y)   ### accuracy 0.88
####t-SNE is non-linear dimensionality reduction technique which is typically used to visualize high dimensional datasets.
#####t-SNE works by minimizing the divergence between a distribution constituted by the pairwise probability similarities
### of the input features in the original high dimensional space and its equivalent in the reduced low dimensional space. 
##t-SNE makes then use of the Kullback-Leiber (KL) divergence in order to measure the dissimilarity of the two different 
####distributions. The KL divergence is then minimized using gradient descent.
from sklearn.manifold import TSNE
start = time.process_time()
tsne = TSNE(n_components=3, verbose=1, perplexity=400, n_iter=3000)
X_tsne = tsne.fit_transform(X)
print(time.process_time() - start)
forest_test(X_tsne, Y)
#####Autoencoders are a family of Machine Learning algorithms which can be used as a 
###dimensionality reduction technique. The main difference between Autoencoders and 
##other dimensionality reduction techniques is that Autoencoders use non-linear 
###transformations to project data from a high dimension to a lower one.
from keras.layers import Input, Dense
from keras.models import Model
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(3, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='softmax')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=600)
autoencoder.fit(X1, Y1,epochs=600,batch_size=600,shuffle=True,verbose = 500,validation_data=(X2, Y2))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)


