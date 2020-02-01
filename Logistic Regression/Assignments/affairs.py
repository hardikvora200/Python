import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# reading a csv file using pandas library
affair=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Logistic Regression\\Assignments\\affairs.csv")
affair.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
affair.columns
affair.drop(["a"],axis=1,inplace=True)
affair.columns
# To get the count of null values in the data 
affair.isnull().sum()
affair.shape # 601 9 => Before dropping null values
# To drop null values ( dropping rows)
affair.dropna().shape # 601 9 => After dropping null values
#####Exploratory Data Analysis#########################################################
affair.mean() ## affairs - 1.455907, age - 32.487521, yearsmarried  - 8.177696,religiousness - 3.116473
  ################education - 16.166389,occupation -4.194676,rating - 3.931780
affair.median() #### affairs - 0.0, age - 32, yearsmarried -7.0,religiousness - 3.0
  ################  education - 16.0,occupation - 5.0 ,rating - 4.0
affair.mode() 
####Measures of Dispersion
affair.var() 
affair.std() ## affairs - 3.298758, age - 9.288762, yearsmarried  - 5.571303, religiousness - 1.167509
  ################ education - 2.402555, occupation - 1.819443,rating - 1.103179
#### Calculate the range value
range1 = max(affair['affairs'])-min(affair['affairs'])  ### 12
range2 = max(affair['age'])-min(affair['age']) ### 39.5
range3 = max(affair['yearsmarried'])-min(affair['yearsmarried']) ### 14.875
range4 = max(affair['religiousness'])-min(affair['religiousness']) ### 4
range5 = max(affair['education'])-min(affair['education'])  ##  11
range6 = max(affair['occupation'])-min(affair['occupation']) ### 6
range7 = max(affair['rating'])-min(affair['rating']) #### 4
### Calculate skewness and Kurtosis
affair.skew() ## affairs - 2.346998, age - 0.889221, yearsmarried  -  0.078189, religiousness -  -0.089023
  ################ education - -0.250273, occupation - -0.740587,rating - -0.836214
affair.kurt() ## affairs -  4.256882, age - 0.231969, yearsmarried  -  -1.570553, religiousness - -1.008357
  ################ education - -0.301831, occupation -  -0.775692,rating -  -0.203801
####Graphidelivery_time Representation 
plt.hist(affair["affairs"])
plt.hist(affair["age"])
plt.hist(affair["yearsmarried"])
plt.hist(affair["religiousness"])
plt.hist(affair["education"])
plt.hist(affair["occupation"])
plt.hist(affair["rating"])
plt.boxplot(affair["affairs"],0,"rs",0)
plt.boxplot(affair["age"],0,"rs",0)
plt.boxplot(affair["yearsmarried"],0,"rs",0)
plt.boxplot(affair["religiousness"],0,"rs",0)
plt.boxplot(affair["education"],0,"rs",0)
plt.boxplot(affair["occupation"],0,"rs",0)
plt.boxplot(affair["rating"],0,"rs",0)
plt.plot(affair["age"],affair["affairs"],"bo");plt.xlabel("age");plt.ylabel("affairs")
plt.plot(affair["yearsmarried"],affair["affairs"],"bo");plt.xlabel("yearsmarried");plt.ylabel("affairs")
plt.plot(affair["religiousness"],affair["affairs"],"bo");plt.xlabel("religiousness");plt.ylabel("affairs")
plt.plot(affair["education"],affair["affairs"],"bo");plt.xlabel("education");plt.ylabel("affairs")
plt.plot(affair["occupation"],affair["affairs"],"bo");plt.xlabel("occupation");plt.ylabel("affairs")
plt.plot(affair["rating"],affair["affairs"],"bo");plt.xlabel("rating");plt.ylabel("affairs")
# table 
pd.crosstab(affair["age"],affair["affairs"])
pd.crosstab(affair["yearsmarried"],affair["affairs"])
pd.crosstab(affair["religiousness"],affair["affairs"])
pd.crosstab(affair["education"],affair["affairs"])
pd.crosstab(affair["occupation"],affair["affairs"])
pd.crosstab(affair["rating"],affair["affairs"])
## Barplot
pd.crosstab(affair["age"],affair["affairs"]).plot(kind = "bar", width = 1.85)
pd.crosstab(affair["yearsmarried"],affair["affairs"]).plot(kind = "bar", width = 1.85)
pd.crosstab(affair["religiousness"],affair["affairs"]).plot(kind = "bar", width = 1.85)
pd.crosstab(affair["education"],affair["affairs"]).plot(kind = "bar", width = 1.85)
pd.crosstab(affair["occupation"],affair["affairs"]).plot(kind = "bar", width = 1.85)
pd.crosstab(affair["rating"],affair["affairs"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="age",data=affair,palette="hls")
sns.countplot(x="yearsmarried",data=affair,palette="hls")
sns.countplot(x="religiousness",data=affair,palette="hls")
sns.countplot(x="education",data=affair,palette="hls")
sns.countplot(x="occupation",data=affair,palette="hls")
sns.countplot(x="rating",data=affair,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="age",y="affairs",data=affair,palette="hls")
sns.boxplot(x="yearsmarried",y="affairs",data=affair,palette="hls")
sns.boxplot(x="religiousness",y="affairs",data=affair,palette="hls")
sns.boxplot(x="education",y="affairs",data=affair,palette="hls")
sns.boxplot(x="occupation",y="affairs",data=affair,palette="hls")
sns.boxplot(x="rating",y="affairs",data=affair,palette="hls")
sns.pairplot(affair.iloc[:,0:5]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(affair,hue="affair",size=2)
affair["age"].value_counts()
affair["yearsmarried"].value_counts()
affair["religiousness"].value_counts()
affair["education"].value_counts()
affair["occupation"].value_counts()
affair["rating"].value_counts()
affair["affairs"].value_counts()
affair["age"].value_counts().plot(kind = "pie")
affair["yearsmarried"].value_counts().plot(kind = "pie")
affair["religiousness"].value_counts().plot(kind = "pie")
affair["education"].value_counts().plot(kind = "pie")
affair["occupation"].value_counts().plot(kind = "pie")
affair["rating"].value_counts().plot(kind = "pie")
affair["affairs"].value_counts().plot(kind = "pie")
sns.pairplot(affair,hue="affairs",size=4,diag_kind = "kde")
sns.FacetGrid(affair,hue="affairs").map(plt.scatter,"age","affairs").add_legend()
sns.FacetGrid(affair,hue="affairs").map(plt.scatter,"yearsmarried","affairs").add_legend()
sns.FacetGrid(affair,hue="affairs").map(plt.scatter,"religiousness","affairs").add_legend()
sns.FacetGrid(affair,hue="affairs").map(plt.scatter,"education","affairs").add_legend()
sns.FacetGrid(affair,hue="affairs").map(plt.scatter,"occupation","affairs").add_legend()
sns.FacetGrid(affair,hue="affairs").map(plt.scatter,"rating","affairs").add_legend()
sns.FacetGrid(affair,hue="affairs").map(sns.kdeplot,"age").add_legend()
sns.FacetGrid(affair,hue="affairs").map(sns.kdeplot,"yearsmarried").add_legend()
sns.FacetGrid(affair,hue="affairs").map(sns.kdeplot,"religiousness").add_legend()
sns.FacetGrid(affair,hue="affairs").map(sns.kdeplot,"education").add_legend()
sns.FacetGrid(affair,hue="affairs").map(sns.kdeplot,"occupation").add_legend()
sns.FacetGrid(affair,hue="affairs").map(sns.kdeplot,"rating").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(affair['age'], dist="norm",plot=pylab)
stats.probplot(np.log(affair['age']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(affair['age']),dist="norm",plot=pylab)
stats.probplot((affair['age'] * affair['age']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['age']),dist="norm",plot=pylab)
reci_1=1/affair['age']
stats.probplot(reci_1,dist="norm",plot=pylab)
stats.probplot(((affair['age'] * affair['age'])+affair['age']),dist="norm",plot=pylab)
stats.probplot(affair['yearsmarried'],dist="norm",plot=pylab)
stats.probplot(np.log(affair['yearsmarried']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(affair['yearsmarried']),dist="norm",plot=pylab)
stats.probplot((affair['yearsmarried'] * affair['yearsmarried']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['yearsmarried']),dist="norm",plot=pylab)
reci_2=1/affair['yearsmarried']
stats.probplot(reci_2,dist="norm",plot=pylab)
stats.probplot(((affair['yearsmarried'] * affair['yearsmarried'])+affair['yearsmarried']),dist="norm",plot=pylab)
stats.probplot(affair['religiousness'],dist="norm",plot=pylab)
stats.probplot(np.log(affair['religiousness']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(affair['religiousness']),dist="norm",plot=pylab)
stats.probplot((affair['religiousness'] * affair['religiousness']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['religiousness']),dist="norm",plot=pylab)
reci_3=1/affair['religiousness']
stats.probplot(reci_3,dist="norm",plot=pylab)
stats.probplot(((affair['religiousness'] * affair['religiousness'])+affair['religiousness']),dist="norm",plot=pylab)
stats.probplot(affair['education'],dist="norm",plot=pylab)
stats.probplot(np.log(affair['education']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(affair['education']),dist="norm",plot=pylab)
stats.probplot((affair['education'] * affair['education']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['education']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['education'])*np.exp(affair['education']),dist="norm",plot=pylab)
reci_4=1/affair['education']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((affair['education'] * affair['education'])+affair['education']),dist="norm",plot=pylab)
stats.probplot(affair['occupation'],dist="norm",plot=pylab)
stats.probplot(np.log(affair['occupation']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(affair['occupation']),dist="norm",plot=pylab)
stats.probplot((affair['occupation'] * affair['occupation']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['occupation']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['occupation'])*np.exp(affair['occupation']),dist="norm",plot=pylab)
reci_5=1/affair['occupation']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((affair['occupation'] * affair['occupation'])+affair['occupation']),dist="norm",plot=pylab)
stats.probplot(affair['rating'],dist="norm",plot=pylab)
stats.probplot(np.log(affair['rating']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(affair['rating']),dist="norm",plot=pylab)
stats.probplot((affair['rating'] * affair['rating']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['rating']),dist="norm",plot=pylab)
stats.probplot(np.exp(affair['rating'])*np.exp(affair['rating']),dist="norm",plot=pylab)
reci_6=1/affair['rating']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((affair['rating'] * affair['rating'])+affair['rating']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### age
stats.norm.ppf(0.975,32.487521,9.288762)# similar to qnorm in R ---- 50.69315998096424
# cdf => cumulative distributive function 
stats.norm.cdf(affair["age"],32.487521,9.288762) # similar to pnorm in R 
#### yearsmarried
stats.norm.ppf(0.975,8.177696,5.571303)# similar to qnorm in R ---- 19.097249226959956
# cdf => cumulative distributive function 
stats.norm.cdf(affair["yearsmarried"],8.177696,5.571303) # similar to pnorm in R 
#### religiousness
stats.norm.ppf(0.975,3.116473,1.167509)# similar to qnorm in R ---- 5.404748591626374.
# cdf => cumulative distributive function 
stats.norm.cdf(affair["religiousness"],3.116473,1.167509) # similar to pnorm in R 
#### education
stats.norm.ppf(0.975, 16.166389, 2.402555)# similar to qnorm in R ---- 20.87531027087663
# cdf => cumulative distributive function 
stats.norm.cdf(affair["education"], 16.166389, 2.402555) # similar to pnorm in R 
#### occupation
stats.norm.ppf(0.975, 4.194676, 1.819443)# similar to qnorm in R ---- 7.76071875192351
# cdf => cumulative distributive function 
stats.norm.cdf(affair["occupation"], 4.194676, 1.819443) # similar to pnorm in R 
#### rating
stats.norm.ppf(0.975, 3.931780, 1.103179)# similar to qnorm in R ---- 6.093971108500912
# cdf => cumulative distributive function 
stats.norm.cdf(affair["rating"], 3.931780, 1.103179) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
affair.corr(method = "pearson")
affair.corr(method = "kendall")
affair["affairs"].corr(affair["age"]) # # correlation value between X and Y -- 0.09
affair["affairs"].corr(affair["yearsmarried"])  ### 0.187
affair["affairs"].corr(affair["religiousness"]) ### -0.144
affair["affairs"].corr(affair["education"]) ### -0.002
affair["affairs"].corr(affair["occupation"]) #### 0.049
affair["affairs"].corr(affair["rating"]) #### -0.279
np.corrcoef(affair["age"],affair["affairs"])
np.corrcoef(affair["yearsmarried"],affair["affairs"])
np.corrcoef(affair["religiousness"],affair["affairs"])
np.corrcoef(affair["education"],affair["affairs"])
np.corrcoef(affair["occupation"],affair["affairs"])
np.corrcoef(affair["rating"],affair["affairs"])
###### Lets do normalization
a_array = np.array(affair['age'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(affair['yearsmarried'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(affair['religiousness'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(affair['education'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(affair['occupation'])
normalized_E = preprocessing.normalize([e_array])
f_array = np.array(affair['rating'])
normalized_F = preprocessing.normalize([f_array])
g_array = np.array(affair['affairs'])
normalized_G = preprocessing.normalize([g_array])
# to get top 6 rows
affair.head(40) # to get top n rows use cars.head(10)
affair.tail(10)
# Correlation matrix 
affair.corr()
# Scatter plot between the variables along with histograms
sns.pairplot(affair)
pd.tools.plotting.scatter_matrix(affair) ##-> also used for plotting all in one graph
# creating dummy columns for the categorical columns 
affair.columns
affair=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Logistic Regression\\Assignments\\affairs.csv")
affair_dummies=pd.get_dummies(affair[["gender","children"]])
affair = pd.concat([affair,affair_dummies],axis=1)
affair.affairs.value_counts()
affair.drop(["gender"],inplace=True,axis = 1)
affair.drop(["children"],inplace=True,axis = 1)
affair.shape
X = affair.iloc[:,[1,2,3,4,5,6]]
Y = affair.iloc[:,0]
classifier = LogisticRegression()
classifier.fit(X,Y)
classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 
y_pred = classifier.predict(X)
affair["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([affair,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
type(y_pred)
accuracy = sum(Y==y_pred)/affair.shape[0] ### 0.56 or 56%
pd.crosstab(y_pred,Y)
####Model Building
affair['affairs']=np.divide(affair['affairs']-affair['affairs'].min(),affair['affairs'].max()-affair['affairs'].min())
##affair[1]=np.divide(affair[1]-affair[1].min(),affair[1].max()-affair[1].min())
affair['age']=np.divide(affair['age']-affair['age'].min(),affair['age'].max()-affair['age'].min())
affair['yearsmarried']=np.divide(affair['yearsmarried']-affair['yearsmarried'].min(),affair['yearsmarried'].max()-affair['yearsmarried'].min())
affair['education']=np.divide(affair['education']-affair['education'].min(),affair['education'].max()-affair['education'].min())
affair['occupation']=np.divide(affair['occupation']-affair['occupation'].min(),affair['occupation'].max()-affair['occupation'].min())
affair['rating']=np.divide(affair['rating']-affair['rating'].min(),affair['rating'].max()-affair['rating'].min())
import statsmodels.formula.api as sm
logit_model = sm.logit('affairs~(age+yearsmarried+religiousness+education+occupation+rating)',data = affair).fit()
st.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
#summary
logit_model.summary()
y_pred1 = logit_model.predict(affair)
affair["pred_prob"] = y_pred1
# filling all the cells with zeroes
affair["Att_val"] = np.zeros(601)
affair["affairs"]
# Taking threshold value as 0.5 and above the prob value will be treated as correct value 
affair.loc[y_pred1>=0.5,"Att_val"] = 1
affair.Att_val
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
affair["affairs"]  ### As affairs is in continuous form we need to convert it back to original form
classification_report(affair.Att_val,affair['affairs'].round())
# confusion matrix 
results = confusion_matrix(affair['affairs'].round(),affair.Att_val)
# Model Accuracy 
Accuracy = accuracy_score(affair['affairs'].round(),affair.Att_val)
### ROC Curve
fpr, tpr, threshold = metrics.roc_curve(affair.affairs.round(), y_pred1)
# the above function is applicable for binary classification class 
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve ---    0.76
### Dividing data into train and test data sets
affair.drop("Att_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split
train,test = train_test_split(affair,test_size=0.35)
# checking na values 
train.isnull().sum();test.isnull().sum()
# Building a model on train data set 
train_model = sm.logit('affairs~(age+yearsmarried+religiousness+education+occupation+rating)',data = train).fit()
#summary
train_model.summary()
train_pred = train_model.predict(train.iloc[:,1:])
# Creating new column for storing predicted class of affairs
# filling all the cells with zeroes
train["train_pred"] = np.zeros(390)
# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1
# confusion matrix 
confusion_matrix1 = pd.crosstab(train['affairs'],train.train_pred)
confusion_matrix1
##accuracy_train ##94.35%
# Prediction on Test data set
test_pred = train_model.predict(test)
# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test["test_pred"] = np.zeros(211)
# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1
# confusion matrix 
confusion_matrix2 = pd.crosstab(test['affairs'],test.test_pred)
confusion_matrix2
##accuracy_test #92.41%
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
affair = affair[affair.columns[affair.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
affair = affair.loc[affair.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
affair = affair.fillna(affair.median())
affair['gender'].fillna(affair['gender'].value_counts().idxmax(), inplace=True)
affair['children'].fillna(affair['children'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##affair['column_name'].fillna(affair['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = affair['affairs'].mean () + affair['affairs'].std () * factor   ### 11.35
lower_lim1= affair['affairs'].mean () - affair['affairs'].std () * factor   ### -8.44
affair1 = affair[(affair['affairs'] < upper_lim1) & (affair['affairs'] > lower_lim1)]
upper_lim2 = affair['age'].mean () + affair['age'].std () * factor  #### 60.35
lower_lim2 = affair['age'].mean () - affair['age'].std () * factor  ## 4.62
affair2 = affair[(affair['age'] < upper_lim2) & (affair['age'] > lower_lim2)]
upper_lim3 = affair['yearsmarried'].mean () + affair['yearsmarried'].std () * factor  ### 24.89
lower_lim3 = affair['yearsmarried'].mean () - affair['yearsmarried'].std () * factor  ### -8.53
affair3 = affair[(affair['yearsmarried'] < upper_lim3) & (affair['yearsmarried'] > lower_lim3)]
upper_lim4 = affair['religiousness'].mean () + affair['religiousness'].std () * factor ###6.62
lower_lim4 = affair['religiousness'].mean () - affair['religiousness'].std () * factor  #####  -0.39
affair4 = affair[(affair['religiousness'] < upper_lim4) & (affair['religiousness'] > lower_lim4)]
upper_lim5 = affair['education'].mean () + affair['education'].std () * factor   #### 23.37
lower_lim5 = affair['education'].mean () - affair['education'].std () * factor  ### 8.96
affair5 = affair[(affair['education'] < upper_lim5) & (affair['education'] > lower_lim5)]
upper_lim6 = affair['occupation'].mean () + affair['occupation'].std () * factor
lower_lim6 = affair['occupation'].mean () - affair['occupation'].std () * factor
affair6 = affair[(affair['occupation'] < upper_lim6) & (affair['occupation'] > lower_lim6)]
upper_lim7 = affair['rating'].mean () + affair['rating'].std () * factor
lower_lim7 = affair['rating'].mean () - affair['rating'].std () * factor
affair = affair[(affair['rating'] < upper_lim7) & (affair['rating'] > lower_lim7)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim8 = affair['affairs'].quantile(.95)
lower_lim8 = affair['affairs'].quantile(.05)
affair8 = affair[(affair['affairs'] < upper_lim8) & (affair['affairs'] > lower_lim8)]
upper_lim9 = affair['age'].quantile(.95)
lower_lim9 = affair['age'].quantile(.05)
affair9 = affair[(affair['age'] < upper_lim9) & (affair['age'] > lower_lim9)]
upper_lim10 = affair['yearsmarried'].quantile(.95)
lower_lim10 = affair['yearsmarried'].quantile(.05)
affair10 = affair[(affair['yearsmarried'] < upper_lim10) & (affair['yearsmarried'] > lower_lim10)]
upper_lim11 = affair['religiousness'].quantile(.95)
lower_lim11 = affair['religiousness'].quantile(.05)
affair11 = affair[(affair['religiousness'] < upper_lim11) & (affair['religiousness'] > lower_lim11)]
upper_lim12 = affair['education'].quantile(.95)
lower_lim12 = affair['education'].quantile(.05)
affair12 = affair[(affair['education'] < upper_lim12) & (affair['education'] > lower_lim12)]
upper_lim13 = affair['occupation'].quantile(.95)
lower_lim13 = affair['occupation'].quantile(.05)
affair13 = affair[(affair['occupation'] < upper_lim13) & (affair['occupation'] > lower_lim13)]
upper_lim14 = affair['rating'].quantile(.95)
lower_lim14 = affair['rating'].quantile(.05)
affair14 = affair[(affair['rating'] < upper_lim14) & (affair['rating'] > lower_lim14)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
affair.loc[(affair['affairs'] > upper_lim8)] = upper_lim8
affair.loc[(affair['affairs'] < lower_lim8)] = lower_lim8
affair.loc[(affair['age'] > upper_lim9)] = upper_lim9
affair.loc[(affair['age'] < lower_lim9)] = lower_lim9
affair.loc[(affair['yearsmarried'] > upper_lim10)] = upper_lim10
affair.loc[(affair['yearsmarried'] < lower_lim10)] = lower_lim10
affair.loc[(affair['religiousness'] > upper_lim11)] = upper_lim11
affair.loc[(affair['religiousness'] < lower_lim11)] = lower_lim11
affair.loc[(affair['education'] > upper_lim12)] = upper_lim12
affair.loc[(affair['education'] < lower_lim12)] = lower_lim12
affair.loc[(affair['occupation'] > upper_lim13)] = upper_lim13
affair.loc[(affair['occupation'] < lower_lim13)] = lower_lim13
affair.loc[(affair['rating'] > upper_lim14)] = upper_lim14
affair.loc[(affair['rating'] < lower_lim14)] = lower_lim14
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
affair['bin1'] = pd.cut(affair['affairs'], bins=[0,3,6,12], labels=["Low", "Mid", "High"])
affair['bin2'] = pd.cut(affair['age'], bins=[17.5,30,43,57], labels=["Low", "Mid", "High"])
affair['bin3'] = pd.cut(affair['yearsmarried'], bins=[0.125,4,8,15], labels=["Low", "Mid", "High"])
affair['bin4'] = pd.cut(affair['religiousness'],bins=[1,2,4,5],labels=["Low", "Mid", "High"])
affair['bin5'] = pd.cut(affair['education'],bins=[9,13,17,20],labels=["Low", "Mid", "High"])
affair['bin6'] = pd.cut(affair['occupation'],bins=[1,3,5,7],labels=["Low", "Mid", "High"])
affair['bin7'] = pd.cut(affair['rating'],bins=[1,2,4,5],labels=["Low", "Mid", "High"])
conditions = [
    affair['gender'].str.contains('male'),
    affair['gender'].str.contains('female')]
choices=['1','2']
affair['choices']=np.select(conditions,choices,default='Other')
conditions1 = [
    affair['children'].str.contains('no'),
    affair['children'].str.contains('yes')]
choices1= ['1','2']
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
affair = pd.DataFrame({'affairs':affair.iloc[:,1]})
affair['log+1'] = (affair['affairs']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['affairs']-affair['affairs'].min()+1).transform(np.log)
affair = pd.DataFrame({'age':affair.iloc[:,3]})
affair['log+1'] = (affair['age']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['age']-affair['age'].min()+1).transform(np.log)
affair = pd.DataFrame({'yearsmarried':affair.iloc[:,4]})
affair['log+1'] = (affair['yearsmarried']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['yearsmarried']-affair['yearsmarried'].min()+1).transform(np.log)
affair = pd.DataFrame({'religiousness':affair.iloc[:,6]})
affair['log+1'] = (affair['religiousness']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['religiousness']-affair['religiousness'].min()+1).transform(np.log)
affair = pd.DataFrame({'education':affair.iloc[:,7]})
affair['log+1'] = (affair['education']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['education']-affair['education'].min()+1).transform(np.log)
affair = pd.DataFrame({'occupation':affair.iloc[:,8]})
affair['log+1'] = (affair['occupation']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['occupation']-affair['occupation'].min()+1).transform(np.log)
affair = pd.DataFrame({'rating':affair.iloc[:,9]})
affair['log+1'] = (affair['rating']+1).transform(np.log)
#Negative Values Handling
affair['log'] = (affair['rating']-affair['rating'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(affair['gender'])
affair = affair.join(encoded_columns.add_suffix('_gender')).drop('gender', axis=1) 
encoded_columns_1 = pd.get_dummies(affair['children'])
affair = affair.join(encoded_columns_1.add_suffix('_children')).drop('children', axis=1)                                    
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = affair.groupby('affairs')
sums = grouped['affairs'].sum().add_suffix('_sum')
avgs = grouped['affairs'].mean().add_suffix('_avg')
####Categorical Column grouping
affair.groupby('affairs').agg(lambda x: x.value_counts().index[0])
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(affair.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(affair.iloc[:,0:9])
##### Feature Extraction
X = affair.drop('affairs', axis=1)
Y=affair['affairs']
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
#X1 = affair.iloc[:,1]
##X2 = affair.iloc[:,3:4]
##X3 = affair.iloc[:,6:9]
X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 600)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=600).fit(X_Train,Y_Train)
    print(time.process_time() - start) 
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, affair['affairs']], axis = 1)
PCA_df['affairs'] = LabelEncoder().fit_transform(PCA_df['affairs'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
affairs = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for affairs, color in zip(affairs, colors):
    plt.scatter(PCA_df.loc[PCA_df['affairs'] == affairs, 'PC1'], 
                PCA_df.loc[PCA_df['affairs'] == affairs, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 20)
plt.legend(['Less', 'More'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
###Additionally, using our two-dimensional dataset, we can now also visualize the decision boundary used by our Random Forest in order to classify each of the different data points.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=600).fit(X_Reduced,Y_Reduced)
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
forest_test(X_ica, Y)
####Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
# run an LDA and use it to transform the features
labels=np.unique(Y)
X_lda = lda.fit(X,Y).transform(X)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])
forest_test(X_lda, Y)
#####LDA can also be used as a classifier. Therefore, we can now test how an LDA Classifier can perform in this situation.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.30,random_state = 101) 
start = time.process_time()
lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)
print(time.process_time() - start)
predictionlda = lda.predict(X_Test_Reduced)
print(confusion_matrix(Y_Test_Reduced,predictionlda))
print(classification_report(Y_Test_Reduced,predictionlda))
#####Locally Linear Embedding is a dimensionality reduction technique based on Manifold 
##Learning. A Manifold is an object of D dimensions which is embedded in an higher-dimensional space.
## Manifold Learning aims then to make this object representable in its original D dimensions instead of being represented in an unnecessary greater space.
from sklearn.manifold import LocallyLinearEmbedding
embedding = LocallyLinearEmbedding(n_components=3)
X_lle = embedding.fit_transform(X) 
forest_test(X_lle, Y)
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

