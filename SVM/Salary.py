import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# reading a csv file using pandas library
salary_train = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\SalaryData_Train.csv")
##salary_train.drop(["Unnamed: 0"],axis=1)
#####Exploratory Data Analysis#########################################################
salary_train.mean() 
salary_train.median()
salary_train.std() 
#### Calculate the range value
range1 = max(salary_train['age'])-min(salary_train['age'])### 73
range2 = max(salary_train['educationno'])-min(salary_train['educationno']) ### 15
range3 = max(salary_train['capitalgain'])-min(salary_train['capitalgain']) ### 99999
range4 = max(salary_train['capitalloss'])-min(salary_train['capitalloss']) ### 4356
range5 = max(salary_train['hoursperweek'])-min(salary_train['hoursperweek']) ## 98
### Calculate skewness and Kurtosis
salary_train.skew() ##age-0.530180, educationno- -0.305378,capitalgain-11.902483,capitalloss-4.428238,hoursperweek-0.330856
salary_train.kurt() ###age- -0.14476,educationno- 0.643605, capitalgain - 153.661174,capitalloss-19.529284,hoursperweek-3.167683
####Graphidelivery_time Representation 
plt.hist(salary_train["age"])
plt.hist(salary_train["educationno"])
plt.hist(salary_train["capitalgain"])
plt.hist(salary_train["capitalloss"])
plt.hist(salary_train["hoursperweek"])
plt.boxplot(salary_train["age"],0,"rs",0)
plt.boxplot(salary_train["educationno"],0,"rs",0)
plt.boxplot(salary_train["capitalgain"],0,"rs",0)
plt.boxplot(salary_train["capitalloss"],0,"rs",0)
plt.boxplot(salary_train["hoursperweek"],0,"rs",0)
plt.plot(salary_train["age"],salary_train["Salary"],"bo");plt.xlabel("age");plt.ylabel("Salary")
plt.plot(salary_train["educationno"],salary_train["Salary"],"bo");plt.xlabel("educationno");plt.ylabel("Salary")
plt.plot(salary_train["capitalgain"],salary_train["Salary"],"bo");plt.xlabel("capitalgain");plt.ylabel("Salary")
plt.plot(salary_train["capitalloss"],salary_train["Salary"],"bo");plt.xlabel("capitalloss");plt.ylabel("Salary")
plt.plot(salary_train["hoursperweek"],salary_train["Salary"],"bo");plt.xlabel("hoursperweek");plt.ylabel("Salary")
plt.plot(salary_train["educationno"],salary_train["age"],"bo");plt.xlabel("educationno");plt.ylabel("age")
plt.plot(salary_train["capitalgain"],salary_train["age"],"bo");plt.xlabel("capitalgain");plt.ylabel("age")
plt.plot(salary_train["capitalloss"],salary_train["age"],"bo");plt.xlabel("capitalloss");plt.ylabel("age")
plt.plot(salary_train["hoursperweek"],salary_train["age"],"bo");plt.xlabel("hoursperweek");plt.ylabel("age")
plt.plot(salary_train["capitalgain"],salary_train["educationno"],"bo");plt.xlabel("capitalgain");plt.ylabel("educationno")
plt.plot(salary_train["capitalloss"],salary_train["educationno"],"bo");plt.xlabel("capitalloss");plt.ylabel("educationno")
plt.plot(salary_train["hoursperweek"],salary_train["educationno"],"bo");plt.xlabel("hoursperweek");plt.ylabel("educationno")
plt.plot(salary_train["capitalloss"],salary_train["capitalgain"],"bo");plt.xlabel("capitalloss");plt.ylabel("capitalgain")
plt.plot(salary_train["hoursperweek"],salary_train["capitalgain"],"bo");plt.xlabel("hoursperweek");plt.ylabel("capitalgain")
plt.plot(salary_train["hoursperweek"],salary_train["capitalloss"],"bo");plt.xlabel("hoursperweek");plt.ylabel("capitalloss")
## Barplot
pd.crosstab(salary_train["age"],salary_train["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["educationno"],salary_train["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalgain"],salary_train["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalloss"],salary_train["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["hoursperweek"],salary_train["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["educationno"],salary_train["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalgain"],salary_train["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalloss"],salary_train["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["hoursperweek"],salary_train["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalgain"],salary_train["educationno"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalloss"],salary_train["educationno"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["hoursperweek"],salary_train["educationno"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["capitalloss"],salary_train["capitalgain"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["hoursperweek"],salary_train["capitalgain"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_train["hoursperweek"],salary_train["capitalloss"]).plot(kind = "bar",width=1.85)
import seaborn as sns 
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="age",y="Salary",data=salary_train)
sns.boxplot(x="educationno",y="Salary",data=salary_train)
sns.boxplot(x="capitalgain",y="Salary",data=salary_train)
sns.boxplot(x="capitalloss",y="Salary",data=salary_train)
sns.boxplot(x="hoursperweek",y="Salary",data=salary_train)
sns.boxplot(x="educationno",y="age",data=salary_train)
sns.boxplot(x="capitalgain",y="age",data=salary_train)
sns.boxplot(x="capitalloss",y="age",data=salary_train)
sns.boxplot(x="hoursperweek",y="age",data=salary_train)
sns.boxplot(x="capitalgain",y="educationno",data=salary_train)
sns.boxplot(x="capitalloss",y="educationno",data=salary_train)
sns.boxplot(x="hoursperweek",y="educationno",data=salary_train)
sns.boxplot(x="capitalloss",y="capitalgain",data=salary_train)
sns.boxplot(x="hoursperweek",y="capitalgain",data=salary_train)
sns.boxplot(x="hoursperweek",y="capitalloss",data=salary_train)
sns.pairplot(salary_train.iloc[:,0:13]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(salary_train,hue="Salary",size=5)
salary_train["age"].value_counts()
salary_train["educationno"].value_counts()
salary_train["capitalgain"].value_counts()
salary_train["capitalloss"].value_counts()
salary_train["hoursperweek"].value_counts()
salary_train["age"].value_counts().plot(kind="pie")
salary_train["educationno"].value_counts().plot(kind="pie")
salary_train["capitalgain"].value_counts().plot(kind="pie")
salary_train["capitalloss"].value_counts().plot(kind="pie")
salary_train["hoursperweek"].value_counts().plot(kind="pie")
sns.pairplot(salary_train,hue="Salary",size=4,diag_kind = "kde")
sns.FacetGrid(salary_train,hue="Salary").map(plt.scatter,"age","Salary").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(plt.scatter,"educationno","Salary").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(plt.scatter,"capitalgain","Salary").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(plt.scatter,"capitalloss","Salary").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(plt.scatter,"hoursperweek","Salary").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(sns.kdeplot,"age").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(sns.kdeplot,"educationno").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(sns.kdeplot,"capitalgain").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(sns.kdeplot,"capitalloss").add_legend()
sns.FacetGrid(salary_train,hue="Salary").map(sns.kdeplot,"hoursperweek").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(salary_train['age'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_train['age']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_train['age']),dist="norm",plot=pylab)
stats.probplot((salary_train['age'] * salary_train['age']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['age']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['age'])*np.exp(salary_train['age']),dist="norm",plot=pylab)
reci_1=1/salary_train['age']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((salary_train['age'] * salary_train['age'])+salary_train['age']),dist="norm",plot=pylab)
stats.probplot(salary_train['educationno'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_train['educationno']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_train['educationno']),dist="norm",plot=pylab)
stats.probplot((salary_train['educationno'] * salary_train['educationno']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['educationno']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['educationno'])*np.exp(salary_train['educationno']),dist="norm",plot=pylab)
reci_2=1/salary_train['educationno']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((salary_train['educationno'] * salary_train['educationno'])+salary_train['educationno']),dist="norm",plot=pylab)
stats.probplot(salary_train['capitalgain'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_train['capitalgain']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_train['capitalgain']),dist="norm",plot=pylab)
stats.probplot((salary_train['capitalgain'] * salary_train['capitalgain']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['capitalgain']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['capitalgain'])*np.exp(salary_train['capitalgain']),dist="norm",plot=pylab)
reci_3=1/salary_train['capitalgain']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((salary_train['capitalgain'] * salary_train['capitalgain'])+salary_train['capitalgain']),dist="norm",plot=pylab)
stats.probplot(salary_train['capitalloss'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_train['capitalloss']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_train['capitalloss']),dist="norm",plot=pylab)
stats.probplot((salary_train['capitalloss'] * salary_train['capitalloss']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['capitalloss']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['capitalloss'])*np.exp(salary_train['capitalloss']),dist="norm",plot=pylab)
reci_4=1/salary_train['capitalloss']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((salary_train['capitalloss'] * salary_train['capitalloss'])+salary_train['capitalloss']),dist="norm",plot=pylab)
stats.probplot(salary_train['hoursperweek'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_train['hoursperweek']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_train['hoursperweek']),dist="norm",plot=pylab)
stats.probplot((salary_train['hoursperweek'] * salary_train['hoursperweek']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['hoursperweek']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_train['hoursperweek'])*np.exp(salary_train['hoursperweek']),dist="norm",plot=pylab)
reci_5=1/salary_train['hoursperweek']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((salary_train['hoursperweek'] * salary_train['hoursperweek'])+salary_train['hoursperweek']),dist="norm",plot=pylab)
# ppf => Percent point function 
####age
stats.norm.ppf(0.975,38.438115,13.134830)# similar to qnorm in R ---- 64.18
# cdf => cumulative distributive function 
stats.norm.cdf(salary_train["age"],38.438115,13.134830) # similar to pnorm in R 
#### educationno
stats.norm.ppf(0.975,10.121316,2.550037)# similar to qnorm in R ---- 15.12
# cdf => cumulative distributive function 
stats.norm.cdf(salary_train["educationno"],10.121316,2.550037) # similar to pnorm in R 
#### capitalgain
stats.norm.ppf(0.975,1092.044064,7406.466611)# similar to qnorm in R ---- 15608.45
# cdf => cumulative distributive function 
stats.norm.cdf(salary_train["capitalgain"],1092.044064,7406.466611) # similar to pnorm in R 
####capitalloss
stats.norm.ppf(0.975, 88.302311, 404.121321)# similar to qnorm in R ---- 880.36
# cdf => cumulative distributive function 
stats.norm.cdf(salary_train["capitalloss"], 88.302311, 404.121321) # similar to pnorm in R 
#### hoursperweek
stats.norm.ppf(0.975,40.931269, 11.980182)# similar to qnorm in R ---- 64.41
# cdf => cumulative distributive function 
stats.norm.cdf(salary_train["hoursperweek"],40.931269,11.980182) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
salary_train.corr(method = "pearson")
salary_train.corr(method = "kendall")
salary_train["age"].corr(salary_train["educationno"]) # # correlation value between X and Y -- 0.04
salary_train["age"].corr(salary_train["capitalgain"])  ### 0.08
salary_train["age"].corr(salary_train["capitalloss"]) ### 0.06
salary_train["age"].corr(salary_train["hoursperweek"])  ## 0.10
salary_train["educationno"].corr(salary_train["capitalgain"])  ###  0.12
salary_train["educationno"].corr(salary_train["capitalloss"])   ##  0.08
salary_train["educationno"].corr(salary_train["hoursperweek"]) #### 0.15
salary_train["capitalgain"].corr(salary_train["capitalloss"])   ##  -0.03
salary_train["capitalgain"].corr(salary_train["hoursperweek"]) #### 0.08
salary_train["capitalloss"].corr(salary_train["hoursperweek"]) #### 0.05
np.corrcoef(salary_train["age"],salary_train["educationno"])
np.corrcoef(salary_train["age"],salary_train["capitalgain"])
np.corrcoef(salary_train["age"],salary_train["capitalloss"])
np.corrcoef(salary_train["age"],salary_train["houseperweek"])
np.corrcoef(salary_train["educationno"],salary_train["capitalgain"])
np.corrcoef(salary_train["educationno"],salary_train["capitalloss"])
np.corrcoef(salary_train["educationno"],salary_train["houseperweek"])
np.corrcoef(salary_train["capitalgain"],salary_train["capitalloss"])
np.corrcoef(salary_train["capitalgain"],salary_train["hoursperweek"])
np.corrcoef(salary_train["capitalloss"],salary_train["hoursperweek"])
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(salary_train['age'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(salary_train['educationno'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(salary_train['capitalgain'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(salary_train['capitalloss'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(salary_train['hoursperweek'])
normalized_E = preprocessing.normalize([e_array])
#### to get top 6 rows
salary_train.head(10) # to get top n rows use cars.head(10)
salary_train.tail(10)
# Correlation matrix 
salary_train.corr()
# We see there exists High collinearity between input variables especially between
# [Profit & R&D Spend] , [Profit & Marketing Spend],[R&D Spend & Marketing Spend]
## so there exists collinearity problem
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(salary_train)
##sns.tools.plotting.scatter_matrix(salary_train) ##-> also used for plotting all in one graph
salary_test=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\SalaryData_Test.csv")
salary_test.columns
##salary_test.drop(["Unnamed: 0"],axis=1)
#####Exploratory Data Analysis#########################################################
salary_test.mean() ## age - 38.438115,educationno-10.121316,capitalgain-1092.044064,capitalloss-88.302311,hoursperweek-40.931269
salary_test.median() ### age - 37,educationno-10,capitalgain-0.0,capitalloss-0.0,hoursperweek-40
salary_test.mode() 
####Measures of Dispersion
salary_test.var() 
salary_test.std() #### age - 13.134830,educationno-2.550037,capitalgain-7406.466611,capitalloss-404.121321,hoursperweek-11.980182
#### Calculate the range value
range1 = max(salary_test['age'])-min(salary_test['age'])  ### 73
range2 = max(salary_test['educationno'])-min(salary_test['educationno']) ### 15
range3 = max(salary_test['capitalgain'])-min(salary_test['capitalgain']) ### 99999
range4 = max(salary_test['capitalloss'])-min(salary_test['capitalloss']) ### 3770
range5 = max(salary_test['hoursperweek'])-min(salary_test['hoursperweek']) ## 98
### Calculate skewness and Kurtosis
salary_test.skew()
salary_test.kurt() 
####Graphidelivery_time Representation 
plt.hist(salary_test["age"])
plt.hist(salary_test["educationno"])
plt.hist(salary_test["capitalgain"])
plt.hist(salary_test["capitalloss"])
plt.hist(salary_test["hoursperweek"])
plt.boxplot(salary_test["age"],0,"rs",0)
plt.boxplot(salary_test["educationno"],0,"rs",0)
plt.boxplot(salary_test["capitalgain"],0,"rs",0)
plt.boxplot(salary_test["capitalloss"],0,"rs",0)
plt.boxplot(salary_test["hoursperweek"],0,"rs",0)
plt.plot(salary_test["age"],salary_test["Salary"],"bo");plt.xlabel("age");plt.ylabel("Salary")
plt.plot(salary_test["educationno"],salary_test["Salary"],"bo");plt.xlabel("educationno");plt.ylabel("Salary")
plt.plot(salary_test["capitalgain"],salary_test["Salary"],"bo");plt.xlabel("capitalgain");plt.ylabel("Salary")
plt.plot(salary_test["capitalloss"],salary_test["Salary"],"bo");plt.xlabel("capitalloss");plt.ylabel("Salary")
plt.plot(salary_test["hoursperweek"],salary_test["Salary"],"bo");plt.xlabel("hoursperweek");plt.ylabel("Salary")
plt.plot(salary_test["educationno"],salary_test["age"],"bo");plt.xlabel("educationno");plt.ylabel("age")
plt.plot(salary_test["capitalgain"],salary_test["age"],"bo");plt.xlabel("capitalgain");plt.ylabel("age")
plt.plot(salary_test["capitalloss"],salary_test["age"],"bo");plt.xlabel("capitalloss");plt.ylabel("age")
plt.plot(salary_test["hoursperweek"],salary_test["age"],"bo");plt.xlabel("hoursperweek");plt.ylabel("age")
plt.plot(salary_test["capitalgain"],salary_test["educationno"],"bo");plt.xlabel("capitalgain");plt.ylabel("educationno")
plt.plot(salary_test["capitalloss"],salary_test["educationno"],"bo");plt.xlabel("capitalloss");plt.ylabel("educationno")
plt.plot(salary_test["hoursperweek"],salary_test["educationno"],"bo");plt.xlabel("hoursperweek");plt.ylabel("educationno")
plt.plot(salary_test["capitalloss"],salary_test["capitalgain"],"bo");plt.xlabel("capitalloss");plt.ylabel("capitalgain")
plt.plot(salary_test["hoursperweek"],salary_test["capitalgain"],"bo");plt.xlabel("hoursperweek");plt.ylabel("capitalgain")
plt.plot(salary_test["hoursperweek"],salary_test["capitalloss"],"bo");plt.xlabel("hoursperweek");plt.ylabel("capitalloss")
## Barplot
pd.crosstab(salary_test["age"],salary_test["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["educationno"],salary_test["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalgain"],salary_test["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalloss"],salary_test["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["hoursperweek"],salary_test["Salary"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["educationno"],salary_test["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalgain"],salary_test["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalloss"],salary_test["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["hoursperweek"],salary_test["age"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalgain"],salary_test["educationno"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalloss"],salary_test["educationno"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["hoursperweek"],salary_test["educationno"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["capitalloss"],salary_test["capitalgain"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["hoursperweek"],salary_test["capitalgain"]).plot(kind = "bar",width=1.85)
pd.crosstab(salary_test["hoursperweek"],salary_test["capitalloss"]).plot(kind = "bar",width=1.85)
import seaborn as sns 
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="age",y="Salary",data=salary_test)
sns.boxplot(x="educationno",y="Salary",data=salary_test)
sns.boxplot(x="capitalgain",y="Salary",data=salary_test)
sns.boxplot(x="capitalloss",y="Salary",data=salary_test)
sns.boxplot(x="hoursperweek",y="Salary",data=salary_test)
sns.boxplot(x="educationno",y="age",data=salary_test)
sns.boxplot(x="capitalgain",y="age",data=salary_test)
sns.boxplot(x="capitalloss",y="age",data=salary_test)
sns.boxplot(x="hoursperweek",y="age",data=salary_test)
sns.boxplot(x="capitalgain",y="educationno",data=salary_test)
sns.boxplot(x="capitalloss",y="educationno",data=salary_test)
sns.boxplot(x="hoursperweek",y="educationno",data=salary_test)
sns.boxplot(x="capitalloss",y="capitalgain",data=salary_test)
sns.boxplot(x="hoursperweek",y="capitalgain",data=salary_test)
sns.boxplot(x="hoursperweek",y="capitalloss",data=salary_test)
sns.pairplot(salary_test.iloc[:,0:13]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(salary_test,hue="Salary",size=5)
salary_test["age"].value_counts()
salary_test["educationno"].value_counts()
salary_test["capitalgain"].value_counts()
salary_test["capitalloss"].value_counts()
salary_test["hoursperweek"].value_counts()
salary_test["age"].value_counts().plot(kind="pie")
salary_test["educationno"].value_counts().plot(kind="pie")
salary_test["capitalgain"].value_counts().plot(kind="pie")
salary_test["capitalloss"].value_counts().plot(kind="pie")
salary_test["hoursperweek"].value_counts().plot(kind="pie")
sns.pairplot(salary_test,hue="Salary",size=4,diag_kind = "kde")
sns.FacetGrid(salary_test,hue="Salary").map(plt.scatter,"age","Salary").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(plt.scatter,"educationno","Salary").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(plt.scatter,"capitalgain","Salary").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(plt.scatter,"capitalloss","Salary").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(plt.scatter,"hoursperweek","Salary").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(sns.kdeplot,"age").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(sns.kdeplot,"educationno").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(sns.kdeplot,"capitalgain").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(sns.kdeplot,"capitalloss").add_legend()
sns.FacetGrid(salary_test,hue="Salary").map(sns.kdeplot,"hoursperweek").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(salary_test['age'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_test['age']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_test['age']),dist="norm",plot=pylab)
stats.probplot((salary_test['age'] * salary_test['age']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['age']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['age'])*np.exp(salary_test['age']),dist="norm",plot=pylab)
reci_6=1/salary_test['age']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((salary_test['age'] * salary_test['age'])+salary_test['age']),dist="norm",plot=pylab)
stats.probplot(salary_test['educationno'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_test['educationno']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_test['educationno']),dist="norm",plot=pylab)
stats.probplot((salary_test['educationno'] * salary_test['educationno']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['educationno']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['educationno'])*np.exp(salary_test['educationno']),dist="norm",plot=pylab)
reci_7=1/salary_test['educationno']
reci_7_2=reci_7 * reci_7
reci_7_4=reci_7_2 * reci_7_2
stats.probplot(reci_7*reci_7,dist="norm",plot=pylab)
stats.probplot(reci_7_2,dist="norm",plot=pylab)
stats.probplot(reci_7_4,dist="norm",plot=pylab)
stats.probplot(reci_7_4*reci_7_4,dist="norm",plot=pylab)
stats.probplot((reci_7_4*reci_7_4)*(reci_7_4*reci_7_4),dist="norm",plot=pylab)
stats.probplot(((salary_test['educationno'] * salary_test['educationno'])+salary_test['educationno']),dist="norm",plot=pylab)
stats.probplot(salary_test['capitalgain'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_test['capitalgain']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_test['capitalgain']),dist="norm",plot=pylab)
stats.probplot((salary_test['capitalgain'] * salary_test['capitalgain']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['capitalgain']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['capitalgain'])*np.exp(salary_test['capitalgain']),dist="norm",plot=pylab)
reci_8=1/salary_test['capitalgain']
reci_8_2=reci_8 * reci_8
reci_8_4=reci_8_2 * reci_8_2
stats.probplot(reci_8*reci_8,dist="norm",plot=pylab)
stats.probplot(reci_8_2,dist="norm",plot=pylab)
stats.probplot(reci_8_4,dist="norm",plot=pylab)
stats.probplot(reci_8_4*reci_8_4,dist="norm",plot=pylab)
stats.probplot((reci_8_4*reci_8_4)*(reci_8_4*reci_8_4),dist="norm",plot=pylab)
stats.probplot(((salary_test['capitalgain'] * salary_test['capitalgain'])+salary_test['capitalgain']),dist="norm",plot=pylab)
stats.probplot(salary_test['capitalloss'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_test['capitalloss']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_test['capitalloss']),dist="norm",plot=pylab)
stats.probplot((salary_test['capitalloss'] * salary_test['capitalloss']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['capitalloss']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['capitalloss'])*np.exp(salary_test['capitalloss']),dist="norm",plot=pylab)
reci_9=1/salary_test['capitalloss']
reci_9_2=reci_9 * reci_9
reci_9_4=reci_9_2 * reci_9_2
stats.probplot(reci_9*reci_9,dist="norm",plot=pylab)
stats.probplot(reci_9_2,dist="norm",plot=pylab)
stats.probplot(reci_9_4,dist="norm",plot=pylab)
stats.probplot(reci_9_4*reci_9_4,dist="norm",plot=pylab)
stats.probplot((reci_9_4*reci_9_4)*(reci_9_4*reci_9_4),dist="norm",plot=pylab)
stats.probplot(((salary_test['capitalloss'] * salary_test['capitalloss'])+salary_test['capitalloss']),dist="norm",plot=pylab)
stats.probplot(salary_test['hoursperweek'],dist="norm",plot=pylab)
stats.probplot(np.log(salary_test['hoursperweek']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(salary_test['hoursperweek']),dist="norm",plot=pylab)
stats.probplot((salary_test['hoursperweek'] * salary_test['hoursperweek']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['hoursperweek']),dist="norm",plot=pylab)
stats.probplot(np.exp(salary_test['hoursperweek'])*np.exp(salary_test['hoursperweek']),dist="norm",plot=pylab)
reci_10=1/salary_test['hoursperweek']
reci_10_2=reci_10 * reci_10
reci_10_4=reci_10_2 * reci_10_2
stats.probplot(reci_10*reci_10,dist="norm",plot=pylab)
stats.probplot(reci_10_2,dist="norm",plot=pylab)
stats.probplot(reci_10_4,dist="norm",plot=pylab)
stats.probplot(reci_10_4*reci_10_4,dist="norm",plot=pylab)
stats.probplot((reci_10_4*reci_10_4)*(reci_10_4*reci_10_4),dist="norm",plot=pylab)
stats.probplot(((salary_test['hoursperweek'] * salary_test['hoursperweek'])+salary_test['hoursperweek']),dist="norm",plot=pylab)
# ppf => Percent point function 
####age
stats.norm.ppf(0.975,38.438115,13.134830)# similar to qnorm in R ---- 64.18
# cdf => cumulative distributive function 
stats.norm.cdf(salary_test["age"],38.438115,13.134830) # similar to pnorm in R 
#### educationno
stats.norm.ppf(0.975,10.121316,2.550037)# similar to qnorm in R ---- 15.12
# cdf => cumulative distributive function 
stats.norm.cdf(salary_test["educationno"],10.121316,2.550037) # similar to pnorm in R 
#### capitalgain
stats.norm.ppf(0.975,1092.044064,7406.466611)# similar to qnorm in R ---- 15608.45
# cdf => cumulative distributive function 
stats.norm.cdf(salary_test["capitalgain"],1092.044064,7406.466611) # similar to pnorm in R 
####capitalloss
stats.norm.ppf(0.975, 88.302311, 404.121321)# similar to qnorm in R ---- 880.36
# cdf => cumulative distributive function 
stats.norm.cdf(salary_test["capitalloss"], 88.302311, 404.121321) # similar to pnorm in R 
#### hoursperweek
stats.norm.ppf(0.975,40.931269, 11.980182)# similar to qnorm in R ---- 64.41
# cdf => cumulative distributive function 
stats.norm.cdf(salary_test["hoursperweek"],40.931269,11.980182) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
salary_test.corr(method = "pearson")
salary_test.corr(method = "kendall")
salary_test["age"].corr(salary_test["educationno"]) # # correlation value between X and Y -- 0.026
salary_test["age"].corr(salary_test["capitalgain"])  ### 0.08
salary_test["age"].corr(salary_test["capitalloss"]) ### 0.06
salary_test["age"].corr(salary_test["hoursperweek"])  ## 0.10
salary_test["educationno"].corr(salary_test["capitalgain"])  ###  0.12
salary_test["educationno"].corr(salary_test["capitalloss"])   ##  0.08
salary_test["educationno"].corr(salary_test["hoursperweek"]) #### 0.15
salary_test["capitalgain"].corr(salary_test["capitalloss"])   ##  -0.03
salary_test["capitalgain"].corr(salary_test["hoursperweek"]) #### 0.08
salary_test["capitalloss"].corr(salary_test["hoursperweek"]) #### 0.05
np.corrcoef(salary_test["age"],salary_test["educationno"])
np.corrcoef(salary_test["age"],salary_test["capitalgain"])
np.corrcoef(salary_test["age"],salary_test["capitalloss"])
np.corrcoef(salary_test["age"],salary_test["houseperweek"])
np.corrcoef(salary_test["educationno"],salary_test["capitalgain"])
np.corrcoef(salary_test["educationno"],salary_test["capitalloss"])
np.corrcoef(salary_test["educationno"],salary_test["houseperweek"])
np.corrcoef(salary_test["capitalgain"],salary_test["capitalloss"])
np.corrcoef(salary_test["capitalgain"],salary_test["hoursperweek"])
np.corrcoef(salary_test["capitalloss"],salary_test["hoursperweek"])
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(salary_test['age'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(salary_test['educationno'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(salary_test['capitalgain'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(salary_test['capitalloss'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(salary_test['hoursperweek'])
normalized_E = preprocessing.normalize([e_array])
#### to get top 6 rows
salary_test.head(10) # to get top n rows use cars.head(10)
salary_test.tail(10)
# Correlation matrix 
salary_test.corr()
# We see there exists High collinearity between input variables especially between
# [Profit & R&D Spend] , [Profit & Marketing Spend],[R&D Spend & Marketing Spend]
## so there exists collinearity problem
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(salary_test)
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
salary_train = salary_train[salary_train.columns[salary_train.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
salary_train = salary_train.loc[salary_train.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
salary_train = salary_train.fillna(salary_train.median())
salary_train['workclass'].fillna(salary_train['workclass'].value_counts().idxmax(), inplace=True)
salary_train['education'].fillna(salary_train['education'].value_counts().idxmax(), inplace=True)
salary_train['maritalstatus'].fillna(salary_train['maritalstatus'].value_counts().idxmax(), inplace=True)
salary_train['occupation'].fillna(salary_train['occupation'].value_counts().idxmax(), inplace=True)
salary_train['relationship'].fillna(salary_train['relationship'].value_counts().idxmax(), inplace=True)
salary_train['race'].fillna(salary_train['race'].value_counts().idxmax(), inplace=True)
salary_train['sex'].fillna(salary_train['sex'].value_counts().idxmax(), inplace=True)
salary_train['native'].fillna(salary_train['native'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##salary_train['column_name'].fillna(salary_train['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = salary_train['age'].mean () + salary_train['age'].std () * factor   
lower_lim1= salary_train['age'].mean () - salary_train['age'].std () * factor 
salary_train1 = salary_train[(salary_train['age'] < upper_lim1) & (salary_train['age'] > lower_lim1)]
upper_lim2 = salary_train['educationno'].mean () + salary_train['educationno'].std () * factor   
lower_lim2= salary_train['educationno'].mean () - salary_train['educationno'].std () * factor 
salary_train2 = salary_train[(salary_train['educationno'] < upper_lim2) & (salary_train['educationno'] > lower_lim2)]
upper_lim3 = salary_train['capitalgain'].mean () + salary_train['capitalgain'].std () * factor  
lower_lim3 = salary_train['capitalgain'].mean () - salary_train['capitalgain'].std () * factor 
salary_train3 = salary_train[(salary_train['capitalgain'] < upper_lim3) & (salary_train['capitalgain'] > lower_lim3)]
upper_lim4 = salary_train['capitalloss'].mean () + salary_train['capitalloss'].std () * factor  
lower_lim4 = salary_train['capitalloss'].mean () - salary_train['capitalloss'].std () * factor 
salary_train4 = salary_train[(salary_train['capitalloss'] < upper_lim4) & (salary_train['capitalloss'] > lower_lim4)]
upper_lim5 = salary_train['hoursperweek'].mean () + salary_train['hoursperweek'].std () * factor  
lower_lim5 = salary_train['hoursperweek'].mean () - salary_train['hoursperweek'].std () * factor 
salary_train5 = salary_train[(salary_train['hoursperweek'] < upper_lim5) & (salary_train['hoursperweek'] > lower_lim5)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim6 = salary_train['age'].quantile(.95)
lower_lim6 = salary_train['age'].quantile(.05)
salary_train6 = salary_train[(salary_train['age'] < upper_lim6) & (salary_train['age'] > lower_lim6)]
upper_lim7 = salary_train['educationno'].quantile(.95)
lower_lim7 = salary_train['educationno'].quantile(.05)
salary_train7 = salary_train[(salary_train['educationno'] < upper_lim7) & (salary_train['educationno'] > lower_lim7)]
upper_lim8 = salary_train['capitalgain'].quantile(.95)
lower_lim8 = salary_train['capitalgain'].quantile(.05)
salary_train8 = salary_train[(salary_train['capitalgain'] < upper_lim8) & (salary_train['capitalgain'] > lower_lim8)]
upper_lim9 = salary_train['capitalloss'].quantile(.95)
lower_lim9 = salary_train['capitalloss'].quantile(.05)
salary_train9 = salary_train[(salary_train['capitalloss'] < upper_lim9) & (salary_train['capitalloss'] > lower_lim9)]
upper_lim10 = salary_train['hoursperweek'].quantile(.95)
lower_lim10 = salary_train['hoursperweek'].quantile(.05)
salary_train10 = salary_train[(salary_train['hoursperweek'] < upper_lim10) & (salary_train['hoursperweek'] > lower_lim10)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
salary_train.loc[(salary_train['age'] > upper_lim6)] = upper_lim6
salary_train.loc[(salary_train['age'] < lower_lim6)] = lower_lim6
salary_train.loc[(salary_train['educationno'] > upper_lim7)] = upper_lim7
salary_train.loc[(salary_train['educationno'] < lower_lim7)] = lower_lim7
salary_train.loc[(salary_train['capitalgain'] > upper_lim8)] = upper_lim8
salary_train.loc[(salary_train['capitalgain'] < lower_lim8)] = lower_lim8
salary_train.loc[(salary_train['capitalloss'] > upper_lim9)] = upper_lim9
salary_train.loc[(salary_train['capitalloss'] < lower_lim9)] = lower_lim9
salary_train.loc[(salary_train['hoursperweek'] > upper_lim10)] = upper_lim10
salary_train.loc[(salary_train['hoursperweek'] < lower_lim10)] = lower_lim10
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
salary_train['bin1'] = pd.cut(salary_train['age'], bins=[17,50,90], labels=["Adult","Senior Citizen"])
salary_train['bin2'] = pd.cut(salary_train['educationno'], bins=[1,8,16], labels=["Low","Good"])
salary_train['bin3'] = pd.cut(salary_train['capitalgain'], bins=[0,45000,99999], labels=["Good","Superb"])
salary_train['bin4'] = pd.cut(salary_train['capitalloss'], bins=[0,2000,4356], labels=["Less","More"])
salary_train['bin5'] = pd.cut(salary_train['hoursperweek'], bins=[1,50,99], labels=["Good","High"])
conditions = [
    salary_train['workclass'].str.contains('Federal-gov'),
    salary_train['workclass'].str.contains('Local-gov'),
    salary_train['workclass'].str.contains('Private'),
    salary_train['workclass'].str.contains('Self-emp-inc'),
    salary_train['workclass'].str.contains('Self-emp-not-inc'),
    salary_train['workclass'].str.contains('State-gov'),
    salary_train['workclass'].str.contains('Without-Pay')]
choices=['1','2','3','4','5','6','7']
salary_train['choices']=np.select(conditions,choices,default='Other')
conditions1 = [
    salary_train['education'].str.contains('10th'),
    salary_train['education'].str.contains('11th'),    
    salary_train['education'].str.contains('12th'),
    salary_train['education'].str.contains('1st-4th'),
    salary_train['education'].str.contains('5th-6th'),
    salary_train['education'].str.contains('7th-8th'),
    salary_train['education'].str.contains('9th'),
    salary_train['education'].str.contains('Assoc-acdm'),
    salary_train['education'].str.contains('Assoc-voc'),
    salary_train['education'].str.contains('Bachelors'),
    salary_train['education'].str.contains('Doctorate'),
    salary_train['education'].str.contains('HS-grad'),
    salary_train['education'].str.contains('Masters'),
    salary_train['education'].str.contains('Preschool'),
    salary_train['education'].str.contains('Prof-School'),
    salary_train['education'].str.contains('Some-College')]
choices1= ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
salary_train['choices1']=np.select(conditions1,choices1,default='Other')
conditions2 = [
    salary_train['maritalstatus'].str.contains('Divorced'),
    salary_train['maritalstatus'].str.contains('Married-AF-spouse'),
    salary_train['maritalstatus'].str.contains('Married-civ-spouse'),
    salary_train['maritalstatus'].str.contains('Married-spouse-absent'),
    salary_train['maritalstatus'].str.contains('Never married'),
    salary_train['maritalstatus'].str.contains('Separated'),
    salary_train['maritalstatus'].str.contains('Widowed')]   
choices2= ['1','2','3','4','5','6','7']
salary_train['choices2']=np.select(conditions2,choices2,default='Other')
conditions3 = [
    salary_train['occupation'].str.contains('Adm-clerical'),
    salary_train['occupation'].str.contains('Armed-Forces'),
    salary_train['occupation'].str.contains('Craft-repair'),
    salary_train['occupation'].str.contains('Exec-managerial'),
    salary_train['occupation'].str.contains('Farming-fishing'),
    salary_train['occupation'].str.contains('Handlers-cleaners'),
    salary_train['occupation'].str.contains('Machine-op-inspct'),
    salary_train['occupation'].str.contains('Other-service'),
    salary_train['occupation'].str.contains('Priv-house-ser'),
    salary_train['occupation'].str.contains('Prof-specialty'),
    salary_train['occupation'].str.contains('Protective-serv'),
    salary_train['occupation'].str.contains('Sales'),
    salary_train['occupation'].str.contains('Tech-support'),
    salary_train['occupation'].str.contains('Transport-moving')]   
choices3= ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
salary_train['choices3']=np.select(conditions3,choices3,default='Other')
conditions4 = [
    salary_train['relationship'].str.contains('Husband'),
    salary_train['relationship'].str.contains('Not-in-family'),
    salary_train['relationship'].str.contains('Other-relative'),
    salary_train['relationship'].str.contains('Own-child'),
    salary_train['relationship'].str.contains('Unmarried'),
    salary_train['relationship'].str.contains('Wife')]   
choices4= ['1','2','3','4','5','6']
salary_train['choices4']=np.select(conditions4,choices4,default='Other')
conditions5 = [
    salary_train['race'].str.contains('Amer-Indian-Eskimo'),
    salary_train['race'].str.contains('Asian-Pac-Islander'),
    salary_train['race'].str.contains('Black'),
    salary_train['race'].str.contains('Other'),
    salary_train['race'].str.contains('White')]   
choices5= ['1','2','3','4','5']
salary_train['choices5']=np.select(conditions5,choices5,default='Others')
conditions6 = [
    salary_train['sex'].str.contains('Male'),
    salary_train['sex'].str.contains('Female')]
choices6= ['1','2']
salary_train['choices6']=np.select(conditions6,choices6,default='Others')
conditions7 = [
    salary_train['Salary'].str.contains('<=50K'),
    salary_train['Salary'].str.contains('>50K')]
choices7= ['1','2']
salary_train['choices7']=np.select(conditions7,choices7,default='Others')
conditions8 = [
    salary_train['native'].str.contains('Cambodia'),
    salary_train['native'].str.contains('Canada'),
    salary_train['native'].str.contains('China'),
    salary_train['native'].str.contains('Columbia'),
    salary_train['native'].str.contains('Cuba'),
    salary_train['native'].str.contains('Dominican-Republic'),
    salary_train['native'].str.contains('Ecuador'),
    salary_train['native'].str.contains('El-Salvador'),
    salary_train['native'].str.contains('England'),
    salary_train['native'].str.contains('France'),
    salary_train['native'].str.contains('Germany'),
    salary_train['native'].str.contains('Greece'),
    salary_train['native'].str.contains('Guatemala'),
    salary_train['native'].str.contains('Haiti'),
    salary_train['native'].str.contains('Honduras'),
    salary_train['native'].str.contains('Hong'),
    salary_train['native'].str.contains('Hungary'),
    salary_train['native'].str.contains('India'),
    salary_train['native'].str.contains('Iran'),
    salary_train['native'].str.contains('Ireland'),
    salary_train['native'].str.contains('Italy'),
    salary_train['native'].str.contains('Jamaiaca'),
    salary_train['native'].str.contains('Japan'),
    salary_train['native'].str.contains('Laos'),
    salary_train['native'].str.contains('Mexico'),
    salary_train['native'].str.contains('Nicaragua'),
    salary_train['native'].str.contains('Outlying-US(Guam-USVI-etc)'),
    salary_train['native'].str.contains('Peru'),
    salary_train['native'].str.contains('Philippines'),
    salary_train['native'].str.contains('Poland'),
    salary_train['native'].str.contains('Portugal'),
    salary_train['native'].str.contains('Puerto-Rico'),
    salary_train['native'].str.contains('Scotland'),
    salary_train['native'].str.contains('South'),
    salary_train['native'].str.contains('Taiwan'),
    salary_train['native'].str.contains('Thailand'),
    salary_train['native'].str.contains('Trinidad&Tobago'),
    salary_train['native'].str.contains('United-States'),
    salary_train['native'].str.contains('Vietnam'),
    salary_train['native'].str.contains('Yugoslavia')]
choices8= ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
           '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40']
salary_train['choices8']=np.select(conditions8,choices8,default='Others')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
salary_train = pd.DataFrame({'age':salary_train.iloc[:,0]})
salary_train['log+1'] = (salary_train['age']+1).transform(np.log)
#Negative Values Handling
salary_train['log'] = (salary_train['age']-salary_train['age'].min()+1).transform(np.log)
salary_train = pd.DataFrame({'educationno':salary_train.iloc[:,3]})
salary_train['log+1'] = (salary_train['educationno']+1).transform(np.log)
#Negative Values Handling
salary_train['log'] = (salary_train['educationno']-salary_train['educationno'].min()+1).transform(np.log)
salary_train = pd.DataFrame({'capitalgain':salary_train.iloc[:,9]})
salary_train['log+1'] = (salary_train['capitalgain']+1).transform(np.log)
#Negative Values Handling
salary_train['log'] = (salary_train['capitalgain']-salary_train['capitalgain'].min()+1).transform(np.log)
salary_train = pd.DataFrame({'capitalloss':salary_train.iloc[:,10]})
salary_train['log+1'] = (salary_train['capitalloss']+1).transform(np.log)
#Negative Values Handling
salary_train['log'] = (salary_train['capitalloss']-salary_train['capitalloss'].min()+1).transform(np.log)
salary_train = pd.DataFrame({'hoursperweek':salary_train.iloc[:,12]})
salary_train['log+1'] = (salary_train['hoursperweek']+1).transform(np.log)
#Negative Values Handling
salary_train['log'] = (salary_train['hoursperweek']-salary_train['hoursperweek'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(salary_train['workclass'])
salary_train = salary_train.join(encoded_columns.add_suffix('_workclass')).drop('workclass', axis=1) 
encoded_columns_1 = pd.get_dummies(salary_train['education'])
salary_train = salary_train.join(encoded_columns_1.add_suffix('_education')).drop('education', axis=1)    
encoded_columns_2 = pd.get_dummies(salary_train['maritalstatus'])
salary_train = salary_train.join(encoded_columns_2.add_suffix('_maritalstatus')).drop('maritalstatus', axis=1)                                  
encoded_columns_3 = pd.get_dummies(salary_train['occupation'])
salary_train = salary_train.join(encoded_columns_2.add_suffix('_occupation')).drop('occupation', axis=1)                                  
encoded_columns_4 = pd.get_dummies(salary_train['relationship'])
salary_train = salary_train.join(encoded_columns_2.add_suffix('_relationship')).drop('relationship', axis=1)
encoded_columns_5 = pd.get_dummies(salary_train['race'])
salary_train = salary_train.join(encoded_columns_2.add_suffix('_race')).drop('race', axis=1)        
encoded_columns_5 = pd.get_dummies(salary_train['sex'])
salary_train = salary_train.join(encoded_columns_2.add_suffix('_sex')).drop('sex', axis=1)        
encoded_columns_5 = pd.get_dummies(salary_train['native'])
salary_train = salary_train.join(encoded_columns_2.add_suffix('_native')).drop('native', axis=1)        
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = salary_train.groupby('Salary')
sums = grouped['Salary'].sum().add_suffix('_sum')
avgs = grouped['Salary'].mean().add_suffix('_avg')
####Categorical Column grouping
salary_train.groupby('Salary').agg(lambda x: x.value_counts().index[0])
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(salary_train.iloc[:,0:15])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(salary_train.iloc[:,0:9])
##### Feature Extraction
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
##from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X = salary_train.drop('Salary', axis=1)
Y=salary_train['Salary']
##Y=pd.concat([y,X],axis=1)
##X = pd.get_dummies(X, prefix_sep='_')
##Y['workclass']=LabelEncoder().fit_transform(salary_train['workclass'])
##Y['education']=LabelEncoder().fit_transform(salary_train['education'])
##Y['maritalstatus']=LabelEncoder().fit_transform(salary_train['maritalstatus'])
##Y['occupation']=LabelEncoder().fit_transform(salary_train['occupation'])
##Y['relationship']=LabelEncoder().fit_transform(salary_train['relationship'])
##Y['race']=LabelEncoder().fit_transform(salary_train['race'])
##Y['sex']=LabelEncoder().fit_transform(salary_train['sex'])
##Y['native']=LabelEncoder().fit_transform(salary_train['native'])
Y=LabelEncoder().fit_transform(Y)
X = pd.get_dummies(X, prefix_sep='_')
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 30161)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=30161).fit(X_Train,Y_Train)
    print(time.process_time() - start) 
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
#labels=['Normal','Excellent']
##bins=['<=50K','>50K']
#salary_train['Salary']=pd.cut(salary_train['Salary'],labels=labels,bins=bins)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, salary_train['Salary']], axis = 1)
PCA_df['Salary'] = LabelEncoder().fit_transform(PCA_df['Salary'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
Salary = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for Salary, color in zip(Salary, colors):
    plt.scatter(PCA_df.loc[PCA_df['Salary'] == Salary, 'PC1'], 
                PCA_df.loc[PCA_df['Salary'] == Salary, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 20)
plt.legend(['Normal', 'Excellent'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
###Additionally, using our two-dimensional dataset, we can now also visualize the decision boundary used by our Random Forest in order to classify each of the different data points.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 30161)
trainedforest = RandomForestClassifier(n_estimators=30161).fit(X_Reduced,Y_Reduced)
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
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.30,random_state = 30161) 
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
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=3000)
X_tsne = tsne.fit_transform(X)
print(time.process_time() - start)
forest_test(X_tsne, Y)
#####Autoencoders are a family of Machine Learning algorithms which can be used as a 
###dimensionality reduction technique. The main difference between Autoencoders and 
##other dimensionality reduction techniques is that Autoencoders use non-linear 
###transformations to project data from a high dimension to a lower one.
from tensorflow import keras
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(3, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='softmax')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
##X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=30161)
salary_train = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\SalaryData_Test.csv")
##workclass = pd.get_dummies(salary_train['workclass'], prefix_sep='_')
##education = pd.get_dummies(salary_train['education'], prefix_sep='_')
##maritalstatus = pd.get_dummies(salary_train['maritalstatus'], prefix_sep='_')
##occupation = pd.get_dummies(salary_train['occupation'], prefix_sep='_')
##relationship = pd.get_dummies(salary_train['relationship'], prefix_sep='_')
##race = pd.get_dummies(salary_train['race'], prefix_sep='_')
##sex = pd.get_dummies(salary_train['sex'], prefix_sep='_')
##native = pd.get_dummies(salary_train['native'], prefix_sep='_')
##salary_train1 = pd.concat([salary_train,workclass,education,maritalstatus,occupation,relationship,race,sex,native],axis=1)
salary_train.drop(['workclass','education','maritalstatus','occupation','relationship','race','sex','native'],axis=1,inplace=True)
###############################################################################
##workclass = pd.get_dummies(salary_test['workclass'], prefix_sep='_')
##education = pd.get_dummies(salary_test['education'], prefix_sep='_')
##maritalstatus = pd.get_dummies(salary_test['maritalstatus'], prefix_sep='_')
##occupation = pd.get_dummies(salary_test['occupation'], prefix_sep='_')
##relationship = pd.get_dummies(salary_test['relationship'], prefix_sep='_')
##race = pd.get_dummies(salary_test['race'], prefix_sep='_')
##sex = pd.get_dummies(salary_test['sex'], prefix_sep='_')
##native = pd.get_dummies(salary_test['native'], prefix_sep='_')
##salary_test1 = pd.concat([salary_test,workclass,education,maritalstatus,occupation,relationship,race,sex,native],axis=1)
salary_test.drop(['workclass','education','maritalstatus','occupation','relationship','race','sex','native'],axis=1,inplace=True)
########################################################################
trainX = salary_train.drop(["Salary"],axis=1)
trainY = salary_train["Salary"]
testX = salary_test.drop(["Salary"],axis=1)
testY = salary_test["Salary"]
autoencoder.fit(trainX, trainY,epochs=30161,batch_size=30161,shuffle=True,verbose = 30161,validation_data=(testX, testY))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)
##################################################################################
sns.pairplot(data=salary_train)
sns.pairplot(data=salary_test)
##salary_train1.drop(['Salary'])
from sklearn.svm import SVC
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(trainX,trainY)
pred_test_linear = model_linear.predict(testX)
np.mean(pred_test_linear==testY) 
# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(trainX,trainY)
pred_test_poly = model_poly.predict(testX)
np.mean(pred_test_poly==testY) 
# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(trainX,trainY)
pred_test_rbf = model_rbf.predict(testX)
np.mean(pred_test_rbf==testY) 
# kernel = tanh
model_tanh = SVC(kernel = "tanh")
model_tanh.fit(trainX,trainY)
pred_test_tanh = model_tanh.predict(testX)
np.mean(pred_test_tanh==testY) 
# kernel = laplace
model_laplace = SVC(kernel = "laplace")
model_laplace.fit(trainX,trainY)
pred_test_laplace = model_laplace.predict(testX)
np.mean(pred_test_laplace==testY) 
# kernel = bessel
model_bessel = SVC(kernel = "bessel")
model_bessel.fit(trainX,trainY)
pred_test_bessel = model_bessel.predict(testX)
np.mean(pred_test_bessel==testY) 
# kernel = anova
model_anova = SVC(kernel = "anova")
model_anova.fit(trainX,trainY)
pred_test_anova = model_anova.predict(testX)
np.mean(pred_test_anova==testY) 
# kernel = anova
model_spline = SVC(kernel = "spline")
model_spline.fit(trainX,trainY)
pred_test_spline = model_spline.predict(testX)
np.mean(pred_test_spline==testY) 
# kernel = spline
model_spline = SVC(kernel = "spline")
model_spline.fit(trainX,trainY)
pred_test_spline = model_spline.predict(testX)
np.mean(pred_test_spline==testY) 
###kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(trainX,trainY)
pred_test_sigmoid = model_sigmoid.predict(testX)
np.mean(pred_test_sigmoid==testY) 
#####################Hyperparameter Tuning Methods########################
########Grid search - Grid search is an approach where we start from preparing 
########the sets of candidates hyperparameters, train the model for every single set of them, and select the best performing set of hyperparameters.
#### Advantage is You can cover all possible prospective sets of parameters. No matter how you
#### strongly believed one set is most viable, who knows, the neighbor could be more successful. You do not lose that possibility with grid search.
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
# Proportion of validation set for early stopping in training set.
r = 0.6 
trainLen = round(len(trainX)*(1-r))
# Splitting training data to training and early stopping validation set.
X_train = trainX.iloc[:trainLen,:]
##from sklearn import MultiColumnLabelEncoder
##MultiColumnLabelEncoder(columns = ['workclass','education','educationno','maritalstatus','occupation','relationship','race','sex','native']).fit_transform(X_train)
##from sklearn.preprocessing import LabelEncoder
##labelencoder = LabelEncoder()
X_train = pd.get_dummies(X_train,suffix='-')
X_train.to_excel(r'C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\X_train.xlsx', sheet_name='X_train')
##X_train.drop([X_train.columns[0]],axis='columns')
##X_train = pd.read_excel('C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\X_train.xlsx')
##X_train.drop(['workclass','education','maritalstatus','occupation','relationship','race','sex','native'],axis=1,inplace=True)
help(pd.get_dummies(X_train))
y_train = trainY[:trainLen]
y_train = pd.get_dummies(y_train)
##y_train.drop(['Salary'],axis=1)
X_val = trainX.iloc[trainLen:,:]
X_val = pd.get_dummies(X_val)
##X_val.to_excel(r'C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\X_val.xlsx', sheet_name='X_val')
y_val = trainY[trainLen:]
y_val = pd.get_dummies(y_val, prefix_sep='-')
# Defining parameter space for grid search.
gridParams = {
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [0.1, 1.0, 2.0],
}
# Define lightgbm and grid search.
reg = LGBMRegressor(learning_rate=0.2, n_estimators=18097, random_state=18097)
reg_gridsearch = GridSearchCV(reg, gridParams, cv=5, scoring='r2', n_jobs=-1) 
# Model fit with early stopping.
reg_gridsearch.fit(X_train, y_train, early_stopping_rounds=100, eval_set=(X_val,y_val))
## Final l2 was l2: 0.0203797.
# Confirm what parameters were selected.
reg_gridsearch.best_params_
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder() 
##onehotencoder = OneHotEncoder(columns = ['workclass','education','educationno','maritalstatus','occupation','relationship','race','sex','native'])
X_train = onehotencoder.fit_transform(X_train).toarray() 
y_train = trainY[:trainLen]
y_train = onehotencoder.fit_transform([y_train]).toarray() 
X_val = trainX.iloc[trainLen:,:]
X_val = onehotencoder.fit_transform(X_val).toarray() 
y_val = trainY[trainLen:]
y_val = onehotencoder.fit_transform([y_val]).toarray() 