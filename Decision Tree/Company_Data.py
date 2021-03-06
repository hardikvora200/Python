import pandas as pd
import numpy as np
import seaborn as sns
##import scipy.stats as st
import matplotlib.pyplot as plt
# reading a csv file using pandas library
company=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Decision Tree\\Assignments\\Company_Data.csv")
##company.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
company.columns
##company.drop(["a"],axis=1,inplace=True)
##company.columns
# To get the count of null values in the data 
company.isnull().sum()
company.isna()
company.shape # 400 11 => Before dropping null values
# To drop null values ( dropping rows)
company.dropna().shape # 400 11 => After dropping null values
#####Exploratory Data Analysis#########################################################
company.mean() ## Sales- 7.496325,CompPrice-124.975000,Income- 68.657500,Advertising- 6.635000,
###### Population-264.840000,Price-115.795000,Age-53.322500,Education-13.900000
company.median() 
company.mode() 
####Measures of Dispersion
company.var() 
company.std() ## Sales-  2.824115,CompPrice-15.334512,Income- 27.986037,Advertising- 6.650364,
###### Population-147.376436,Price-23.676664,Age-16.200297,Education-2.620528
#### Calculate the range value
range1 = max(company['Sales'])-min(company['Sales'])  ### 16.27
range2 = max(company['CompPrice'])-min(company['CompPrice']) ### 98
range3 = max(company['Income'])-min(company['Income']) ### 99
range4 = max(company['Advertising'])-min(company['Advertising']) ### 29
range5 = max(company['Population'])-min(company['Population'])  ##  499
range6 = max(company['Price'])-min(company['Price']) ### 167
range7 = max(company['Age'])-min(company['Age']) #### 55
range8 = max(company['Education'])-min(company['Education'])  ## 8
### Calculate skewness and Kurtosis
company.skew()
company.kurt() 
####Graphidelivery_time Representation 
plt.hist(company["Sales"])
plt.hist(company["CompPrice"])
plt.hist(company["Income"])
plt.hist(company["Advertising"])
plt.hist(company["Population"])
plt.hist(company["Price"])
plt.hist(company["Age"])
plt.hist(company["Education"])
plt.boxplot(company["Sales"],0,"rs",0)
plt.boxplot(company["CompPrice"],0,"rs",0)
plt.boxplot(company["Income"],0,"rs",0)
plt.boxplot(company["Advertising"],0,"rs",0)
plt.boxplot(company["Population"],0,"rs",0)
plt.boxplot(company["Price"],0,"rs",0)
plt.boxplot(company["Age"],0,"rs",0)
plt.boxplot(company["Education"],0,"rs",0)
plt.plot(company["CompPrice"],company["Sales"],"bo");plt.xlabel("CompPrice");plt.ylabel("Sales")
plt.plot(company["Income"],company["Sales"],"bo");plt.xlabel("CompPrice");plt.ylabel("Sales")
plt.plot(company["Advertising"],company["Sales"],"bo");plt.xlabel("Advertising");plt.ylabel("Sales")
plt.plot(company["Population"],company["Sales"],"bo");plt.xlabel("Population");plt.ylabel("Sales")
plt.plot(company["Price"],company["Sales"],"bo");plt.xlabel("Price");plt.ylabel("Sales")
plt.plot(company["Age"],company["Sales"],"bo");plt.xlabel("Age");plt.ylabel("Sales")
plt.plot(company["Education"],company["Sales"],"bo");plt.xlabel("Education");plt.ylabel("Sales")
# table 
pd.crosstab(company["CompPrice"],company["Sales"])
pd.crosstab(company["Income"],company["Sales"])
pd.crosstab(company["Advertising"],company["Sales"])
pd.crosstab(company["Population"],company["Sales"])
pd.crosstab(company["Price"],company["Sales"])
pd.crosstab(company["Age"],company["Sales"])
pd.crosstab(company["Education"],company["Sales"])
## Barplot
pd.crosstab(company["CompPrice"],company["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(company["Income"],company["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(company["Advertising"],company["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(company["Population"],company["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(company["Price"],company["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(company["Age"],company["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(company["Education"],company["Sales"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Sales",data=company,palette="hls")
sns.countplot(x="CompPrice",data=company,palette="hls")
sns.countplot(x="Income",data=company,palette="hls")
sns.countplot(x="Advertising",data=company,palette="hls")
sns.countplot(x="Population",data=company,palette="hls")
sns.countplot(x="Price",data=company,palette="hls")
sns.countplot(x="Age",data=company,palette="hls")
sns.countplot(x="Education",data=company,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="CompPrice",y="Sales",data=company,palette="hls")
sns.boxplot(x="Income",y="Sales",data=company,palette="hls")
sns.boxplot(x="Advertising",y="Sales",data=company,palette="hls")
sns.boxplot(x="Population",y="Sales",data=company,palette="hls")
sns.boxplot(x="Price",y="Sales",data=company,palette="hls")
sns.boxplot(x="Age",y="Sales",data=company,palette="hls")
sns.boxplot(x="Education",y="Sales",data=company,palette="hls")
sns.pairplot(company.iloc[:,0:7]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(company,hue="Sales",size=2)
company["Sales"].value_counts()
company["CompPrice"].value_counts()
company["Income"].value_counts()
company["Advertising"].value_counts()
company["Population"].value_counts()
company["Price"].value_counts()
company["Age"].value_counts()
company["Education"].value_counts()
company["Sales"].value_counts().plot(kind = "pie")
company["CompPrice"].value_counts().plot(kind = "pie")
company["Income"].value_counts().plot(kind = "pie")
company["Advertising"].value_counts().plot(kind = "pie")
company["Population"].value_counts().plot(kind = "pie")
company["Price"].value_counts().plot(kind = "pie")
company["Age"].value_counts().plot(kind = "pie")
company["Education"].value_counts().plot(kind = "pie")
sns.pairplot(company,hue="Sales",size=4,diag_kind = "kde")
sns.pairplot(company,hue="CompPrice",size=4,diag_kind = "kde")
sns.pairplot(company,hue="Income",size=4,diag_kind = "kde")
sns.pairplot(company,hue="Advertising",size=4,diag_kind = "kde")
sns.pairplot(company,hue="Population",size=4,diag_kind = "kde")
sns.pairplot(company,hue="Price",size=4,diag_kind = "kde")
sns.pairplot(company,hue="Age",size=4,diag_kind = "kde")
sns.pairplot(company,hue="Education",size=4,diag_kind = "kde")
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"CompPrice","Sales").add_legend()
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"Income","Sales").add_legend()
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"Advertising","Sales").add_legend()
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"Population","Sales").add_legend()
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"Price","Sales").add_legend()
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"Age","Sales").add_legend()
sns.FacetGrid(company,hue="Sales").map(plt.scatter,"Education","Sales").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(plt.scatter,"Income","CompPrice").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(plt.scatter,"Advertising","CompPrice").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(plt.scatter,"Population","CompPrice").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(plt.scatter,"Price","CompPrice").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(plt.scatter,"Age","CompPrice").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(plt.scatter,"Education","CompPrice").add_legend()
sns.FacetGrid(company,hue="Income").map(plt.scatter,"Advertising","Income").add_legend()
sns.FacetGrid(company,hue="Income").map(plt.scatter,"Population","Income").add_legend()
sns.FacetGrid(company,hue="Income").map(plt.scatter,"Price","Income").add_legend()
sns.FacetGrid(company,hue="Income").map(plt.scatter,"Age","Income").add_legend()
sns.FacetGrid(company,hue="Income").map(plt.scatter,"Education","Income").add_legend()
sns.FacetGrid(company,hue="Advertising").map(plt.scatter,"Population","Advertising").add_legend()
sns.FacetGrid(company,hue="Advertising").map(plt.scatter,"Price","Advertising").add_legend()
sns.FacetGrid(company,hue="Advertising").map(plt.scatter,"Age","Advertising").add_legend()
sns.FacetGrid(company,hue="Advertising").map(plt.scatter,"Education","Advertising").add_legend()
sns.FacetGrid(company,hue="Population").map(plt.scatter,"Price","Population").add_legend()
sns.FacetGrid(company,hue="Population").map(plt.scatter,"Age","Population").add_legend()
sns.FacetGrid(company,hue="Population").map(plt.scatter,"Education","Population").add_legend()
sns.FacetGrid(company,hue="Price").map(plt.scatter,"Age","Price").add_legend()
sns.FacetGrid(company,hue="Price").map(plt.scatter,"Education","Price").add_legend()
sns.FacetGrid(company,hue="Age").map(plt.scatter,"Education","Age").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"CompPrice").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"Income").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"Advertising").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"Population").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"Price").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"Age").add_legend()
sns.FacetGrid(company,hue="Sales").map(sns.kdeplot,"Education").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(sns.kdeplot,"Income").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(sns.kdeplot,"Advertising").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(sns.kdeplot,"Population").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(sns.kdeplot,"Price").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(sns.kdeplot,"Age").add_legend()
sns.FacetGrid(company,hue="CompPrice").map(sns.kdeplot,"Education").add_legend()
sns.FacetGrid(company,hue="Income").map(sns.kdeplot,"Advertising").add_legend()
sns.FacetGrid(company,hue="Income").map(sns.kdeplot,"Population").add_legend()
sns.FacetGrid(company,hue="Income").map(sns.kdeplot,"Price").add_legend()
sns.FacetGrid(company,hue="Income").map(sns.kdeplot,"Age").add_legend()
sns.FacetGrid(company,hue="Income").map(sns.kdeplot,"Education").add_legend()
sns.FacetGrid(company,hue="Advertising").map(sns.kdeplot,"Population").add_legend()
sns.FacetGrid(company,hue="Advertising").map(sns.kdeplot,"Price").add_legend()
sns.FacetGrid(company,hue="Advertising").map(sns.kdeplot,"Age").add_legend()
sns.FacetGrid(company,hue="Advertising").map(sns.kdeplot,"Education").add_legend()
sns.FacetGrid(company,hue="Population").map(sns.kdeplot,"Price").add_legend()
sns.FacetGrid(company,hue="Population").map(sns.kdeplot,"Age").add_legend()
sns.FacetGrid(company,hue="Population").map(sns.kdeplot,"Education").add_legend()
sns.FacetGrid(company,hue="Price").map(sns.kdeplot,"Age").add_legend()
sns.FacetGrid(company,hue="Price").map(sns.kdeplot,"Education").add_legend()
sns.FacetGrid(company,hue="Age").map(sns.kdeplot,"Education").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab
# Checking Whether data is normally distributed
stats.probplot(company['CompPrice'],dist="norm",plot=pylab)
stats.probplot(np.log(company['CompPrice']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['CompPrice']),dist="norm",plot=pylab)
stats.probplot((company['CompPrice'] * company['CompPrice']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['CompPrice']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['CompPrice'])*np.exp(company['CompPrice']),dist="norm",plot=pylab)
reci_1=1/company['CompPrice']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((company['CompPrice'] * company['CompPrice'])+company['CompPrice']),dist="norm",plot=pylab)
stats.probplot(company['Income'],dist="norm",plot=pylab)
stats.probplot(np.log(company['Income']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['Income']),dist="norm",plot=pylab)
stats.probplot((company['Income'] * company['Income']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Income']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Income'])*np.exp(company['Income']),dist="norm",plot=pylab)
reci_2=1/company['Income']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((company['Income'] * company['Income'])+company['Income']),dist="norm",plot=pylab)
stats.probplot(company['Advertising'],dist="norm",plot=pylab)
stats.probplot(np.log(company['Advertising']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['Advertising']),dist="norm",plot=pylab)
stats.probplot((company['Advertising'] * company['Advertising']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Advertising']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Advertising'])*np.exp(company['Advertising']),dist="norm",plot=pylab)
reci_3=1/company['Advertising']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((company['Advertising'] * company['Advertising'])+company['Advertising']),dist="norm",plot=pylab)
stats.probplot(company['Population'],dist="norm",plot=pylab)
stats.probplot(np.log(company['Population']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['Population']),dist="norm",plot=pylab)
stats.probplot((company['Population'] * company['Population']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Population']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Population'])*np.exp(company['Population']),dist="norm",plot=pylab)
reci_4=1/company['Population']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((company['Population'] * company['Population'])+company['Population']),dist="norm",plot=pylab)
stats.probplot(company['Price'],dist="norm",plot=pylab)
stats.probplot(np.log(company['Price']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['Price']),dist="norm",plot=pylab)
stats.probplot((company['Price'] * company['Price']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Price']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Price'])*np.exp(company['Price']),dist="norm",plot=pylab)
reci_5=1/company['Price']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((company['Price'] * company['Price'])+company['Price']),dist="norm",plot=pylab)
stats.probplot(company['Age'],dist="norm",plot=pylab)
stats.probplot(np.log(company['Age']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['Age']),dist="norm",plot=pylab)
stats.probplot((company['Age'] * company['Age']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Age']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Age'])*np.exp(company['Age']),dist="norm",plot=pylab)
reci_6=1/company['Age']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((company['Age'] * company['Age'])+company['Age']),dist="norm",plot=pylab)
stats.probplot(company['Education'],dist="norm",plot=pylab)
stats.probplot(np.log(company['Education']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(company['Education']),dist="norm",plot=pylab)
stats.probplot((company['Education'] * company['Education']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Education']),dist="norm",plot=pylab)
stats.probplot(np.exp(company['Education'])*np.exp(company['Education']),dist="norm",plot=pylab)
rec_7=1/company['Education']
rec_7_2=rec_7 * rec_7
rec_7_4=rec_7_2 * rec_7_2
stats.probplot(rec_7*rec_7,dist="norm",plot=pylab)
stats.probplot(rec_7_2,dist="norm",plot=pylab)
stats.probplot(rec_7_4,dist="norm",plot=pylab)
stats.probplot(rec_7_4*rec_7_4,dist="norm",plot=pylab)
stats.probplot((rec_7_4*rec_7_4)*(rec_7_4*rec_7_4),dist="norm",plot=pylab)
stats.probplot(((company['Education'] * company['Education'])+company['Education']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Sales
stats.norm.ppf(0.975,7.496325, 2.824115)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Sales"],7.496325, 2.824115) # similar to pnorm in R 
#### CompPrice
stats.norm.ppf(0.975,124.975000,15.334512)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["CompPrice"],124.975000,15.334512) # similar to pnorm in R 
#### Income
stats.norm.ppf(0.975,68.657500,27.986037)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Income"],68.657500,27.986037) # similar to pnorm in R 
#### Advertising
stats.norm.ppf(0.975, 6.635000, 6.650364)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Advertising"], 6.635000,6.650364) # similar to pnorm in R 
#### Population
stats.norm.ppf(0.975,264.840000,147.376436)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Population"],264.840000,147.376436) # similar to pnorm in R 
#### Price
stats.norm.ppf(0.975, 115.795000, 23.676664)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Price"],115.795000,23.676664) # similar to pnorm in R 
#### Age
stats.norm.ppf(0.975,53.322500,16.200297)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Age"],53.322500,16.200297) # similar to pnorm in R 
#### Educaton
stats.norm.ppf(0.975,13.90000,2.620528)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(company["Education"],13.90000,2.620528) # similar to pnorm in R  
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
company.corr(method = "pearson")
company.corr(method = "kendall")
# to get top 6 rows
company.head(40) # to get top n rows use cars.head(10)
company.tail(10)
### Normalization
##def norm_func(i):
  ##  x = (i-i.mean())/(i.std())
 ##   return (x)
### Normalized data frame (considering the numerical part of data)
##df_norm = norm_func(company.iloc[:,0:])
# Scatter plot between the variables along with histograms
sns.pairplot(company)
company.dropna()
company['Sales'].unique()
labels=['average','best']
bins=[0,10,16.27]
company['Sales']=pd.cut(company['Sales'],bins=bins,labels=labels)
company.head()
y=company.iloc[:,10]
company1= company.drop('Sales', axis=1)
company2=pd.concat([y,company1],axis=1)
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
company2['ShelveLoc']=lb_make.fit_transform(company['ShelveLoc'])
company2['Urban']=lb_make.fit_transform(company['Urban'])
company2['US']=lb_make.fit_transform(company['US'])
colnames=list(company2.columns)
predx= colnames[1:11]
predy= colnames[0]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
train,test=train_test_split(company2,test_size=0.3)
train.dropna()
model=DecisionTreeClassifier(criterion='entropy')
train = train.fillna(train.median())
model.fit(train[predx],train[predy])
preds=model.predict(test[predx])
pd.Series(preds).value_counts()
pd.crosstab(test[predy],preds)
np.mean(train['Sales'] == model.predict(train[predx])) #100% accuracy
np.mean(preds==test['Sales'])#85.83% accuracy
###################################################################
from xgboost import XGBClassifier
seed = 7
test_size = 0.2
model_1=XGBClassifier()
model_1.fit(train[predx],train[predy])
print(model)
##### Lets do predictions with XGBoost model with test data
target_pred=model_1.predict(test[predx])
predictions = [round(value) for value in target_pred]
np.mean(train['Sales'] == model_1.predict(train[predx])) 
np.mean(target_pred==test['Sales']) 
## Lets check with regression metrics
from xgboost import XGBRegressor
model_2=XGBRegressor()
model_2.fit(train[predx],train[predy])
target_pred_2=model_2.predict(test[predx])
predictions_2 = [round(value) for value in target_pred_2]
np.mean(train['Sales'] == model_2.predict(train[predx])) 
np.mean(target_pred_2==test['Sales'])
 ###################################################################
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
array=company.values
company_dummies=pd.get_dummies(company[["Undergrad","Marital.Status","Urban"]])
company = pd.concat([company,company_dummies],axis=1)
company['Sales'].value_counts()
colnames=list(company.columns)
company.drop(["Undergrad"],inplace=True,axis = 1)
company.drop(["Marital.Status"],inplace=True,axis = 1)
company.drop(["Urban"],inplace=True,axis = 1)
X1=company.iloc[:,0:4]
X2=company.iloc[:,6:7]
X=pd.concat([X1,X2],axis=1)
X=np.reshape(-1,1)
Y=company.iloc[:,10]
##X=norm_func(X)
X=X.astype('int')
Y=Y.astype('int')
seed = 7
kfold=model_selection.KFold(n_splits=400,random_state=seed)
model=DecisionTreeRegressor()
num_trees=35
model_bagging=BaggingClassifier(base_estimator=model,n_estimators=num_trees,random_state=seed)
results=model_selection.cross_val_score(model_bagging,X,Y,cv=kfold)
print(results.mean())
max_features=2
model_rfc=RandomForestClassifier(n_estimators=num_trees,random_state=seed)
results_rfc=model_selection.cross_val_score(model_rfc,X,Y,cv=kfold)
print(results_rfc.mean())
max_features=10
model_extra=ExtraTreesClassifier(n_estimators=num_trees,max_features=max_features)
results_extra=model_selection.cross_val_score(model_extra,X,Y,cv=kfold)
print(results_extra.mean())
num_trees=30
model_adaboost=AdaBoostClassifier(n_estimators=num_trees,random_state=seed)
results_adaboost=model_selection.cross_val_score(model_adaboost,X,Y,cv=kfold)
print(results_adaboost.mean())  
### Stochastic Gradient Boosting
num_trees=60
model_stochastic=GradientBoostingClassifier(n_estimators=num_trees,random_state=seed)
results_stochastic=model_selection.cross_val_score(model_stochastic,X,Y,cv=kfold)
print(results_stochastic.mean())
###Voting Ensemble for classification
estimators=[]
model_log=LogisticRegression()
estimators.append(('abc',model_log))
model_dtc=DecisionTreeClassifier()
estimators.append(('def',model_dtc))
model_svc=SVC(gamma='auto')
estimators.append(('ghi',model_svc))
##Create the ensemble model
ensemble=VotingClassifier(estimators)
results_ens=model_selection.cross_val_score(ensemble,X,Y,cv=kfold)
print(results_ens.mean())
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
company = company[company.columns[company.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
company = company.loc[company.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
company = company.fillna(company.median())
company['ShelveLoc'].fillna(company['ShelveLoc'].value_counts().idxmax(), inplace=True)
company['Urban'].fillna(company['Urban'].value_counts().idxmax(), inplace=True)
company['US'].fillna(company['US'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##company['column_name'].fillna(company['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = company['CompPrice'].mean () + company['CompPrice'].std () * factor   
lower_lim1= company['CompPrice'].mean () - company['CompPrice'].std () * factor 
company1 = company[(company['CompPrice'] < upper_lim1) & (company['CompPrice'] > lower_lim1)]
upper_lim2 = company['Income'].mean () + company['Income'].std () * factor   
lower_lim2= company['Income'].mean () - company['Income'].std () * factor 
company2 = company[(company['Income'] < upper_lim2) & (company['Income'] > lower_lim2)]
upper_lim3 = company['Advertising'].mean () + company['Advertising'].std () * factor  
lower_lim3 = company['Advertising'].mean () - company['Advertising'].std () * factor 
company3 = company[(company['Advertising'] < upper_lim3) & (company['Advertising'] > lower_lim3)]
upper_lim4 = company['Population'].mean () + company['Population'].std () * factor  
lower_lim4 = company['Population'].mean () - company['Population'].std () * factor 
company4 = company[(company['Population'] < upper_lim4) & (company['Population'] > lower_lim4)]
upper_lim5 = company['Price'].mean () + company['Price'].std () * factor  
lower_lim5 = company['Price'].mean () - company['Price'].std () * factor 
company5 = company[(company['Price'] < upper_lim5) & (company['Price'] > lower_lim5)]
upper_lim6 = company['Age'].mean () + company['Age'].std () * factor  
lower_lim6 = company['Age'].mean () - company['Age'].std () * factor 
company6 = company[(company['Age'] < upper_lim6) & (company['Age'] > lower_lim6)]
upper_lim7 = company['Education'].mean () + company['Education'].std () * factor  
lower_lim7 = company['Education'].mean () - company['Education'].std () * factor 
company7 = company[(company['Education'] < upper_lim7) & (company['Education'] > lower_lim7)]
upper_lim8 = company['Sales'].mean () + company['Sales'].std () * factor  
lower_lim8 = company['Sales'].mean () - company['Sales'].std () * factor 
company8 = company[(company['Sales'] < upper_lim8) & (company['Sales'] > lower_lim8)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim9 = company['CompPrice'].quantile(.95)
lower_lim9 = company['CompPrice'].quantile(.05)
company9 = company[(company['CompPrice'] < upper_lim9) & (company['CompPrice'] > lower_lim9)]
upper_lim10 = company['Income'].quantile(.95)
lower_lim10 = company['Income'].quantile(.05)
company10 = company[(company['Income'] < upper_lim10) & (company['Income'] > lower_lim10)]
upper_lim11 = company['Advertising'].quantile(.95)
lower_lim11 = company['Advertising'].quantile(.05)
company11 = company[(company['Advertising'] < upper_lim11) & (company['Advertising'] > lower_lim11)]
upper_lim12 = company['Population'].quantile(.95)
lower_lim12 = company['Population'].quantile(.05)
company12 = company[(company['Population'] < upper_lim12) & (company['Population'] > lower_lim12)]
upper_lim13 = company['Price'].quantile(.95)
lower_lim13 = company['Price'].quantile(.05)
company13 = company[(company['Price'] < upper_lim13) & (company['Price'] > lower_lim13)]
upper_lim14 = company['Age'].quantile(.95)
lower_lim14 = company['Age'].quantile(.05)
company14 = company[(company['Age'] < upper_lim14) & (company['Age'] > lower_lim14)]
upper_lim15 = company['Education'].quantile(.95)
lower_lim15 = company['Education'].quantile(.05)
company15 = company[(company['Education'] < upper_lim15) & (company['Education'] > lower_lim15)]
upper_lim16 = company['Sales'].quantile(.95)
lower_lim16 = company['Sales'].quantile(.05)
company16 = company[(company['Sales'] < upper_lim16) & (company['Sales'] > lower_lim16)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
company.loc[(company['CompPrice'] > upper_lim9)] = upper_lim9
company.loc[(company['CompPrice'] < lower_lim9)] = lower_lim9
company.loc[(company['Income'] > upper_lim10)] = upper_lim10
company.loc[(company['Income'] < lower_lim10)] = lower_lim10
company.loc[(company['Advertising'] > upper_lim11)] = upper_lim11
company.loc[(company['Advertising'] < lower_lim11)] = lower_lim11
company.loc[(company['Population'] > upper_lim12)] = upper_lim12
company.loc[(company['Population'] < lower_lim12)] = lower_lim12
company.loc[(company['Price'] > upper_lim13)] = upper_lim13
company.loc[(company['Price'] < lower_lim13)] = lower_lim13
company.loc[(company['Age'] > upper_lim14)] = upper_lim14
company.loc[(company['Age'] < lower_lim14)] = lower_lim14
company.loc[(company['Education'] > upper_lim15)] = upper_lim15
company.loc[(company['Education'] < lower_lim15)] = lower_lim15
company.loc[(company['Sales'] > upper_lim16)] = upper_lim16
company.loc[(company['Sales'] < lower_lim16)] = lower_lim16
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
company['bin1'] = pd.cut(company['CompPrice'], bins=[77,125,175], labels=["Good","Best"])
company['bin2'] = pd.cut(company['Income'], bins=[21,70,120], labels=["Low","Good"])
company['bin3'] = pd.cut(company['Advertising'], bins=[0,10,29], labels=["Low","Good"])
company['bin4'] = pd.cut(company['Population'], bins=[10,200,509], labels=["Good","Over"])
company['bin5'] = pd.cut(company['Price'], bins=[24,100,191], labels=["Good","High"])
company['bin6'] = pd.cut(company['Age'], bins=[25,40,80], labels=["Normal","Over"])
company['bin7'] = pd.cut(company['Education'], bins=[10,15,18], labels=["Normal","Over"])
company['bin8'] = pd.cut(company['Sales'], bins=[0,10,16.27], labels=["Normal","Excellent"])
conditions = [
    company['ShelveLoc'].str.contains('Bad'),
    company['ShelveLoc'].str.contains('Good'),
    company['ShelveLoc'].str.contains('Medium')]
choices=['1','2','3']
company['choices']=np.select(conditions,choices,default='Other')
conditions1 = [
    company['Urban'].str.contains('Yes'),
    company['Urban'].str.contains('No')]
choices1= ['1','2']
company['choices1']=np.select(conditions1,choices1,default='Other')
conditions2 = [
    company['US'].str.contains('Yes'),
    company['US'].str.contains('No')]   
choices2= ['1','2']
company['choices2']=np.select(conditions2,choices2,default='Other')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
company = pd.DataFrame({'CompPrice':company.iloc[:,0]})
company['log+1'] = (company['CompPrice']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['CompPrice']-company['CompPrice'].min()+1).transform(np.log)
company = pd.DataFrame({'Income':company.iloc[:,1]})
company['log+1'] = (company['Income']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Income']-company['Income'].min()+1).transform(np.log)
company = pd.DataFrame({'Advertising':company.iloc[:,2]})
company['log+1'] = (company['Advertising']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Advertising']-company['Advertising'].min()+1).transform(np.log)
company = pd.DataFrame({'Population':company.iloc[:,3]})
company['log+1'] = (company['Population']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Population']-company['Population'].min()+1).transform(np.log)
company = pd.DataFrame({'Price':company.iloc[:,4]})
company['log+1'] = (company['Price']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Price']-company['Price'].min()+1).transform(np.log)
company = pd.DataFrame({'Age':company.iloc[:,6]})
company['log+1'] = (company['Age']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Age']-company['Age'].min()+1).transform(np.log)
company = pd.DataFrame({'Education':company.iloc[:,7]})
company['log+1'] = (company['Education']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Education']-company['Education'].min()+1).transform(np.log)
company = pd.DataFrame({'Sales':company.iloc[:,10]})
company['log+1'] = (company['Sales']+1).transform(np.log)
#Negative Values Handling
company['log'] = (company['Sales']-company['Sales'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(company['ShelveLoc'])
company = company.join(encoded_columns.add_suffix('_ShelveLoc')).drop('ShelveLoc', axis=1) 
encoded_columns_1 = pd.get_dummies(company['Urban'])
company = company.join(encoded_columns_1.add_suffix('_Urban')).drop('Urban', axis=1)    
encoded_columns_2 = pd.get_dummies(company['US'])
company = company.join(encoded_columns_2.add_suffix('_US')).drop('US', axis=1)                                  
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = company.groupby('Sales')
sums = grouped['Sales'].sum().add_suffix('_sum')
avgs = grouped['Sales'].mean().add_suffix('_avg')
####Categorical Column grouping
company.groupby('Sales').agg(lambda x: x.value_counts().index[0])
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(company.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(company.iloc[:,0:9])
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
X = company.drop('Sales', axis=1)
y=company.iloc[:,10]
Y=pd.concat([y,X],axis=1)
##X = pd.get_dummies(X, prefix_sep='_')
Y['ShelveLoc']=LabelEncoder().fit_transform(company['ShelveLoc'])
Y['Urban']=LabelEncoder().fit_transform(company['Urban'])
Y['US']=LabelEncoder().fit_transform(company['US'])
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 400)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=400).fit(X_Train,Y_Train)
    print(time.process_time() - start) 
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
labels=['Normal','Excellent']
bins=[0,10,16.27]
company['Sales']=pd.cut(company['Sales'],bins=bins,labels=labels)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, company['Sales']], axis = 1)
PCA_df['Sales'] = LabelEncoder().fit_transform(PCA_df['Sales'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
Sales = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for Sales, color in zip(Sales, colors):
    plt.scatter(PCA_df.loc[PCA_df['Sales'] == Sales, 'PC1'], 
                PCA_df.loc[PCA_df['Sales'] == Sales, 'PC2'], 
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
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 400)
trainedforest = RandomForestClassifier(n_estimators=400).fit(X_Reduced,Y_Reduced)
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
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.30,random_state = 300) 
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
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
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
X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=400)
autoencoder.fit(X1, Y1,epochs=400,batch_size=400,shuffle=True,verbose = 300,validation_data=(X2, Y2))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)

