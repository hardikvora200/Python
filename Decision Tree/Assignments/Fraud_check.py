import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# reading a csv file using pandas library
fraud=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Decision Tree\\Assignments\\Fraud_check.csv")
##fraud.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
fraud.columns
##fraud.drop(["a"],axis=1,inplace=True)
##fraud.columns
# To get the count of null values in the data 
fraud.isnull().sum()
fraud.shape # 600 6 => Before dropping null values
# To drop null values ( dropping rows)
fraud.dropna().shape # 400 11 => After dropping null values
#####Exploratory Data Analysis#########################################################
fraud.mean() ### Taxable.Income - 55208.375000,City.Population-108747.368333,Work.Experience- 15.558333
fraud.median() 
fraud.mode() 
####Measures of Dispersion
fraud.var() 
fraud.std() ### Taxable.Income - 26204.827597,City.Population-49850.075134,Work.Experience- 8.842147
#### Calculate the range value
range1 = max(fraud['Taxable.Income'])-min(fraud['Taxable.Income'])  ### 89616
range2 = max(fraud['City.Population'])-min(fraud['City.Population']) ### 173999
range3 = max(fraud['Work.Experience'])-min(fraud['Work.Experience']) ### 30
### Calculate skewness and Kurtosis
fraud.skew()
fraud.kurt() 
####Graphidelivery_time Representation 
plt.hist(fraud["Taxable.Income"])
plt.hist(fraud["City.Population"])
plt.hist(fraud["Work.Experience"])
plt.boxplot(fraud["Taxable.Income"],0,"rs",0)
plt.boxplot(fraud["City.Population"],0,"rs",0)
plt.boxplot(fraud["Work.Experience"],0,"rs",0)
plt.plot(fraud["City.Population"],fraud["Taxable.Income"],"bo");plt.xlabel("City.Population");plt.ylabel("Taxable.Income")
plt.plot(fraud["Work.Experience"],fraud["Taxable.Income"],"bo");plt.xlabel("Work.Experience");plt.ylabel("Taxable.Income")
plt.plot(fraud["City.Population"],fraud["Work.Experience"],"bo");plt.xlabel("City.Population");plt.ylabel("Work.Experience")
# table 
pd.crosstab(fraud["City.Population"],fraud["Taxable.Income"])
pd.crosstab(fraud["Work.Experience"],fraud["Taxable.Income"])
pd.crosstab(fraud["City.Population"],fraud["Work.Experience"])
## Barplot
pd.crosstab(fraud["City.Population"],fraud["Taxable.Income"]).plot(kind = "bar", width = 1.85)
pd.crosstab(fraud["Work.Experience"],fraud["Taxable.Income"]).plot(kind = "bar", width = 1.85)
pd.crosstab(fraud["City.Population"],fraud["Work.Experience"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="City.Population",data=fraud,palette="hls")
sns.countplot(x="Work.Experience",data=fraud,palette="hls")
sns.countplot(x="Taxable.Income",data=fraud,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="City.Population",y="Taxable.Income",data=fraud,palette="hls")
sns.boxplot(x="Work.Experience",y="Taxable.Income",data=fraud,palette="hls")
sns.boxplot(x="City.Population",y="Work.Experience",data=fraud,palette="hls")
sns.boxplot(x="Work.Experience",y="City.Population",data=fraud,palette="hls")
sns.pairplot(fraud.iloc[:,0:7]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(fraud,hue="Taxable.Income",size=2)
fraud["City.Population"].value_counts()
fraud["Work.Experience"].value_counts()
fraud["Taxable.Income"].value_counts()
fraud["City.Population"].value_counts().plot(kind = "pie")
fraud["Work.Experience"].value_counts().plot(kind = "pie")
fraud["Taxable.Income"].value_counts().plot(kind = "pie")
sns.pairplot(fraud,hue="City.Population",size=4,diag_kind = "kde")
sns.pairplot(fraud,hue="Work.Experience",size=4,diag_kind = "kde")
sns.pairplot(fraud,hue="Taxable.Income",size=4,diag_kind = "kde")
sns.FacetGrid(fraud,hue="Taxable.Income").map(plt.scatter,"City.Population","Taxable.Income").add_legend()
sns.FacetGrid(fraud,hue="Taxable.Income").map(plt.scatter,"Work.Experience","Taxable.Income").add_legend()
sns.FacetGrid(fraud,hue="Work.Experience").map(plt.scatter,"City.Population","Work.Experience").add_legend()
sns.FacetGrid(fraud,hue="Taxable.Income").map(sns.kdeplot,"City.Population").add_legend()
sns.FacetGrid(fraud,hue="Taxable.Income").map(sns.kdeplot,"Work.Experience").add_legend()
sns.FacetGrid(fraud,hue="Work.Experience").map(sns.kdeplot,"City.Population").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab
# Checking Whether data is normally distributed
stats.probplot(fraud['Taxable.Income'],dist="norm",plot=pylab)
stats.probplot(np.log(fraud['Taxable.Income']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(fraud['Taxable.Income']),dist="norm",plot=pylab)
stats.probplot((fraud['Taxable.Income'] * fraud['Taxable.Income']),dist="norm",plot=pylab)
stats.probplot(np.exp(fraud['Taxable.Income']),dist="norm",plot=pylab)
stats.probplot(np.exp(fraud['Taxable.Income'])*np.exp(fraud['Taxable.Income']),dist="norm",plot=pylab)
reci_1=1/fraud['Taxable.Income']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((fraud['Taxable.Income'] * fraud['Taxable.Income'])+fraud['Taxable.Income']),dist="norm",plot=pylab)
stats.probplot(fraud['Work.Experience'],dist="norm",plot=pylab)
stats.probplot(np.log(fraud['Work.Experience']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(fraud['Work.Experience']),dist="norm",plot=pylab)
stats.probplot((fraud['Work.Experience'] * fraud['Work.Experience']),dist="norm",plot=pylab)
stats.probplot(np.exp(fraud['Work.Experience']),dist="norm",plot=pylab)
stats.probplot(np.exp(fraud['Work.Experience'])*np.exp(fraud['Work.Experience']),dist="norm",plot=pylab)
reci_2=1/fraud['Work.Experience']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((fraud['Work.Experience'] * fraud['Work.Experience'])+fraud['Work.Experience']),dist="norm",plot=pylab)
stats.probplot(fraud['City.Population'],dist="norm",plot=pylab)
stats.probplot(np.log(fraud['City.Population']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(fraud['City.Population']),dist="norm",plot=pylab)
stats.probplot((fraud['City.Population'] * fraud['City.Population']),dist="norm",plot=pylab)
stats.probplot(np.exp(fraud['City.Population']),dist="norm",plot=pylab)
stats.probplot(np.exp(fraud['City.Population'])*np.exp(fraud['City.Population']),dist="norm",plot=pylab)
reci_3=1/fraud['City.Population']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((fraud['City.Population'] * fraud['City.Population'])+fraud['City.Population']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Taxable.Income
stats.norm.ppf(0.975,55208.375000,26204.827597)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(fraud["Taxable.Income"], 55208.375000,26204.827597) # similar to pnorm in R 
#### City.Population
stats.norm.ppf(0.975,108747.368333,49850.075134)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(fraud["City.Population"],108747.368333,49850.075134) # similar to pnorm in R 
#### Work.Experience
stats.norm.ppf(0.975,15.558333,8.842147)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(fraud["Work.Experience"],15.558333,8.842147) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
fraud.corr(method = "pearson")
fraud.corr(method = "kendall")
# to get top 6 rows
fraud.head(40) # to get top n rows use cars.head(10)
fraud.tail(10)
### Normalization
##def norm_func(i):
  ##  x = (i-i.mean())/(i.std())
 ##   return (x)
### Normalized data frame (considering the numerical part of data)
##df_norm = norm_func(fraud.iloc[:,0:])
# Scatter plot between the variables along with histograms
sns.pairplot(fraud)
fraud['Taxable.Income'].unique()
labels=['good','risky']
bins=[10000,30000,99619]
fraud['Taxable.Income']=pd.cut(fraud['Taxable.Income'],bins=bins,labels=labels)
fraud.head()
y=fraud.iloc[:,5]
fraud1= fraud.drop('Taxable.Income', axis=1)
fraud2=pd.concat([y,fraud1],axis=1)
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
fraud2['Undergrad']=lb_make.fit_transform(fraud['Undergrad'])
fraud2['Marital.Status']=lb_make.fit_transform(fraud['Marital.Status'])
fraud2['Urban']=lb_make.fit_transform(fraud['Urban'])
colnames=list(fraud2.columns)
predx= colnames[1:6]
predy=colnames[0]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
train,test=train_test_split(fraud2,test_size=0.4)
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train[predx],train[predy])
preds=model.predict(test[predx])
pd.Series(preds).value_counts()
pd.crosstab(test[predy],preds)
np.mean(train['Taxable.Income'] == model.predict(train[predx])) #100% accuracy
np.mean(preds==test['Taxable.Income'])#61.67% accuracy
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
np.mean(train['Taxable.Income'] == model_1.predict(train[predx])) 
np.mean(target_pred==test['Taxable.Income']) 
## Lets check with regression metrics
from xgboost import XGBRegressor
model_2=XGBRegressor()
model_2.fit(train[predx],train[predy])
target_pred_2=model_2.predict(test[predx])
predictions_2 = [round(value) for value in target_pred_2]
np.mean(train['Taxable.Income'] == model_2.predict(train[predx])) 
np.mean(target_pred_2==test['Taxable.Income'])
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
array=fraud.values
fraud_dummies=pd.get_dummies(fraud[["Undergrad","Marital.Status","Urban"]])
fraud = pd.concat([fraud,fraud_dummies],axis=1)
fraud['Taxable.Income'].value_counts()
colnames=list(fraud.columns)
fraud.drop(["Undergrad"],inplace=True,axis = 1)
fraud.drop(["Marital.Status"],inplace=True,axis = 1)
fraud.drop(["Urban"],inplace=True,axis = 1)
X1=fraud.iloc[:,0:2]
X2=fraud.iloc[:,3:10]
X=pd.concat([X1,X2],axis=1)
X=np.reshape(-1,1)
Y=fraud.iloc[:,2]
##X=norm_func(X)
X=X.astype('int')
Y=Y.astype('int')
seed = 7
kfold=model_selection.KFold(n_splits=600,random_state=seed)
model=DecisionTreeRegressor()
num_trees=50
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
num_trees=100
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
fraud = fraud[fraud.columns[fraud.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
fraud = fraud.loc[fraud.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
fraud = fraud.fillna(fraud.median())
fraud['Undergrad'].fillna(fraud['Undergrad'].value_counts().idxmax(), inplace=True)
fraud['Marital.Status'].fillna(fraud['Marital.Status'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##fraud['column_name'].fillna(fraud['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = fraud['City.Population'].mean () + fraud['City.Population'].std () * factor   
lower_lim1= fraud['City.Population'].mean () - fraud['City.Population'].std () * factor 
fraud1 = fraud[(fraud['City.Population'] < upper_lim1) & (fraud['City.Population'] > lower_lim1)]
upper_lim2 = fraud['Work.Experience'].mean () + fraud['Work.Experience'].std () * factor  
lower_lim2 = fraud['Work.Experience'].mean () - fraud['Work.Experience'].std () * factor  
fraud2 = fraud[(fraud['Work.Experience'] < upper_lim2) & (fraud['Work.Experience'] > lower_lim2)]
upper_lim3 = fraud['Taxable.Income'].mean () + fraud['Taxable.Income'].std () * factor  
lower_lim3 = fraud['Taxable.Income'].mean () - fraud['Taxable.Income'].std () * factor 
fraud3 = fraud[(fraud['Taxable.Income'] < upper_lim3) & (fraud['Taxable.Income'] > lower_lim3)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim4 = fraud['City.Population'].quantile(.95)
lower_lim4 = fraud['City.Population'].quantile(.05)
fraud4 = fraud[(fraud['City.Population'] < upper_lim4) & (fraud['City.Population'] > lower_lim4)]
upper_lim5 = fraud['Work.Experience'].quantile(.95)
lower_lim5 = fraud['Work.Experience'].quantile(.05)
fraud5 = fraud[(fraud['Work.Experience'] < upper_lim5) & (fraud['Work.Experience'] > lower_lim5)]
upper_lim6 = fraud['Taxable.Income'].quantile(.95)
lower_lim6 = fraud['Taxable.Income'].quantile(.05)
fraud6 = fraud[(fraud['Taxable.Income'] < upper_lim6) & (fraud['Taxable.Income'] > lower_lim6)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
fraud.loc[(fraud['City.Population'] > upper_lim4)] = upper_lim4
fraud.loc[(fraud['City.Population'] < lower_lim4)] = lower_lim4
fraud.loc[(fraud['Work.Experience'] > upper_lim5)] = upper_lim5
fraud.loc[(fraud['Work.Experience'] < lower_lim5)] = lower_lim5
fraud.loc[(fraud['Taxable.Income'] > upper_lim6)] = upper_lim6
fraud.loc[(fraud['Taxable.Income'] < lower_lim6)] = lower_lim6
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
fraud['bin1'] = pd.cut(fraud['City.Population'], bins=[25779,50000,199778], labels=["Good","Risky"])
fraud['bin2'] = pd.cut(fraud['Work.Experience'], bins=[0,10,30], labels=["Low","Good"])
fraud['bin3'] = pd.cut(fraud['Taxable.Income'], bins=[10000,30000,99619], labels=["Good","Risky"])
conditions = [
    fraud['Undergrad'].str.contains('NO'),
    fraud['Undergrad'].str.contains('YES')]
choices=['1','2']
fraud['choices']=np.select(conditions,choices,default='Other')
conditions1 = [
    fraud['Marital.Status'].str.contains('Single'),
    fraud['Marital.Status'].str.contains('Divorced'),
    fraud['Marital.Status'].str.contains('Married')]
choices1= ['1','2','3']
fraud['choices1']=np.select(conditions1,choices1,default='Other')
conditions2 = [
    fraud['Work.Experience'].str.contains('NO'),
    fraud['Work.Experience'].str.contains('YES')]   
choices2= ['1','2']
fraud['choices2']=np.select(conditions2,choices2,default='Other')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
fraud = pd.DataFrame({'City.Population':fraud.iloc[:,2]})
fraud['log+1'] = (fraud['City.Population']+1).transform(np.log)
#Negative Values Handling
fraud['log'] = (fraud['City.Population']-fraud['City.Population'].min()+1).transform(np.log)
fraud = pd.DataFrame({'Work.Experience':fraud.iloc[:,3]})
fraud['log+1'] = (fraud['Work.Experience']+1).transform(np.log)
#Negative Values Handling
fraud['log'] = (fraud['Work.Experience']-fraud['Work.Experience'].min()+1).transform(np.log)
fraud = pd.DataFrame({'Taxable.Income':fraud.iloc[:,5]})
fraud['log+1'] = (fraud['Taxable.Income']+1).transform(np.log)
#Negative Values Handling
fraud['log'] = (fraud['Taxable.Income']-fraud['Taxable.Income'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(fraud['Undergrad'])
fraud = fraud.join(encoded_columns.add_suffix('_Undergrad')).drop('Undergrad', axis=1) 
encoded_columns_1 = pd.get_dummies(fraud['Marital.Status'])
fraud = fraud.join(encoded_columns_1.add_suffix('_Marital.Status')).drop('Marital.Status', axis=1)    
encoded_columns_2 = pd.get_dummies(fraud['Urban'])
fraud = fraud.join(encoded_columns_2.add_suffix('_Urban')).drop('Urban', axis=1)                                  
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = fraud.groupby('Taxable.Income')
sums = grouped['frauds'].sum().add_suffix('_sum')
avgs = grouped['frauds'].mean().add_suffix('_avg')
####Categorical Column grouping
fraud.groupby('Taxable.Income').agg(lambda x: x.value_counts().index[0])
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(fraud.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(fraud.iloc[:,0:9])
##### Feature Extraction
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X = fraud.drop('taxincome', axis=1)
y=y=fraud.iloc[:,2]
Y=pd.concat([y,X],axis=1)
##X = pd.get_dummies(X, prefix_sep='_')
Y['undergrad']=LabelEncoder().fit_transform(fraud['undergrad'])
Y['maritalstat']=LabelEncoder().fit_transform(fraud['maritalstat'])
Y['urban']=LabelEncoder().fit_transform(fraud['urban'])
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
labels=['risky','good']
bins=[10000,30000,99619]
fraud['Taxable.Income']=pd.cut(fraud['Taxable.Income'],bins=bins,labels=labels)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, fraud['Taxable.Income']], axis = 1)
PCA_df['Taxable.Income'] = LabelEncoder().fit_transform(PCA_df['Taxable.Income'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
y = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for y, color in zip(y,colors):
    plt.scatter(PCA_df.loc[PCA_df['y'] == y, 'PC1'], 
                PCA_df.loc[PCA_df['y'] == y, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 20)
plt.legend(['Good', 'Risky'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
###Additionally, using our two-dimensional dataset, we can now also visualize the decision boundary used by our Random Forest in order to classify each of the different data points.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 600)
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

