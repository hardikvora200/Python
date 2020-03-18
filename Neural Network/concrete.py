import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
# reading a csv file using pandas library
concrete=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Neural Network\\Assignments\\concrete.csv")
concrete.columns
#####Exploratory Data Analysis#########################################################
concrete.mean() ## cement - 281.167864, slag - 73.895825, ash - 54.188350, water - 181.567282, superplastic-6.204660,coarseagg-972.918932, fineagg-773.580485,age-45.662136,strength-35.817961
concrete.median() ### cement - 272.9, slag - 22.00, ash - 0.000, water - 185.000, superplastic-6.4,coarseagg-968.00, fineagg-779.5,age-28.00,strength-34.445
concrete.mode() 
####Measures of Dispersion
concrete.var() 
concrete.std() ##cement - 10921.580220,slag - 7444.124812,ash-4095.616541,water-456.002651, superplastic-35.686781,coarseagg-6045.677357,fineagg-6428.187792,age-3990.437729,strength-279.081814
#### Calculate the range value
range1 = max(concrete['cement'])-min(concrete['cement'])  ### 438
range2 = max(concrete['slag'])-min(concrete['slag']) ### 359.4
range3 = max(concrete['ash'])-min(concrete['ash']) ###  200.1
range4 = max(concrete['water'])-min(concrete['water']) ### 125.2
range5 = max(concrete['superplastic'])-min(concrete['superplastic']) ### 32.2
range6 = max(concrete['coarseagg'])-min(concrete['coarseagg']) ### 344.0
range7 = max(concrete['fineagg'])-min(concrete['fineagg']) ### 398.6
range8 = max(concrete['age'])-min(concrete['age'])   ###364
range9 = max(concrete['strength'])-min(concrete['strength'])  ### 80.27
### Calculate skewness and Kurtosis
concrete.skew() ## cement - 0.509481, slag - 0.800717, ash - 0.537354, water - 0.074628, superplastic-0.907203,coarseagg- -0.040220, fineagg- -0.253010,age-3.269177,strength-0.416977
concrete.kurt() ## cement - -0.520652, slag - -0.508175, ash - -1.328746, water - 0.122082, superplastic-1.411269,coarseagg- -0.599016, fineagg- -0.102177,age-12.168989,strength- -0.313725
####Graphidelivery_time Representation 
plt.hist(concrete["cement"])
plt.hist(concrete["slag"])
plt.hist(concrete["ash"])
plt.hist(concrete["water"])
plt.hist(concrete["superplastic"])
plt.hist(concrete["coarseagg"])
plt.hist(concrete["fineagg"])
plt.hist(concrete["age"])
plt.hist(concrete["strength"])
plt.boxplot(concrete["cement"],0,"rs",0)
plt.boxplot(concrete["slag"],0,"rs",0)
plt.boxplot(concrete["ash"],0,"rs",0)
plt.boxplot(concrete["water"],0,"rs",0)
plt.boxplot(concrete["superplastic"],0,"rs",0)
plt.boxplot(concrete["coarseagg"],0,"rs",0)
plt.boxplot(concrete["fineagg"],0,"rs",0)
plt.boxplot(concrete["age"],0,"rs",0)
plt.boxplot(concrete["strength"],0,"rs",0)
plt.plot(concrete["cement"],concrete["strength"],"bo");plt.xlabel("cement");plt.ylabel("strength")
plt.plot(concrete["slag"],concrete["strength"],"bo");plt.xlabel("slag");plt.ylabel("strength")
plt.plot(concrete["ash"],concrete["strength"],"bo");plt.xlabel("ash");plt.ylabel("strength")
plt.plot(concrete["water"],concrete["strength"],"bo");plt.xlabel("water");plt.ylabel("strength")
plt.plot(concrete["superplastic"],concrete["strength"],"bo");plt.xlabel("superplastic");plt.ylabel("strength")
plt.plot(concrete["coarseagg"],concrete["strength"],"bo");plt.xlabel("coarseagg");plt.ylabel("strength")
plt.plot(concrete["fineagg"],concrete["strength"],"bo");plt.xlabel("fineagg");plt.ylabel("strength")
plt.plot(concrete["age"],concrete["strength"],"bo");plt.xlabel("age");plt.ylabel("strength")
# table 
pd.crosstab(concrete["cement"],concrete["strength"])
pd.crosstab(concrete["slag"],concrete["strength"])
pd.crosstab(concrete["ash"],concrete["strength"])
pd.crosstab(concrete["water"],concrete["strength"])
pd.crosstab(concrete["superplastic"],concrete["strength"])
pd.crosstab(concrete["coarseagg"],concrete["strength"])
pd.crosstab(concrete["fineagg"],concrete["strength"])
pd.crosstab(concrete["age"],concrete["strength"])
## Barplot
pd.crosstab(concrete["cement"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["slag"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["ash"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["water"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["superplastic"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["coarseagg"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["fineagg"],concrete["strength"]).plot(kind = "bar", width = 1.85)
pd.crosstab(concrete["age"],concrete["strength"]).plot(kind = "bar", width = 1.85)
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="cement",y="strength",data=concrete)
sns.boxplot(x="slag",y="strength",data=concrete)
sns.boxplot(x="ash",y="strength",data=concrete)
sns.boxplot(x="water",y="strength",data=concrete)
sns.boxplot(x="superplastic",y="strength",data=concrete)
sns.boxplot(x="coarseagg",y="strength",data=concrete)
sns.boxplot(x="fineagg",y="strength",data=concrete)
sns.boxplot(x="strength",y="strength",data=concrete)
sns.boxplot(x="age",y="strength",data=concrete)
sns.pairplot(concrete.iloc[:,0:8]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(concrete,hue="strength",size=5)
concrete["cement"].value_counts()
concrete["slag"].value_counts()
concrete["ash"].value_counts()
concrete["water"].value_counts()
concrete["superplastic"].value_counts()
concrete["coarseagg"].value_counts()
concrete["fineagg"].value_counts()
concrete["age"].value_counts()
concrete["strength"].value_counts()
concrete["cement"].value_counts().plot(kind = "pie")
concrete["slag"].value_counts().plot(kind = "pie")
concrete["ash"].value_counts().plot(kind = "pie")
concrete["water"].value_counts().plot(kind = "pie")
concrete["superplastic"].value_counts().plot(kind = "pie")
concrete["coarseagg"].value_counts().plot(kind = "pie")
concrete["fineagg"].value_counts().plot(kind = "pie")
concrete["age"].value_counts().plot(kind = "pie")
concrete["strength"].value_counts().plot(kind = "pie")
sns.pairplot(concrete,hue="strength",size=4,diag_kind = "kde")
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"cement","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"slag","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"ash","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"water","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"superplastic","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"coarseagg","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"fineagg","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(plt.scatter,"age","strength").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"cement").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"slag").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"ash").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"water").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"superplastic").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"coarseagg").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"fineagg").add_legend()
sns.FacetGrid(concrete,hue="strength").map(sns.kdeplot,"age").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(concrete['cement'], dist="norm",plot=pylab)
stats.probplot(concrete['slag'],dist="norm",plot=pylab)
stats.probplot(concrete['ash'],dist="norm",plot=pylab)
stats.probplot(concrete['water'],dist="norm",plot=pylab)
stats.probplot(concrete['superplastic'], dist="norm",plot=pylab)
stats.probplot(concrete['coarseagg'],dist="norm",plot=pylab)
stats.probplot(concrete['fineagg'],dist="norm",plot=pylab)
stats.probplot(concrete['age'],dist="norm",plot=pylab)
# ppf => Percent point function 
#### cement
stats.norm.ppf(0.975,281.167864,10921.580220)# similar to qnorm in R ---- 21687.071749465038
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["cement"],281.167864,10921.580220) # similar to pnorm in R 
#### slag
stats.norm.ppf(0.975,73.895825,7444.124812)# similar to qnorm in R ---- 14664.112352941
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["slag"],73.895825,7444.124812) # similar to pnorm in R 
#### ash
stats.norm.ppf(0.975,54.188350,4095.616541)# similar to qnorm in R ---- 8081.449264846514
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["ash"],54.188350,4095.616541) # similar to pnorm in R 
#### water
stats.norm.ppf(0.975, 181.567282, 456.002651)# similar to qnorm in R ---- 1075.3160548147878
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["water"], 181.567282, 456.002651) # similar to pnorm in R 
#### superplastic
stats.norm.ppf(0.975,6.204660,35.686781)# similar to qnorm in R ---- 76.1494654841683
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["superplastic"],6.204660,35.686781) # similar to pnorm in R 
#### coarseagg
stats.norm.ppf(0.975,972.918932,6045.677357)# similar to qnorm in R ---- 12822.228813869302
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["coarseagg"],972.918932,6045.677357) # similar to pnorm in R 
#### fineagg
stats.norm.ppf(0.975,773.580485,6428.187792)# similar to qnorm in R ---- 13372.59704
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["fineagg"],773.580485,6428.187792) # similar to pnorm in R 
#### age
stats.norm.ppf(0.975, 45.662136, 3990.437729)# similar to qnorm in R ---- 7866.776367389804
# cdf => cumulative distributive function 
stats.norm.cdf(concrete["age"], 45.662136, 3990.437729) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
concrete.corr(method = "pearson")
concrete.corr(method = "kendall")
concrete["strength"].corr(concrete["cement"]) # # correlation value between X and Y -- 0.497
np.corrcoef(concrete["cement"],concrete["strength"])
np.corrcoef(concrete["slag"],concrete["strength"])
np.corrcoef(concrete["ash"],concrete["strength"])
np.corrcoef(concrete["water"],concrete["strength"])
np.corrcoef(concrete["superplastic"],concrete["strength"])
np.corrcoef(concrete["coarseagg"],concrete["strength"])
np.corrcoef(concrete["fineagg"],concrete["strength"])
np.corrcoef(concrete["age"],concrete["strength"])
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(concrete['cement'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(concrete['slag'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(concrete['ash'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(concrete['water'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(concrete['superplastic'])
normalized_E = preprocessing.normalize([e_array])
f_array = np.array(concrete['coarseagg'])
normalized_F = preprocessing.normalize([f_array])
g_array = np.array(concrete['fineagg'])
normalized_G = preprocessing.normalize([g_array])
h_array = np.array(concrete['age'])
normalized_H = preprocessing.normalize([h_array])
# to get top 6 rows
concrete.head(40) # to get top n rows use cars.head(10)
concrete.tail(10)
# Correlation matrix 
concrete.corr()
# We see there exists High collinearity between input variables especially between
# [Profit & R&D Spend] , [Profit & Marketing Spend],[R&D Spend & Marketing Spend]
## so there exists collinearity problem
# Scatter plot between the variables along with histograms
sns.set(style="ticks")
sns.pairplot(concrete,hue="strength")
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
concrete = concrete[concrete.columns[concrete.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
concrete = concrete.loc[concrete.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
concrete = concrete.fillna(concrete.median())
##concrete['State'].fillna(concrete['State'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##concrete['column_name'].fillna(concrete['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = concrete['cement'].mean () + concrete['cement'].std () * factor 
lower_lim1= concrete['cement'].mean () - concrete['cement'].std () * factor 
concrete = concrete[(concrete['cement'] < upper_lim1) & (concrete['cement'] > lower_lim1)]
upper_lim2 = concrete['slag'].mean () + concrete['slag'].std () * factor  
lower_lim2 = concrete['slag'].mean () - concrete['slag'].std () * factor  
concrete = concrete[(concrete['slag'] < upper_lim2) & (concrete['slag'] > lower_lim2)]
upper_lim3 = concrete['ash'].mean () + concrete['ash'].std () * factor  
lower_lim3 = concrete['ash'].mean () - concrete['ash'].std () * factor  
concrete = concrete[(concrete['ash'] < upper_lim3) & (concrete['ash'] > lower_lim3)]
upper_lim4 = concrete['water'].mean () + concrete['water'].std () * factor  
lower_lim4 = concrete['water'].mean () - concrete['water'].std () * factor  
concrete = concrete[(concrete['water'] < upper_lim4) & (concrete['water'] > lower_lim4)]
upper_lim5 = concrete['coarseagg'].mean () + concrete['coarseagg'].std () * factor  
lower_lim5 = concrete['coarseagg'].mean () - concrete['coarseagg'].std () * factor  
concrete = concrete[(concrete['coarseagg'] < upper_lim5) & (concrete['coarseagg'] > lower_lim5)]
upper_lim6 = concrete['fineagg'].mean () + concrete['fineagg'].std () * factor  
lower_lim6 = concrete['fineagg'].mean () - concrete['fineagg'].std () * factor  
concrete = concrete[(concrete['fineagg'] < upper_lim6) & (concrete['fineagg'] > lower_lim6)]
upper_lim7 = concrete['age'].mean () + concrete['age'].std () * factor  
lower_lim7 = concrete['age'].mean () - concrete['age'].std () * factor  
concrete = concrete[(concrete['age'] < upper_lim7) & (concrete['age'] > lower_lim7)]
upper_lim8 = concrete['superplastic'].mean () + concrete['superplastic'].std () * factor  
lower_lim8 = concrete['superplastic'].mean () - concrete['superplastic'].std () * factor  
concrete = concrete[(concrete['superplastic'] < upper_lim8) & (concrete['superplastic'] > lower_lim8)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim9 = concrete['cement'].quantile(.95)
lower_lim9 = concrete['cement'].quantile(.05)
concrete = concrete[(concrete['cement'] < upper_lim9) & (concrete['cement'] > lower_lim9)]
upper_lim10 = concrete['slag'].quantile(.95)
lower_lim10 = concrete['slag'].quantile(.05)
concrete = concrete[(concrete['slag'] < upper_lim10) & (concrete['slag'] > lower_lim10)]
upper_lim11 = concrete['ash'].quantile(.95)
lower_lim11 = concrete['ash'].quantile(.05)
concrete = concrete[(concrete['ash'] < upper_lim11) & (concrete['ash'] > lower_lim11)]
upper_lim12 = concrete['water'].quantile(.95)
lower_lim12 = concrete['water'].quantile(.05)
concrete = concrete[(concrete['water'] < upper_lim12) & (concrete['water'] > lower_lim12)]
upper_lim13 = concrete['coarseagg'].quantile(.95)
lower_lim13 = concrete['coarseagg'].quantile(.05)
concrete = concrete[(concrete['coarseagg'] < upper_lim13) & (concrete['coarseagg'] > lower_lim13)]
upper_lim14 = concrete['fineagg'].quantile(.95)
lower_lim14 = concrete['fineagg'].quantile(.05)
concrete = concrete[(concrete['fineagg'] < upper_lim14) & (concrete['fineagg'] > lower_lim14)]
upper_lim15 = concrete['age'].quantile(.95)
lower_lim15 = concrete['age'].quantile(.05)
concrete = concrete[(concrete['age'] < upper_lim15) & (concrete['age'] > lower_lim15)]
upper_lim16 = concrete['superplastic'].quantile(.95)
lower_lim16 = concrete['superplastic'].quantile(.05)
concrete = concrete[(concrete['superplastic'] < upper_lim16) & (concrete['superplastic'] > lower_lim16)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
upper_lim = concrete['cement'].quantile(.95)
lower_lim = concrete['cement'].quantile(.05)
concrete.loc[(concrete['cement'] > upper_lim)] = upper_lim
concrete.loc[(concrete['cement'] < lower_lim)] = lower_lim
upper_lim = concrete['slag'].quantile(.95)
lower_lim = concrete['slag'].quantile(.05)
concrete.loc[(concrete['slag'] > upper_lim)] = upper_lim
concrete.loc[(concrete['slag'] < lower_lim)] = lower_lim
upper_lim = concrete['ash'].quantile(.95)
lower_lim = concrete['ash'].quantile(.05)
concrete.loc[(concrete['ash'] > upper_lim)] = upper_lim
concrete.loc[(concrete['ash'] < lower_lim)] = lower_lim
upper_lim = concrete['water'].quantile(.95)
lower_lim = concrete['water'].quantile(.05)
concrete.loc[(concrete['water'] > upper_lim)] = upper_lim
concrete.loc[(concrete['water'] < lower_lim)] = lower_lim
upper_lim = concrete['coarseagg'].quantile(.95)
lower_lim = concrete['coarseagg'].quantile(.05)
concrete.loc[(concrete['coarseagg'] > upper_lim)] = upper_lim
concrete.loc[(concrete['coarseagg'] < lower_lim)] = lower_lim
upper_lim = concrete['fineagg'].quantile(.95)
lower_lim = concrete['fineagg'].quantile(.05)
concrete.loc[(concrete['fineagg'] > upper_lim)] = upper_lim
concrete.loc[(concrete['fineagg'] < lower_lim)] = lower_lim
upper_lim = concrete['age'].quantile(.95)
lower_lim = concrete['age'].quantile(.05)
concrete.loc[(concrete['age'] > upper_lim)] = upper_lim
concrete.loc[(concrete['age'] < lower_lim)] = lower_lim
upper_lim = concrete['superplastic'].quantile(.95)
lower_lim = concrete['superplastic'].quantile(.05)
concrete.loc[(concrete['superplastic'] > upper_lim)] = upper_lim
concrete.loc[(concrete['superplastic'] < lower_lim)] = lower_lim
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
concrete['bin1'] = pd.cut(concrete['cement'], bins=[102,300,540], labels=["Low", "High"])
concrete['bin2'] = pd.cut(concrete['slag'], bins=[0,100,359.4], labels=["Low", "High"])
concrete['bin3'] = pd.cut(concrete['ash'], bins=[0,100,200.1], labels=["Low","High"])
concrete['bin4'] = pd.cut(concrete['water'],bins=[121.8,175,247],labels=["Low","High"])
concrete['bin5'] = pd.cut(concrete['coarseagg'], bins=[801,1000,1145], labels=["Low", "High"])
concrete['bin6'] = pd.cut(concrete['fineagg'], bins=[594,750,992.6], labels=["Low", "High"])
concrete['bin7'] = pd.cut(concrete['age'], bins=[1,200,365], labels=["Low","High"])
concrete['bin8'] = pd.cut(concrete['superplastic'],bins=[0,17,32.2],labels=["Low","High"])
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
concrete = pd.DataFrame({'cement':concrete.iloc[:,0]})
concrete['log+1'] = (concrete['cement']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['cement']-concrete['cement'].min()+1).transform(np.log)
concrete = pd.DataFrame({'slag':concrete.iloc[:,1]})
concrete['log+1'] = (concrete['slag']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['slag']-concrete['slag'].min()+1).transform(np.log)
concrete = pd.DataFrame({'ash':concrete.iloc[:,2]})
concrete['log+1'] = (concrete['ash']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['ash']-concrete['ash'].min()+1).transform(np.log)
concrete = pd.DataFrame({'water':concrete.iloc[:,3]})
concrete['log+1'] = (concrete['water']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['water']-concrete['water'].min()+1).transform(np.log)
concrete = pd.DataFrame({'superplastic':concrete.iloc[:,4]})
concrete['log+1'] = (concrete['superplastic']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['superplastic']-concrete['superplastic'].min()+1).transform(np.log)
concrete = pd.DataFrame({'coarseagg':concrete.iloc[:,5]})
concrete['log+1'] = (concrete['coarseagg']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['coarseagg']-concrete['coarseagg'].min()+1).transform(np.log)
concrete = pd.DataFrame({'fineagg':concrete.iloc[:,6]})
concrete['log+1'] = (concrete['fineagg']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['fineagg']-concrete['fineagg'].min()+1).transform(np.log)
concrete = pd.DataFrame({'age':concrete.iloc[:,7]})
concrete['log+1'] = (concrete['age']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['age']-concrete['age'].min()+1).transform(np.log)
concrete = pd.DataFrame({'strength':concrete.iloc[:,8]})
concrete['log+1'] = (concrete['strength']+1).transform(np.log)
#Negative Values Handling
concrete['log'] = (concrete['strength']-concrete['strength'].min()+1).transform(np.log)
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = concrete.groupby('strength')
sums = grouped['strength'].sum().add_suffix('_sum')
avgs = grouped['strength'].mean().add_suffix('_avg')
####Categorical Column grouping
concrete.groupby('strength').agg(lambda x: x.value_counts().index[0])
#Pivot table Pandas Example
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(concrete.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(concrete.iloc[:,0:9])
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
X = concrete.iloc[:,0:8]
Y = concrete['strength']
X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 1030)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=1030).fit(X_Train,Y_Train)
    print(time.process_time() - start) ####2.106013499999989
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, concrete['strength']], axis = 1)
PCA_df['strength'] = LabelEncoder().fit_transform(PCA_df['strength'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
strength = concrete["strength"]
colors = ['r', 'b']
for strength, color in zip(strength, colors):
    plt.scatter(PCA_df.loc[PCA_df['strength'] == strength, 'PC1'], 
                PCA_df.loc[PCA_df['strength'] == strength, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 15)
plt.legend(['Less', 'More'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
###Additionally, using our two-dimensional dataset, we can now also visualize the decision boundary used by our Random Forest in order to classify each of the different data points.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 101)
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Reduced,Y_Reduced)
x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('Random Forest', fontsize = 15)
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
X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=50)
autoencoder.fit(X1, Y1,epochs=50,batch_size=50,shuffle=True,verbose = 30,validation_data=(X2, Y2))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)
################Lets prepare Neural Network model
from keras.models import Sequential
##from keras.layers import Dense, Activation,Layer,Lambda
from sklearn.cross_validation import train_test_split
def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

column_names = list(concrete.columns)
predictors = column_names[0:8]
target = column_names[8]
first_model = prep_model([8,50,1])
first_model.fit(np.array(concrete[predictors]),np.array(concrete[target]),epochs=900)
pred_train = first_model.predict(np.array(concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,concrete[target],"bo")
np.corrcoef(pred_train,concrete[target]) 
