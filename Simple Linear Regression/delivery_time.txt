# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# reading a csv file using pandas library
delivery_time=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Simple Linear Regression_Later\\Assignment\\delivery_time.csv")
delivery_time.columns
#####Exploratory Data Analysis#########################################################
delivery_time.mean() ##Sorting Time -  6.190476, Delivery Time -  16.790952
delivery_time.median() ### Sorting Time - 6.00, Delivery Time - 17.83
delivery_time.mode() 
####Measures of Dispersion
delivery_time.var() ##### Sorting Time -  6.461905 , Delivery Time - 25.754619
delivery_time.std() #### Sorting Time - 2.542028 , Delivery Time - 5.074901
#### Calculate the range value
range1 = max(delivery_time['Sorting Time'])-min(delivery_time['Sorting Time'])
range1 ## 8
range2 = max(delivery_time['Delivery Time'])-min(delivery_time['Delivery Time'])
range2 ### 21.0
### Calculate skewness and Kurtosis
delivery_time.skew() ###Sorting Time - 0.047115 , Delivery Time - 0.352390
delivery_time.kurt() ### Sorting Time - -1.148455 , Delivery Time - 0.317960
####Graphidelivery_time Representation 
plt.hist(delivery_time["Sorting Time"])
plt.hist(delivery_time["Delivery Time"])
plt.boxplot(delivery_time["Sorting Time"],0,"rs",0)
plt.boxplot(delivery_time["Delivery Time"],0,"rs",0)
plt.plot(delivery_time["Sorting Time"],delivery_time["Delivery Time"],"bo");plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")
# table 
pd.crosstab(delivery_time["Sorting Time"],delivery_time["Delivery Time"])
## Barplot
pd.crosstab(delivery_time["Sorting Time"],delivery_time["Delivery Time"]).plot(kind = "bar", width = 1.55)
import seaborn as sns 
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="Sorting Time",y="Delivery Time",data=delivery_time)
sns.pairplot(delivery_time.iloc[:,0:2]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(delivery_time,hue="Delivery Time",size=3)
delivery_time["Sorting Time"].value_counts()
delivery_time["Delivery Time"].value_counts()
delivery_time["Sorting Time"].value_counts().plot(kind="pie")
delivery_time["Delivery Time"].value_counts().plot(kind="pie")
sns.pairplot(delivery_time,hue="Delivery Time",size=3,diag_kind = "kde")
sns.FacetGrid(delivery_time,hue="Delivery Time").map(plt.scatter,"Sorting Time","Delivery Time").add_legend()
sns.FacetGrid(delivery_time,hue="Delivery Time").map(sns.kdeplot,"Sorting Time").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(delivery_time['Sorting Time'], dist="norm",plot=pylab)
stats.probplot(delivery_time['Delivery Time'],dist="norm",plot=pylab)
### For sorting time
# ppf => Percent point function 
stats.norm.ppf(0.975,6.190476,2.542028)# similar to qnorm in R ----11.17
# cdf => cumulative distributive function 
stats.norm.cdf(delivery_time["Sorting Time"],6.190476,2.542028) # similar to pnorm in R ---0.986941255297407
#### For delivery_time
stats.norm.ppf(0.975,16.790952,5.074901)# similar to qnorm in R ---- 26.74
# cdf => cumulative distributive function 
stats.norm.cdf(delivery_time["Delivery Time"],16.790952,5.074901) # similar to pnorm in R ---0.9809239269169406
stats.t.ppf(0.975, 20) # similar to qt in R ------  2.08
####Correlation 
delivery_time.corr(method = "pearson")
delivery_time.corr(method = "kendall")
delivery_time.corr()
delivery_time["Delivery Time"].corr(delivery_time["Sorting Time"]) # # correlation value between X and Y -- 0.83
np.corrcoef(delivery_time["Sorting Time"],delivery_time["Delivery Time"])
###### Lets do normalization
from sklearn import preprocessing
x_array = np.array(delivery_time['Sorting Time'])
normalized_X = preprocessing.normalize([x_array])
y_array = np.array(delivery_time['Delivery Time'])
normalized_Y = preprocessing.normalize([y_array])
# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols('delivery_time["Delivery Time"]~ delivery_time["Sorting Time"]',data=delivery_time).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()
model.conf_int(0.05) # 95% confidence interval
##pred = model.predict(delivery_time.iloc[:,0]) # Predicted values of Delivery Time using the model
pred = model.predict(pd.DataFrame(delivery_time['Sorting Time']))
pred.corr(delivery_time['Delivery Time']) # 0.95
# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=delivery_time['Sorting Time'],y=delivery_time['Delivery Time'],color='red');plt.plot(delivery_time['Sorting Time'],pred,color='black');plt.xlabel('WEIGHT GAINED (GRAMS)');plt.ylabel('CALORIES CONSUMED')
# Transforming variables for accuracy
####Lets start with Logarithmic transformation
model2 = smf.ols('delivery_time["Delivery Time"]~ np.log(delivery_time["Sorting Time"])',data=delivery_time).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(delivery_time['Sorting Time']))
pred2.corr(delivery_time['Delivery Time'])  #### 0.834
# pred2 = model2.predict(wcat.iloc[:,0])
plt.scatter(x=np.log(delivery_time['Sorting Time']),y=delivery_time['Delivery Time'],color='green');plt.plot(np.log(delivery_time['Sorting Time']),pred2,color='blue');plt.xlabel('LOG(WEIGHT GAINED (GRAMS))');plt.ylabel('CALORIES CONSUMED')
resid_2 = pred2-delivery_time["Delivery Time"]
# Exponential transformation
model3 = smf.ols('np.log(delivery_time["Delivery Time"])~ delivery_time["Sorting Time"]',data=delivery_time).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(delivery_time['Sorting Time']))
pred3=np.exp(pred_log)  # as we have used log(delivery_time["Delivery Time"]) in preparing model so we need to convert it back
pred3
pred3.corr(delivery_time["Delivery Time"])  #### 0.8085780108289262
plt.scatter(x=delivery_time['Sorting Time'],y=np.log(delivery_time['Delivery Time']),color='green');plt.plot(delivery_time['Sorting Time'],np.exp(pred_log),color='blue');plt.xlabel('WEIGHT GAINED (GRAMS)');plt.ylabel('LOG(CALORIES CONSUMED)')
resid_3 = pred3-delivery_time["Delivery Time"]
# Quadratic model
delivery_time["Sorting Time-Sq"] = delivery_time['Sorting Time'] * delivery_time['Sorting Time']
##model_quad = smf.ols('delivery_time["Delivery Time"]~(delivery_time['Sorting Time'] + delivery_time['Sorting Time-Sq']'),data=delivery_time).fit()
model_quad = smf.ols('delivery_time["Delivery Time"]~delivery_time["Sorting Time"] + delivery_time["Sorting Time-Sq"]',data=delivery_time).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(delivery_time["Sorting Time"])
model_quad.conf_int(0.05) 
pred_quad.corr(delivery_time['Delivery Time']) ###0.08
plt.scatter(delivery_time["Sorting Time"] + delivery_time["Sorting Time-Sq"],delivery_time["Delivery Time"],color='blue');plt.plot(delivery_time['Sorting Time'] + delivery_time["Sorting Time-Sq"],pred_quad,color = 'r')
plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
plt.hist(model_quad.resid_pearson) # histogram for residual values 
resid_4 = pred_quad-delivery_time["Delivery Time"]
##### Square Transformation
model5 = smf.ols('delivery_time["Delivery Time"]~ (delivery_time["Sorting Time"]*delivery_time["Sorting Time"])',data=delivery_time).fit()
model5.params
model5.summary()
print(model5.conf_int(0.01)) # 99% confidence level
pred5 = model5.predict(pd.DataFrame(delivery_time['Sorting Time']*delivery_time['Sorting Time']))
pred5.corr(delivery_time['Delivery Time'])  ###  0.8259972607955327
# pred2 = model2.predict(wcat.iloc[:,0])
plt.scatter(x=(delivery_time['Sorting Time']*delivery_time['Sorting Time']),y=delivery_time['Delivery Time'],color='green');plt.plot((delivery_time['Sorting Time']*delivery_time['Sorting Time']),pred5,color='blue');plt.xlabel('WEIGHT GAINED (GRAMS) * WEIGHT GAINED (GRAMS)');plt.ylabel('CALORIES CONSUMED')
plt.hist(model5.resid_pearson)
resid_5 = pred5-delivery_time["Delivery Time"]
#### Square root Transformation
model6 = smf.ols('delivery_time["Delivery Time"]~ np.sqrt(delivery_time["Sorting Time"])',data=delivery_time).fit()
model6.params
model6.summary()
print(model6.conf_int(0.01)) # 99% confidence level
pred6 = model6.predict(pd.DataFrame(np.sqrt(delivery_time['Sorting Time'])))
pred6.corr(delivery_time['Delivery Time'])  #### 0.9559736469475248
# pred2 = model2.predict(wcat.iloc[:,0])
plt.scatter(x=np.sqrt(delivery_time['Sorting Time']),y=delivery_time['Delivery Time'],color='green');plt.plot(np.sqrt(delivery_time['Sorting Time']),pred6,color='blue');plt.xlabel('SQRT(WEIGHT GAINED (GRAMS))');plt.ylabel('CALORIES CONSUMED')
resid_6 = pred6-delivery_time["Delivery Time"]
##### Reciprodelivery_time Transformation
reci = 1/delivery_time["Sorting Time"]
model7 = smf.ols('delivery_time["Delivery Time"]~ reci',data=delivery_time).fit()
model7.params
model7.summary()
print(model7.conf_int(0.01)) # 99% confidence level
pred7 = model7.predict(pd.DataFrame(reci))
pred7.corr(delivery_time['Delivery Time'])  #### 0.7852620940058638
# pred2 = model2.predict(wcat.iloc[:,0])
##plt.scatter(x=1/delivery_time["Sorting Time"],y=delivery_time['Delivery Time'],color='green');plt.plot(1/delivery_time["Sorting Time"]),pred7,color='blue');plt.xlabel('1/(WEIGHT GAINED (GRAMS))');plt.ylabel('CALORIES CONSUMED')
plt.scatter(reci,y=delivery_time['Delivery Time'],color='green');plt.plot(reci,pred7,color='green');plt.xlabel('1/(WEIGHT GAINED (GRAMS))');plt.ylabel('CALORIES CONSUMED')
resid_7 = pred7-delivery_time["Delivery Time"]
############################### Implementing the Linear Regression model from sklearn library
from sklearn.linear_model import LinearRegression
import numpy as np
plt.scatter(delivery_time["Sorting Time"],delivery_time["Delivery Time"])
model1 = LinearRegression()
model1.fit(delivery_time["Sorting Time"].values.reshape(-1,1),delivery_time["Delivery Time"])
pred1 = model1.predict(delivery_time["Sorting Time"].values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(delivery_time["Sorting Time"].values.reshape(-1,1),delivery_time["Delivery Time"])  ####0.68
rmse1 = np.sqrt(np.mean((pred1-delivery_time["Delivery Time"])**2)) #2.79
model1.coef_
model1.intercept_  ###1577.200
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-delivery_time["Delivery Time"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred1-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred1-delivery_time["Delivery Time"],dist="norm",plot=pylab)
#### Check for Logarithmic Transformation
plt.scatter(np.log(delivery_time["Sorting Time"]),delivery_time["Delivery Time"])
model2 = LinearRegression()
model2.fit(np.log(delivery_time["Sorting Time"]).values.reshape(-1,1),delivery_time["Delivery Time"])
pred2 = model2.predict(np.log(delivery_time["Sorting Time"]).values.reshape(-1,1))
# Adjusted R-Squared value
model2.score(np.log(delivery_time["Sorting Time"]).values.reshape(-1,1),delivery_time["Delivery Time"])  ####  0.69
rmse2 = np.sqrt(np.mean((pred2-delivery_time["Delivery Time"])**2)) #### 2.73
model2.coef_
model2.intercept_  ### 1.16
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-delivery_time["Delivery Time"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred2-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred2-delivery_time["Delivery Time"],dist="norm",plot=pylab)
#### Check for Exponential Transformation
plt.scatter(np.exp(delivery_time["Sorting Time"]),delivery_time["Delivery Time"])
model3 = LinearRegression()
model_3 = model3.fit(delivery_time["Sorting Time"].values.reshape(-1,1),np.log(delivery_time["Delivery Time"]))
pred3 = model_3.predict(delivery_time["Sorting Time"].values.reshape(-1,1))
pred3exp=np.exp(pred3)
# Adjusted R-Squared value
model3.score(delivery_time["Sorting Time"].values.reshape(-1,1),np.log(delivery_time["Delivery Time"]))  ####  0.711
rmse3 = np.sqrt(np.mean((pred3exp-delivery_time["Delivery Time"])**2)) #2.94
model3.coef_
model3.intercept_  ### 2.12
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred3,(pred3-delivery_time["Delivery Time"]),color="blue")
plt.hlines(y=-3000,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred3-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred3-delivery_time["Delivery Time"],dist="norm",plot=pylab)
### Fitting Quadratic Regression 
delivery_time["Sorting Time Sq2"] = delivery_time["Sorting Time"] * delivery_time["Sorting Time"]
plt.scatter(delivery_time["Sorting Time"]+delivery_time["Sorting Time Sq2"],delivery_time["Delivery Time"])
model4 = LinearRegression()
model4.fit((delivery_time["Sorting Time"]+delivery_time["Sorting Time Sq2"]).values.reshape(-1,1),delivery_time["Delivery Time"])
pred4 = model4.predict((delivery_time["Sorting Time"]+delivery_time["Sorting Time Sq2"]).values.reshape(-1,1))
# Adjusted R-Squared value
model4.score((delivery_time["Sorting Time"]+delivery_time["Sorting Time Sq2"]).values.reshape(-1,1),delivery_time["Delivery Time"])# 0.635
rmse4 = np.sqrt(np.mean((pred4-delivery_time["Delivery Time"])**2)) # 2.99
model4.coef_
model4.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred4,(pred4-delivery_time["Delivery Time"]),color="violet")
plt.hlines(y=0,xmin=0,xmax=5000)  
# Checking normal distribution
plt.hist(pred4-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred4-delivery_time["Delivery Time"],dist="norm",plot=pylab)
# Let us prepare a model by applying transformation on dependent variable
####Square Transformation
plt.scatter((delivery_time["Sorting Time"] * delivery_time["Sorting Time"]),delivery_time["Delivery Time"])
model5 = LinearRegression()
model5.fit((delivery_time["Sorting Time"] * delivery_time["Sorting Time"]).values.reshape(-1,1),delivery_time["Delivery Time"])
pred5 = model5.predict((delivery_time["Sorting Time"] * delivery_time["Sorting Time"]).values.reshape(-1,1))
# Adjusted R-Squared value
model5.score((delivery_time["Sorting Time"] * delivery_time["Sorting Time"]).values.reshape(-1,1),delivery_time["Delivery Time"])  ####  0.63
rmse5 = np.sqrt(np.mean((pred5-delivery_time["Delivery Time"])**2)) #3.011
model5.coef_
model5.intercept_  ### 11.24
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred5,(pred5-delivery_time["Delivery Time"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred5-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred5-delivery_time["Delivery Time"],dist="norm",plot=pylab)
###### Square-root Transformation
plt.scatter(np.sqrt(delivery_time["Sorting Time"]),delivery_time["Delivery Time"])
model6 = LinearRegression()
model6.fit(np.sqrt(delivery_time["Sorting Time"]).values.reshape(-1,1),delivery_time["Delivery Time"])
pred6 = model6.predict(np.sqrt(delivery_time["Sorting Time"]).values.reshape(-1,1))
# Adjusted R-Squared value
model6.score(np.sqrt(delivery_time["Sorting Time"]).values.reshape(-1,1),delivery_time["Delivery Time"])  ####  0.69
rmse6 = np.sqrt(np.mean((pred6-delivery_time["Delivery Time"])**2)) #### 2.73
model6.coef_
model6.intercept_  ### -2.52
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred6,(pred6-delivery_time["Delivery Time"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred6-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred6-delivery_time["Delivery Time"],dist="norm",plot=pylab)
###### Reciprodelivery_time last one
reci = delivery_time["Sorting Time"]
plt.scatter(reci,delivery_time["Delivery Time"])
model7 = LinearRegression()
model7.fit(reci.values.reshape(-1,1),delivery_time["Delivery Time"])
pred7 = model7.predict(reci.values.reshape(-1,1))
# Adjusted R-Squared value
model7.score(reci.values.reshape(-1,1),delivery_time["Delivery Time"])  ####  0.6822
rmse7 = np.sqrt(np.mean((pred7-delivery_time["Delivery Time"])**2)) ### 2.79
model7.coef_
model7.intercept_  ### 6.58
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred7,(pred7-delivery_time["Delivery Time"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred7-delivery_time["Delivery Time"])
import pylab
import scipy.stats as st
st.probplot(pred7-delivery_time["Delivery Time"],dist="norm",plot=pylab)
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
delivery_time = delivery_time[delivery_time.columns[delivery_time.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
delivery_time = delivery_time.loc[delivery_time.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
delivery_time = delivery_time.fillna(delivery_time.median())
#Max fill function for categorical columns
##delivery_time['column_name'].fillna(delivery_time['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = delivery_time['Delivery Time'].mean () + delivery_time['Delivery Time'].std () * factor
lower_lim = delivery_time['Delivery Time'].mean () - delivery_time['Delivery Time'].std () * factor
delivery_time = delivery_time[(delivery_time['Delivery Time'] < upper_lim) & (delivery_time['Delivery Time'] > lower_lim)]
upper_lim = delivery_time['Sorting Time'].mean () + delivery_time['Sorting Time'].std () * factor
lower_lim = delivery_time['Sorting Time'].mean () - delivery_time['Sorting Time'].std () * factor
delivery_time = delivery_time[(delivery_time['Sorting Time'] < upper_lim) & (delivery_time['Sorting Time'] > lower_lim)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim = delivery_time['Delivery Time'].quantile(.95)
lower_lim = delivery_time['Delivery Time'].quantile(.05)
delivery_time = delivery_time[(delivery_time['Delivery Time'] < upper_lim) & (delivery_time['Delivery Time'] > lower_lim)]
upper_lim = delivery_time['Sorting Time'].quantile(.95)
lower_lim = delivery_time['Sorting Time'].quantile(.05)
delivery_time = delivery_time[(delivery_time['Sorting Time'] < upper_lim) & (delivery_time['Sorting Time'] > lower_lim)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
upper_lim = delivery_time['Delivery Time'].quantile(.95)
lower_lim = delivery_time['Delivery Time'].quantile(.05)
delivery_time.loc[(delivery_time['Delivery Time'] > upper_lim)] = upper_lim
delivery_time.loc[(delivery_time['Delivery Time'] < lower_lim)] = lower_lim
upper_lim = delivery_time['Sorting Time'].quantile(.95)
lower_lim = delivery_time['Sorting Time'].quantile(.05)
delivery_time.loc[(delivery_time['Sorting Time'] > upper_lim)] = upper_lim
delivery_time.loc[(delivery_time['Sorting Time'] < lower_lim)] = lower_lim
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
delivery_time['bin'] = pd.cut(delivery_time['Delivery Time'], bins=[8,14,20,29], labels=["Low", "Mid", "High"])
delivery_time['bin'] = pd.cut(delivery_time['Sorting Time'], bins=[2,4,7,10], labels=["Low", "Mid", "High"])
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
delivery_time = pd.DataFrame({'Delivery Time':[21,13.5,19.75,24,29,15.35,19,9.5,17.9,18.75,19.83,10.75,16.68,11.5,12.03,14.88,13.75,18.11,8,17.83,21.5]})
delivery_time['log+1'] = (delivery_time['Delivery Time']+1).transform(np.log)
#Negative Values Handling
delivery_time['log'] = (delivery_time['Delivery Time']-delivery_time['Delivery Time'].min()+1).transform(np.log)
delivery_time = pd.DataFrame({'Sorting Time':[10,4,6,9,10,6,7,3,10,9,8,4,7,3,3,4,6,7,2,7,5]})
delivery_time['log+1'] = (delivery_time['Sorting Time']+1).transform(np.log)
#Negative Values Handling
delivery_time['log'] = (delivery_time['Sorting Time']-delivery_time['Sorting Time'].min()+1).transform(np.log)
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = delivery_time.groupby('Delivery Time')
sums = grouped['Delivery Time'].sum().add_suffix('_sum')
avgs = grouped['Delivery Time'].mean().add_suffix('_avg')
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(delivery_time.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
df_std = stan(delivery_time.iloc[:,0:2])
##### Feature Extraction
import time
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X=delivery_time['Sorting Time']
Y=delivery_time['Delivery Time']
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
## Updated X and Y is
X=df_std['Sorting Time']
Y=df_std['Delivery Time']
### Create a function (forest_test) to divide the input data into train and test
### sets and then train and test a Random Forest Classifier.
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 21)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=21)
    trainedforest.fit(X_Train,Y_Train)
    print(time.process_time() - start)
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    forest_test(X, Y)
#### Lets Start with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X=[[ 1.49861618 ,-0.8617043,  -0.07493081 , 1.10522943,  1.49861618 ,-0.07493081
  ,0.31845594, -1.25509105 , 1.49861618 , 1.10522943 , 0.71184268 ,-0.8617043,
  0.31845594, -1.25509105, -1.25509105, -0.8617043 , -0.07493081 , 0.31845594,
 -1.64847779,  0.31845594 ,-0.46831755]*1]*21
Y=[[ 0.829385, -0.648476,0.583075,1.420530,2.405771, -0.283937,0.435289,-1.436669,
     0.218536,0.386027,0.598839,-1.190359,-0.021863,-1.042573,-0.938137,-0.376550,
   -0.599214,0.259916,-1.732241,0.204742,0.927909]*1]*21
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
Y=df_norm['Delivery Time']
PCA_df = pd.concat([PCA_df,Y],axis=1)
PCA_df['Delivery Time'] = LabelEncoder().fit_transform(PCA_df['Delivery Time'])
PCA_df.head()
### Plot data distribution in a 2D scatter plot
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
DeliveryTime = PCA_df['Delivery Time']
colors = ['r', 'b']
for clas, color in zip(DeliveryTime, colors):
    plt.scatter(PCA_df.loc[PCA_df['Delivery Time'] == clas, 'PC1'], 
                PCA_df.loc[PCA_df['Delivery Time'] == clas, 'PC2'],
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 15)
plt.legend(['High Risk', 'Low Risk'])
plt.grid()
#########################################################################
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
### Visualize the decision boundary used by Random Forest in order to 
### classify each of the different data points
##from itertools import product
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
###Let's check LDA to reduce our dataset to just one feature, test its accuracy and plot the results
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
# run an LDA and use it to transform the features
X=[[1.49861618],[-0.8617043],[-0.07493081],[1.10522943], [1.49861618] ,[-0.07493081],
 [0.31845594], [-1.25509105] , [1.49861618],[1.10522943],[0.71184268] ,[-0.8617043],
 [ 0.31845594], [-1.25509105], [-1.25509105], [-0.8617043] ,[-0.07493081],[0.31845594],
 [-1.64847779],  [0.31845594] ,[-0.46831755]]
Y= [[0.829385],[-0.648476],[0.583075],[1.420530],[2.405771],[-0.283937],[0.435289],[-1.436669],
     [0.218536],[0.386027],[0.598839],[-1.190359],[-0.021863],[-1.042573],[-0.938137],[-0.376550],
   [-0.599214],[0.259916],[-1.732241],[0.204742],[0.927909]]
X=np.array(X,dtype=np.float64)
Y=np.array(Y,dtype=np.float64)
labels=np.unique(Y)
print(labels)
## [-1.64847779 -1.25509105 -0.8617043  -0.46831755 -0.07493081  0.31845594
##  0.71184268  1.10522943  1.49861618]
X=[[-1.64847779] ,[-1.25509105],[-0.8617043],[-0.46831755],[-0.07493081],[0.31845594],[0.71184268],[1.10522943],[1.49861618]]
Y=[[-1.732241],[-1.042573],[-0.376550],[0.927909],[-0.599214],[0.259916],[0.598839],[0.386027],[-1.190359]]
X_lda = lda.fit(X, Y).transform(X)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y, test_size = 0.30, random_state = 101)
start = time.process_time()
lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)
print(time.process_time() - start)
predictionlda = lda.predict(X_Test_Reduced)
print(confusion_matrix(Y_Test_Reduced,predictionlda))
print(classification_report(Y_Test_Reduced,predictionlda))
########################################################################
###Run LLE on dataset to reduce data dimensionality to 3 dimensions, test the accuracy and plot the results
from sklearn.manifold import LocallyLinearEmbedding
embedding=LocallyLinearEmbedding(n_components=3)
X_lle=embedding.fit_transform(X)
forest_test(X_lle,Y)
########################################################################
### Using ICA to reduce datasets to 3 features, test its accuracy using a Random Forest Classifier and plot the results
from sklearn.decomposition import FastICA
ica=FastICA(n_components=3)
np.isnan(X).any()
np.isinf(X).any()
X=[[1.49861618],[-0.8617043],[-0.07493081],[1.10522943], [1.49861618] ,[-0.07493081],
 [0.31845594], [-1.25509105] , [1.49861618],[1.10522943],[0.71184268] ,[-0.8617043],
 [ 0.31845594], [-1.25509105], [-1.25509105], [-0.8617043] ,[-0.07493081],[0.31845594],
 [-1.64847779],  [0.31845594] ,[-0.46831755]]
X_ica=ica.fit_transform(X)
forest_test(X_ica, Y)
#############################################################################
##t-distributed Stochastic Neighbour Embedding(t-SNE)
### t-SNE works by minimizing the divergence between a distribution constituted by the pairwise probability similarities
### of the input features in the original high dimensional space and its equivalent in the reduced low dimensional space
from sklearn.manifold import TSNE
start=time.process_time()
tsne=TSNE()
X_tsne=tsne.fit_transform(X)
print(time.process_time()-start) ##1.1388072999999963
forest_test(X_tsne,Y)
############################################################################
###Autoencoders use non-linear transformations to project data from a high dimension to a lower one
from keras.layers import Input,Dense
from keras.models import Model
input_layer =Input(shape=(X.shape[1],))
encoded=Dense(3,activation='relu')(input_layer)
decoded=Dense(X.shape[1],activation='softmax')(encoded)
autoencoder=Model(input_layer,decoded)
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
X1,X2,Y1,Y2=train_test_split(X,X,test_size=0.3,random_state=101)
autoencoder.fit(X1,Y1,epochs=21,batch_size=21,shuffle=True,verbose=21,validation_data=(X2,Y2))
encoder=Model(input_layer,encoded)
X_ae=encoder.predict(X)
forest_test(X_ae,Y)
############################################################################

