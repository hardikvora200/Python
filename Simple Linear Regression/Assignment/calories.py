# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# reading a csv file using pandas library
cal=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Simple Linear Regression_Later\\Assignment\\calories_consumed.csv")
cal.columns
#####Exploratory Data Analysis#########################################################
cal.mean() ##Weight gained (grams) - 357.714286, Calories Consumed - 2340.714286
cal.median() ### Weight gained(grams) - 200.00, Calories Consumed - 2250.0
cal.mode() #### Weight gained(grams) - 200.00, Calories Consumed - 1900
####Measures of Dispersion
cal.var() ##### Weight gained(grams) - 111350.681319 , Calories Consumed - 565668.681319
cal.std() ####Weight gained(grams) - 333.692495 , Calories Consumed - 752.109488
#### Calculate the range value
range1 = max(cal['Weight gained (grams)'])-min(cal['Weight gained (grams)'])
range1 ## 1038
range2 = max(cal['Calories Consumed'])-min(cal['Calories Consumed'])
range2 ###2500
### Calculate skewness and Kurtosis
cal.skew() ###Weight gained (grams) - 1.255737 , Calories Consumed - 0.654930
cal.kurt() ### Weight gained (grams) - 0.431272 , Calories Consumed - -0.290481
####Graphical Representation 
plt.hist(cal["Weight gained (grams)"])
plt.hist(cal["Calories Consumed"])
plt.boxplot(cal["Weight gained (grams)"],0,"rs",0)
plt.boxplot(cal["Calories Consumed"],0,"rs",0)
plt.plot(cal["Calories Consumed"],cal["Weight gained (grams)"],"bo");plt.xlabel("Calories Consumed");plt.ylabel("Weight gained (grams)")
# table 
pd.crosstab(cal["Calories Consumed"],cal["Weight gained (grams)"])
## Barplot
pd.crosstab(cal["Calories Consumed"],cal["Weight gained (grams)"]).plot(kind = "bar", width = 1.85)
import seaborn as sns 
# getting boxplot of Calories Consumed with respect to each category of Weight gained (grams) 
sns.boxplot(x="Calories Consumed",y="Weight gained (grams)",data=cal)
sns.pairplot(cal.iloc[:,0:2]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(cal,hue="Weight gained (grams)",size=3)
cal["Weight gained (grams)"].value_counts()
cal["Calories Consumed"].value_counts()
cal["Weight gained (grams)"].value_counts().plot(kind="pie")
cal["Calories Consumed"].value_counts().plot(kind="pie")
sns.pairplot(cal,hue="Weight gained (grams)",size=3,diag_kind = "kde")
sns.FacetGrid(cal,hue="Weight gained (grams)").map(plt.scatter,"Calories Consumed","Weight gained (grams)").add_legend()
sns.FacetGrid(cal,hue="Weight gained (grams)").map(sns.kdeplot,"Weight gained (grams)").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(cal['Weight gained (grams)'], dist="norm",plot=pylab)
stats.probplot(cal['Calories Consumed'],dist="norm",plot=pylab)
# ppf => Percent point function 
stats.norm.ppf(0.975,357.714286,333.692495)# similar to qnorm in R ----1011.7395581113121
# cdf => cumulative distributive function 
stats.norm.cdf(cal["Weight gained (grams)"],357.714286,333.692495) # similar to pnorm in R ---0.986941255297407
#### For calories consumed
stats.norm.ppf(0.975,2340.714286,752.109488)# similar to qnorm in R ----3814.8217949108603
# cdf => cumulative distributive function 
stats.norm.cdf(cal["Calories Consumed"],2340.714286,752.109488) # similar to pnorm in R ---0.9809239269169406
stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
cal.corr(method = "pearson")
cal.corr(method = "kendall")
cal.corr()
cal["Calories Consumed"].corr(cal["Weight gained (grams)"]) # # correlation value between X and Y -- 0.9469910088554458
np.corrcoef(cal["Calories Consumed"],cal["Weight gained (grams)"])
###### Lets do normalization
from sklearn import preprocessing
x_array = np.array(cal['Weight gained (grams)'])
normalized_X = preprocessing.normalize([x_array])
y_array = np.array(cal['Calories Consumed'])
normalized_Y = preprocessing.normalize([y_array])
# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols('cal["Weight gained (grams)"]~ cal["Weight gained (grams)"]',data=cal).fit()
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()
model.conf_int(0.05) # 95% confidence interval
pred = model.predict(cal.iloc[:,0]) # Predicted values of Calories Consumed using the model
pred = model.predict(pd.DataFrame(cal['Weight gained (grams)']))
# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=cal['Calories Consumed'],y=cal['Weight gained (grams)'],color='red');plt.plot(cal['Calories Consumed'],pred,color='black');plt.xlabel('CALORIES CONSUMED');plt.ylabel('WEIGHT GAINED(GRAMS)')
pred.corr(cal['Calories Consumed']) # 0.95
# Transforming variables for accuracy
####Lets start with Logarithmic transformation
model2 = smf.ols('cal["Weight gained (grams)"]~ np.log(cal["Calories Consumed"])',data=cal).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(pd.DataFrame(cal['Calories Consumed']))
pred2.corr(cal['Weight gained (grams)'])  #### 0.9368036903364726
# pred2 = model2.predict(wcat.iloc[:,0])
plt.scatter(x=np.log(cal['Calories Consumed']),y=cal['Weight gained (grams)'],color='green');plt.plot(np.log(cal['Calories Consumed']),pred2,color='blue');plt.xlabel('LOG(CALORIES CONSUMED)');plt.ylabel('WEIGHT GAINED(GRAMS)')
resid_2 = pred2-cal["Weight gained (grams)"]
# Exponential transformation
model3 = smf.ols('np.log(cal["Weight gained (grams)"])~ cal["Calories Consumed"]',data=cal).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(cal['Calories Consumed']))
pred3=np.exp(pred_log)  # as we have used log(cal["Weight gained (grams)"]) in preparing model so we need to convert it back
pred3
pred3.corr(cal["Weight gained (grams)"])  
plt.scatter(x=cal['Calories Consumed'],y=np.log(cal['Weight gained (grams)']),color='green');plt.plot(cal['Calories Consumed'],np.exp(pred_log),color='blue');plt.xlabel('CALORIES CONSUMED');plt.ylabel('LOG(WEIGHT GAINED(GRAMS))')
resid_3 = pred3-cal["Weight gained (grams)"]
# Quadratic model
cal["Calories Consumed-Sq"] = cal['Calories Consumed'] * cal['Calories Consumed']
model_quad = smf.ols('cal["Weight gained (grams)"]~(cal["Calories Consumed"] + cal["Calories Consumed-Sq"])',data=cal).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(cal["Calories Consumed"])
model_quad.conf_int(0.05)  
plt.scatter(cal["Calories Consumed"] + cal["Calories Consumed-Sq"],cal["Weight gained (grams)"],color='blue');plt.plot(cal['Calories Consumed'] + cal["Calories Consumed-Sq"],pred_quad,color = 'r')
plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
plt.hist(model_quad.resid_pearson) # histogram for residual values 
resid_4 = pred_quad-cal["Weight gained (grams)"]
##### Square Transformation
model5 = smf.ols('cal["Weight gained (grams)"]~ (cal["Calories Consumed"]*cal["Calories Consumed"])',data=cal).fit()
model5.params
model5.summary()
print(model5.conf_int(0.01)) # 99% confidence level
pred5 = model5.predict(pd.DataFrame(cal["Calories Consumed"]*cal["Calories Consumed"]))
pred5.corr(cal['Weight gained (grams)']) 
# pred2 = model2.predict(wcat.iloc[:,0])
plt.scatter(x=(cal['Calories Consumed']*cal['Calories Consumed']),y=cal['Weight gained (grams)'],color='green');plt.plot((cal['Calories Consumed']*cal['Calories Consumed']),pred5,color='blue');plt.xlabel('CALORIES CONSUMED * CALORIES CONSUMED');plt.ylabel('WEIGHT GAINED(GRAMS)')
resid_5 = pred5-cal["Weight gained (grams)"]
#### Square root Transformation
model6 = smf.ols('cal["Weight gained (grams)"]~ np.sqrt(cal["Calories Consumed"])',data=cal).fit()
model6.params
model6.summary()
print(model6.conf_int(0.01)) # 99% confidence level
pred6 = model6.predict(pd.DataFrame(np.sqrt(cal['Calories Consumed'])))
pred6.corr(cal['Weight gained (grams)'])  
# pred2 = model2.predict(wcat.iloc[:,0])
plt.scatter(x=np.sqrt(cal['Calories Consumed']),y=cal['Weight gained (grams)'],color='green');plt.plot(np.sqrt(cal['Calories Consumed']),pred6,color='blue');plt.xlabel('SQRT(CALORIES CONSUMED)');plt.ylabel('WEIGHT GAINED(GRAMS)')
resid_6 = pred6-cal["Weight gained (grams)"]
##### Reciprocal Transformation
reci = 1/cal["Calories Consumed"]
model7 = smf.ols('cal["Weight gained (grams)"]~ reci',data=cal).fit()
model7.params
model7.summary()
print(model7.conf_int(0.01)) # 99% confidence level
pred7 = model7.predict(pd.DataFrame(reci))
pred7.corr(cal['Weight gained (grams)'])  
# pred2 = model2.predict(wcat.iloc[:,0])
##plt.scatter(x=1/cal["Weight gained (grams)"],y=cal['Calories Consumed'],color='green');plt.plot(1/cal["Weight gained (grams)"]),pred7,color='blue');plt.xlabel('1/(WEIGHT GAINED (GRAMS))');plt.ylabel('CALORIES CONSUMED')
plt.scatter(x=reci,y=cal['Weight gained (grams)'],color='green');plt.plot(reci,pred7,color='green');plt.xlabel('1/(CALORIES CONSUMED)');plt.ylabel('WEIGHT GAINED(GRAMS)')
resid_7 = pred7-cal["Weight gained (grams)"]
############################### Implementing the Linear Regression model from sklearn library
from sklearn.linear_model import LinearRegression
import numpy as np
plt.scatter(cal["Calories Consumed"],cal["Weight gained (grams)"])
model1 = LinearRegression()
model1.fit(cal["Calories Consumed"].values.reshape(-1,1),cal["Weight gained (grams)"])
pred1 = model1.predict(cal["Calories Consumed"].values.reshape(-1,1))
# Adjusted R-Squared value
model1.score(cal["Calories Consumed"].values.reshape(-1,1),cal["Weight gained (grams)"]) 
rmse1 = np.sqrt(np.mean((pred1-cal["Weight gained (grams)"])**2)) 
model1.coef_
model1.intercept_ 
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred1,(pred1-cal["Weight gained (grams)"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred1-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred1-cal["Weight gained (grams)"],dist="norm",plot=pylab)
#### Check for Logarithmic Transformation
plt.scatter(np.log(cal["Calories Consumed"]),cal["Weight gained (grams)"])
model2 = LinearRegression()
model2.fit(np.log(cal["Calories Consumed"]).values.reshape(-1,1),cal["Weight gained (grams)"])
pred2 = model2.predict(np.log(cal["Calories Consumed"]).values.reshape(-1,1))
# Adjusted R-Squared value
model2.score(np.log(cal["Calories Consumed"]).values.reshape(-1,1),cal["Weight gained (grams)"]) 
rmse2 = np.sqrt(np.mean((pred2-cal["Weight gained (grams)"])**2))
model2.coef_
model2.intercept_  
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred2,(pred2-cal["Weight gained (grams)"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred2-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred2-cal["Weight gained (grams)"],dist="norm",plot=pylab)
#### Check for Exponential Transformation
plt.scatter(np.exp(cal["Calories Consumed"]),cal["Weight gained (grams)"])
model3 = LinearRegression()
model_3 = model3.fit(cal["Calories Consumed"].values.reshape(-1,1),np.log(cal["Weight gained (grams)"]))
pred3 = model_3.predict(cal["Calories Consumed"].values.reshape(-1,1))
pred3exp=np.exp(pred3)
# Adjusted R-Squared value
model3.score(cal["Calories Consumed"].values.reshape(-1,1),np.log(cal["Weight gained (grams)"])) 
rmse3 = np.sqrt(np.mean((pred3exp-cal["Calories Consumed"])**2)) 
model3.coef_
model3.intercept_  
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred3,(pred3-cal["Weight gained (grams)"]),color="blue")
plt.hlines(y=-3000,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred3-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred3-cal["Weight gained (grams)"],dist="norm",plot=pylab)
### Fitting Quadratic Regression 
cal["Calories Consumed Sq2"] = cal["Calories Consumed"] * cal["Calories Consumed"]
plt.scatter(cal["Calories Consumed"]+cal["Calories Consumed Sq2"],cal["Weight gained (grams)"])
model4 = LinearRegression()
model4.fit((cal["Calories Consumed"]+cal["Calories Consumed Sq2"]).values.reshape(-1,1),cal["Weight gained (grams)"])
pred4 = model4.predict((cal["Calories Consumed"]+cal["Calories Consumed Sq2"]).values.reshape(-1,1))
# Adjusted R-Squared value
model4.score((cal["Calories Consumed"]+cal["Calories Consumed Sq2"]).values.reshape(-1,1),cal["Weight gained (grams)"])
rmse4 = np.sqrt(np.mean((pred4-cal["Weight gained (grams)"])**2)) 
model4.coef_
model4.intercept_
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred4,(pred4-cal["Weight gained (grams)"]),color="violet")
plt.hlines(y=0,xmin=0,xmax=5000)  
# Checking normal distribution
plt.hist(pred4-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred4-cal["Weight gained (grams)"],dist="norm",plot=pylab)
# Let us prepare a model by applying transformation on dependent variable
####Square Transformation
plt.scatter((cal["Calories Consumed"] * cal["Calories Consumed"]),cal["Weight gained (grams)"])
model5 = LinearRegression()
model5.fit((cal["Calories Consumed"] * cal["Calories Consumed"]).values.reshape(-1,1),cal["Weight gained (grams)"])
pred5 = model5.predict((cal["Calories Consumed"] * cal["Calories Consumed"]).values.reshape(-1,1))
# Adjusted R-Squared value
model5.score((cal["Calories Consumed"] * cal["Calories Consumed"]).values.reshape(-1,1),cal["Weight gained (grams)"])  
rmse5 = np.sqrt(np.mean((pred5-cal["Weight gained (grams)"])**2)) 
model5.coef_
model5.intercept_  
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred5,(pred5-cal["Weight gained (grams)"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred5-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred5-cal["Weight gained (grams)"],dist="norm",plot=pylab)
###### Square-root Transformation
plt.scatter(np.sqrt(cal["Calories Consumed"]),cal["Weight gained (grams)"])
model6 = LinearRegression()
model6.fit(np.sqrt(cal["Calories Consumed"]).values.reshape(-1,1),cal["Weight gained (grams)"])
pred6 = model6.predict(np.sqrt(cal["Calories Consumed"]).values.reshape(-1,1))
# Adjusted R-Squared value
model6.score(np.sqrt(cal["Calories Consumed"]).values.reshape(-1,1),cal["Weight gained (grams)"])  
rmse6 = np.sqrt(np.mean((pred6-cal["Weight gained (grams)"])**2)) 
model6.coef_
model6.intercept_  
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred6,(pred6-cal["Weight gained (grams)"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred6-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred6-cal["Weight gained (grams)"],dist="norm",plot=pylab)
###### Reciprocal last one
reci = 1/(cal["Calories Consumed"])
plt.scatter(reci,cal["Weight gained (grams)"])
model7 = LinearRegression()
model7.fit(reci.values.reshape(-1,1),cal["Weight gained (grams)"])
pred7 = model7.predict(reci.values.reshape(-1,1))
# Adjusted R-Squared value
model7.score(reci.values.reshape(-1,1),cal["Weight gained (grams)"])  
rmse7 = np.sqrt(np.mean((pred7-cal["Weight gained (grams)"])**2)) 
model7.coef_
model7.intercept_  
#### Residuals Vs Fitted values
import matplotlib.pyplot as plt
plt.scatter(pred7,(pred7-cal["Weight gained (grams)"]),color="blue")
plt.hlines(y=0,xmin=0,xmax=4500) 
# checking normal distribution for residual
plt.hist(pred7-cal["Weight gained (grams)"])
import pylab
import scipy.stats as st
st.probplot(pred7-cal["Weight gained (grams)"],dist="norm",plot=pylab)
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
cal = cal[cal.columns[cal.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
cal = cal.loc[cal.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
cal = cal.fillna(cal.median())
#Max fill function for categorical columns
##cal['column_name'].fillna(cal['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = cal['Weight gained (grams)'].mean () + cal['Weight gained (grams)'].std () * factor
lower_lim = cal['Weight gained (grams)'].mean () - cal['Weight gained (grams)'].std () * factor
cal = cal[(cal['Weight gained (grams)'] < upper_lim) & (cal['Weight gained (grams)'] > lower_lim)]
upper_lim = cal['Calories Consumed'].mean () + cal['Calories Consumed'].std () * factor
lower_lim = cal['Calories Consumed'].mean () - cal['Calories Consumed'].std () * factor
cal = cal[(cal['Calories Consumed'] < upper_lim) & (cal['Calories Consumed'] > lower_lim)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim = cal['Weight gained (grams)'].quantile(.95)
lower_lim = cal['Weight gained (grams)'].quantile(.05)
cal = cal[(cal['Weight gained (grams)'] < upper_lim) & (cal['Weight gained (grams)'] > lower_lim)]
upper_lim = cal['Calories Consumed'].quantile(.95)
lower_lim = cal['Calories Consumed'].quantile(.05)
cal = cal[(cal['Calories Consumed'] < upper_lim) & (cal['Calories Consumed'] > lower_lim)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
upper_lim = cal['Weight gained (grams)'].quantile(.95)
lower_lim = cal['Weight gained (grams)'].quantile(.05)
cal.loc[(cal['Weight gained (grams)'] > upper_lim)] = upper_lim
cal.loc[(cal['Weight gained (grams)'] < lower_lim)] = lower_lim
upper_lim = cal['Calories Consumed'].quantile(.95)
lower_lim = cal['Calories Consumed'].quantile(.05)
cal.loc[(cal['Calories Consumed'] > upper_lim)] = upper_lim
cal.loc[(cal['Calories Consumed'] < lower_lim)] = lower_lim
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
cal['bin'] = pd.cut(cal['Weight gained (grams)'], bins=[62,350,750,1100], labels=["Low", "Mid", "High"])
cal['bin'] = pd.cut(cal['Calories Consumed'], bins=[1400,2000,3000,3900], labels=["Low", "Mid", "High"])
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
cal = pd.DataFrame({'Weight gained (grams)':[108,200,900,200,300,110,128,62,600,1100,100,150,350,700]})
cal['log+1'] = (cal['Weight gained (grams)']+1).transform(np.log)
#Negative Values Handling
cal['log'] = (cal['Weight gained (grams)']-cal['Weight gained (grams)'].min()+1).transform(np.log)
cal = pd.DataFrame({'Calories Consumed':[1500,2300,3400,2200,2500,1600,1400,1900,2800,3900,1670,1900,2700,3000]})
cal['log+1'] = (cal['Calories Consumed']+1).transform(np.log)
#Negative Values Handling
cal['log'] = (cal['Calories Consumed']-cal['Calories Consumed'].min()+1).transform(np.log)
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = cal.groupby('Weight gained (grams)')
sums = grouped['Weight gained (grams)'].sum().add_suffix('_sum')
avgs = grouped['Weight gained (grams)'].mean().add_suffix('_avg')
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(cal.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
df_std = stan(cal.iloc[:,0:2])
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
X=cal['Calories Consumed']
Y=cal['Weight gained (grams)']
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
## Updated X and Y is
X=df_std['Calories Consumed']
Y=df_std['Weight gained (grams)']
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
X=[[ -1.117808,-0.054133,1.408420,-0.187093,0.211785,-0.984849,-1.250768,-0.585971
    ,0.610663,2.073216,-0.891777,-0.585971,0.477704,0.876582]*1]*14
Y=[[-0.748337,-0.472634,1.625106,-0.472634,-0.172956,-0.742343, -0.688401, -0.886188,
    0.726075, 2.224460,-0.772311,-0.622472,-0.023118, 1.025752 ]*1]*14
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
Y=df_norm['Weight gained (grams)']
PCA_df = pd.concat([PCA_df,Y],axis=1)
PCA_df['Weight gained (grams)'] = LabelEncoder().fit_transform(PCA_df['Weight gained (grams)'])
PCA_df.head()
### Plot data distribution in a 2D scatter plot
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
DeliveryTime = PCA_df['Weight gained (grams)']
colors = ['r', 'b']
for clas, color in zip(DeliveryTime, colors):
    plt.scatter(PCA_df.loc[PCA_df['Weight gained (grams)'] == clas, 'PC1'], 
                PCA_df.loc[PCA_df['Weight gained (grams)'] == clas, 'PC2'],
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 15)
plt.legend(['Weight gained more', 'Weight gained normal'])
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
X=[[-1.117808],[-0.054133],[1.408420],[-0.187093],[0.211785],[-0.984849],[-1.250768],[-0.585971]
    ,[0.610663],[2.073216],[-0.891777],[-0.585971],[0.477704],[0.876582]*1]*14
Y=[[-0.748337],[-0.472634],[1.625106],[-0.472634],[-0.172956],[-0.742343],[-0.688401],[-0.886188],
    [0.726075],[2.224460],[-0.772311],[-0.622472],[-0.023118],[1.025752]*1]*14
X=np.array(X,dtype=np.float64)
Y=np.array(Y,dtype=np.float64)
labels=np.unique(X)
print(labels)
labels1=np.unique(Y)
print(labels1)
## [-1.64847779 -1.25509105 -0.8617043  -0.46831755 -0.07493081  0.31845594
##  0.71184268  1.10522943  1.49861618]
X=[[-1.250768],[-1.117808],[-0.984849],[-0.891777],[-0.585971],[-0.187093],
       [-0.054133],[0.211785],[0.477704],[0.610663],[0.876582],[1.40842] ,[2.073216]]
Y=[[-0.688401],[-0.748337],[-0.742343],[-0.772311],[-0.622472],[-0.472634],
   [-0.472634],[-0.172956],[-0.023118],[0.726075],[1.025752],[1.625106],[2.224460]]
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
X=[[-1.117808],[-0.054133],[1.408420],[-0.187093],[0.211785],[-0.984849],[-1.250768],[-0.585971]
    ,[0.610663],[2.073216],[-0.891777],[-0.585971],[0.477704],[0.876582]]
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
print(time.process_time()-start) 
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
autoencoder.fit(X1,Y1,epochs=14,batch_size=14,shuffle=True,verbose=14,validation_data=(X2,Y2))
encoder=Model(input_layer,encoded)
X_ae=encoder.predict(X)
forest_test(X_ae,Y)
############################################################################

