# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# reading a csv file using pandas library
startup=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Multilinear Regression\\Assignments\\50Startups.csv")
startup.columns
#####Exploratory Data Analysis#########################################################
startup.mean() ## R&D Spend - 73721.6156, Administration - 121344.6396, Marketing Spend - 211025.0978, Profit - 112012.6392
startup.median() ### R&D Spend - 73051.080, Administration - 122699.795, Marketing Spend - 212716.240, Profit - 107978.190
startup.mode() 
####Measures of Dispersion
startup.var() 
startup.std() #### R&D Spend - 45902.256482, Administration - 28017.802755, Marketing Spend - 122290.310726, Profit - 40306.180338
#### Calculate the range value
range1 = max(startup['R&D Spend'])-min(startup['R&D Spend'])  ###165349.2
range2 = max(startup['Administration'])-min(startup['Administration']) ### 131362.42
range3 = max(startup['Marketing Spend'])-min(startup['Marketing Spend']) ### 471784.1
range4 = max(startup['Profit'])-min(startup['Profit']) ###177580.43
### Calculate skewness and Kurtosis
startup.skew() ### R&D Spend - 0.164002, Administration - -0.489025, Marketing Spend - -0.046472, Profit - 0.023291
startup.kurt() ### R&D Spend - -0.761465, Administration - 0.225071, Marketing Spend - -0.671701, Profit - 0.063859
####Graphidelivery_time Representation 
plt.hist(startup["R&D Spend"])
plt.hist(startup["Administration"])
plt.hist(startup["Marketing Spend"])
plt.hist(startup["Profit"])
plt.boxplot(startup["R&D Spend"],0,"rs",0)
plt.boxplot(startup["Administration"],0,"rs",0)
plt.boxplot(startup["Marketing Spend"],0,"rs",0)
plt.boxplot(startup["Profit"],0,"rs",0)
plt.plot(startup["R&D Spend"],startup["Profit"],"bo");plt.xlabel("R&D Spend");plt.ylabel("Profit")
plt.plot(startup["Administration"],startup["Profit"],"bo");plt.xlabel("Administration");plt.ylabel("Profit")
plt.plot(startup["Marketing Spend"],startup["Profit"],"bo");plt.xlabel("Marketing Spend");plt.ylabel("Profit")
# table 
pd.crosstab(startup["R&D Spend"],startup["Profit"])
pd.crosstab(startup["Administration"],startup["Profit"])
pd.crosstab(startup["Marketing Spend"],startup["Profit"])
## Barplot
pd.crosstab(startup["R&D Spend"],startup["Profit"]).plot(kind = "bar", width = 1.85)
pd.crosstab(startup["Administration"],startup["Profit"]).plot(kind = "bar", width = 1.85)
pd.crosstab(startup["Marketing Spend"],startup["Profit"]).plot(kind = "bar", width = 1.85)
import seaborn as sns 
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="R&D Spend",y="Profit",data=startup)
sns.boxplot(x="Administration",y="Profit",data=startup)
sns.boxplot(x="Marketing Spend",y="Profit",data=startup)
sns.pairplot(startup.iloc[:,0:3]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(startup,hue="Profit",size=5)
startup["R&D Spend"].value_counts()
startup["Administration"].value_counts()
startup["Marketing Spend"].value_counts()
startup["R&D Spend"].value_counts().plot(kind = "pie")
startup["Administration"].value_counts().plot(kind = "pie")
startup["Marketing Spend"].value_counts().plot(kind = "pie")
sns.pairplot(startup,hue="Profit",size=4,diag_kind = "kde")
sns.FacetGrid(startup,hue="Profit").map(plt.scatter,"R&D Spend","Profit").add_legend()
sns.FacetGrid(startup,hue="Profit").map(plt.scatter,"Administration","Profit").add_legend()
sns.FacetGrid(startup,hue="Profit").map(plt.scatter,"Marketing Speed","Profit").add_legend()
sns.FacetGrid(startup,hue="Profit").map(sns.kdeplot,"R&D Spend").add_legend()
sns.FacetGrid(startup,hue="Profit").map(sns.kdeplot,"Administration").add_legend()
sns.FacetGrid(startup,hue="Profit").map(sns.kdeplot,"Marketing Spend").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(startup['R&D Spend'], dist="norm",plot=pylab)
stats.probplot(startup['Administration'],dist="norm",plot=pylab)
stats.probplot(startup['Marketing Spend'],dist="norm",plot=pylab)
stats.probplot(startup['Profit'],dist="norm",plot=pylab)
# ppf => Percent point function 
####R&D Spend
stats.norm.ppf(0.975,73721.6156,45902.256482)# similar to qnorm in R ---- 163688.38511384022
# cdf => cumulative distributive function 
stats.norm.cdf(startup["R&D Spend"],73721.6156,45902.256482) # similar to pnorm in R 
#### Administration
stats.norm.ppf(0.975,121344.6396,28017.802755)# similar to qnorm in R ---- 176258.52392574708
# cdf => cumulative distributive function 
stats.norm.cdf(startup["Administration"],121344.6396,28017.802755) # similar to pnorm in R 
#### Marketing Spend
stats.norm.ppf(0.975,211025.0978,122290.310726)# similar to qnorm in R ---- 450709.7024811723
# cdf => cumulative distributive function 
stats.norm.cdf(startup["Marketing Spend"],211025.0978,122290.310726) # similar to pnorm in R 
#### Profit
stats.norm.ppf(0.975, 112012.6392, 40306.180338)# similar to qnorm in R ---- 191011.30101685645
# cdf => cumulative distributive function 
stats.norm.cdf(startup["Profit"], 112012.6392, 40306.180338) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
startup.corr(method = "pearson")
startup.corr(method = "kendall")
startup["Profit"].corr(startup["Marketing Spend"]) # # correlation value between X and Y -- 0.7477657217414765
np.corrcoef(startup["Marketing Spend"],startup["Profit"])
np.corrcoef(startup["R&D Spend"],startup["Profit"])
np.corrcoef(startup["Administration"],startup["Profit"])
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(startup['Marketing Spend'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(startup['R&D Spend'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(startup['Administration'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(startup['Profit'])
normalized_D = preprocessing.normalize([d_array])
# to get top 6 rows
startup.head(40) # to get top n rows use cars.head(10)
startup.tail(10)
# Correlation matrix 
startup.corr()
# We see there exists High collinearity between input variables especially between
# [Profit & R&D Spend] , [Profit & Marketing Spend],[R&D Spend & Marketing Spend]
## so there exists collinearity problem
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup)
pd.tools.plotting.scatter_matrix(startup) ##-> also used for plotting all in one graph
#### preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
# Preparing model                  
ml1 = smf.ols('startup["Profit"]~ startup["R&D Spend"]+startup["Administration"]+startup["Marketing Spend"]',data=startup).fit() # regression model
# Getting coefficients of variables               
ml1.params
ml1.summary()
# p-values for Administration and Marketing Spend are more than 0.05 and also we know that [Administration, Marketing Spend] has high correlation value 
# preparing model based only on Administration
ml_a=smf.ols('startup["Profit"]~startup["Administration"]',data = startup).fit()  
ml_a.summary() # 0.162
# Preparing model based only on Marketing Spend
ml_ms=smf.ols('startup["Profit"]~startup["Marketing Spend"]',data = startup).fit()  
ml_ms.summary()
#### P-value < 0.05 which means Marketing Spend is Significant
# Preparing model based only on Marketing Spend and Administration
ml_msa=smf.ols('startup["Profit"]~startup["Marketing Spend"]+startup["Administration"]',data = startup).fit()  
ml_msa.summary()
# Both coefficients p-value became significant... 
# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 46,48 and 49 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
startup_new=startup.drop(startup.index[[46,48,49]],axis=0)
ml_new = smf.ols('startup_new["Profit"]~ startup_new["R&D Spend"]+startup_new["Administration"]+startup_new["Marketing Spend"]',data = startup_new).fit()    
# Getting coefficients of variables        
ml_new.params
# Summary
ml_new.summary() # 0.961
# Confidence values 99%
print(ml_new.conf_int(0.01)) # 99% confidence level
# Predicted values of MPG 
profit_pred = ml_new.predict(startup_new[['R&D Spend','Administration','Marketing Spend']])
profit_pred
startup_new.head()
# calculating VIF's values of independent variables
rsq_rd = smf.ols('startup_new["R&D Spend"]~startup_new["Administration"]+startup_new["Marketing Spend"]',data=startup_new).fit().rsquared  
vif_rd = 1/(1-rsq_rd) #2.708
rsq_ad = smf.ols('startup_new["Administration"]~startup_new["Marketing Spend"]+startup_new["R&D Spend"]',data=startup_new).fit().rsquared  
vif_ad = 1/(1-rsq_ad) # 1.23
rsq_ms = smf.ols('startup_new["Marketing Spend"]~startup_new["Administration"]+startup_new["R&D Spend"]',data=startup_new).fit().rsquared  
vif_ms = 1/(1-rsq_ms) # 2.68
# Storing vif values in a data frame
d1 = {'Variables':['startup_new["R&D Spend"]','startup_new["Administration"]','startup_new["Marketing Spend"]'],'VIF':[vif_rd,vif_ad,vif_ms]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame 
### All have VIF < 10
# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)
# added varible plot is not showing any significance 
# final model
reci = ((startup_new["Administration"]*startup_new["Administration"])+startup_new["Administration"])/startup_new["Administration"]
final_ml= smf.ols('startup_new["Profit"]~ startup_new["R&D Spend"]+reci+startup_new["Marketing Spend"]',data = startup_new).fit()
final_ml.params
final_ml.summary() # 0.809
# As we can see that r-squared value has increased from 0.960 to 0.962.
profit_pred = final_ml.predict(startup_new)
import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)
######  Linearity #########
# Observed values VS Fitted values
plt.scatter(startup_new["Profit"],profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
# Residuals VS Fitted Values 
plt.scatter(profit_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
########    Normality plot for residuals ######
# histogram
plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed
# QQ plot for residuals 
import pylab          
import scipy.stats as st
# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)
############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(profit_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(startup_new,test_size = 0.2) # 20% size
# preparing the model on train data 
reci_train = ((startup_train["Administration"]*startup_train["Administration"])+startup_train["Administration"])/startup_train["Administration"]
model_train = smf.ols('startup_train["Profit"]~ startup_train["R&D Spend"]+reci_train+startup_train["Marketing Spend"]',data=startup_train).fit()
# train_data prediction
train_pred = model_train.predict(startup_train)
# train residual values 
train_resid  = train_pred - startup_train["Profit"]
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
reci_test = ((startup_test["Administration"]*startup_test["Administration"])+startup_test["Administration"])/startup_test["Administration"]
model_test = smf.ols('startup_test["Profit"]~ startup_test["R&D Spend"]+reci_test+startup_test["Marketing Spend"]',data=startup_test).fit()
# prediction on test data set 
test_pred = model_test.predict(startup_test)
# test residual values 
test_resid  = test_pred - startup_test["Profit"]
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
startup = startup[startup.columns[startup.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
startup = startup.loc[startup.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
startup = startup.fillna(startup.median())
startup['State'].fillna(startup['State'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##startup['column_name'].fillna(startup['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = startup['R&D Spend'].mean () + startup['R&D Spend'].std () * factor   ###211428.38
lower_lim = startup['R&D Spend'].mean () - startup['R&D Spend'].std () * factor   ###-63985.15
startup = startup[(startup['R&D Spend'] < upper_lim) & (startup['R&D Spend'] > lower_lim)]
upper_lim = startup['Administration'].mean () + startup['Administration'].std () * factor  ####205398.05
lower_lim = startup['Administration'].mean () - startup['Administration'].std () * factor  ##37291.23
startup = startup[(startup['Administration'] < upper_lim) & (startup['Administration'] > lower_lim)]
upper_lim = startup['Marketing Spend'].mean () + startup['Marketing Spend'].std () * factor  ###577896.03
lower_lim = startup['Marketing Spend'].mean () - startup['Marketing Spend'].std () * factor  ##-155845.83
startup = startup[(startup['Marketing Spend'] < upper_lim) & (startup['Marketing Spend'] > lower_lim)]
upper_lim = startup['Profit'].mean () + startup['Profit'].std () * factor
lower_lim = startup['Profit'].mean () - startup['Profit'].std () * factor
startup = startup[(startup['Profit'] < upper_lim) & (startup['Profit'] > lower_lim)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim = startup['R&D Spend'].quantile(.95)
lower_lim = startup['R&D Spend'].quantile(.05)
startup = startup[(startup['R&D Spend'] < upper_lim) & (startup['R&D Spend'] > lower_lim)]
upper_lim = startup['Administration'].quantile(.95)
lower_lim = startup['Administration'].quantile(.05)
startup = startup[(startup['Administration'] < upper_lim) & (startup['Administration'] > lower_lim)]
upper_lim = startup['Marketing Spend'].quantile(.95)
lower_lim = startup['Marketing Spend'].quantile(.05)
startup = startup[(startup['Marketing Spend'] < upper_lim) & (startup['Marketing Spend'] > lower_lim)]
upper_lim = startup['Profit'].quantile(.95)
lower_lim = startup['Profit'].quantile(.05)
startup = startup[(startup['Profit'] < upper_lim) & (startup['Profit'] > lower_lim)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
upper_lim = startup['R&D Spend'].quantile(.95)
lower_lim = startup['R&D Spend'].quantile(.05)
startup.loc[(startup['R&D Spend'] > upper_lim)] = upper_lim
startup.loc[(startup['R&D Spend'] < lower_lim)] = lower_lim
upper_lim = startup['Administration'].quantile(.95)
lower_lim = startup['Administration'].quantile(.05)
startup.loc[(startup['Administration'] > upper_lim)] = upper_lim
startup.loc[(startup['Administration'] < lower_lim)] = lower_lim
upper_lim = startup['Marketing Spend'].quantile(.95)
lower_lim = startup['Marketing Spend'].quantile(.05)
startup.loc[(startup['Marketing Spend'] > upper_lim)] = upper_lim
startup.loc[(startup['Marketing Spend'] < lower_lim)] = lower_lim
upper_lim = startup['Profit'].quantile(.95)
lower_lim = startup['Profit'].quantile(.05)
startup.loc[(startup['Profit'] > upper_lim)] = upper_lim
startup.loc[(startup['Profit'] < lower_lim)] = lower_lim
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
startup['bin1'] = pd.cut(startup['R&D Spend'], bins=[0,30000,120000,165349.2], labels=["Low", "Mid", "High"])
startup['bin2'] = pd.cut(startup['Administration'], bins=[51283.14,85000,150000,182645.56], labels=["Low", "Mid", "High"])
startup['bin3'] = pd.cut(startup['Marketing Speed'], bins=[0,100000,400000,471784.1], labels=["Low", "Mid", "High"])
startup['bin4'] = pd.cut(startup['Profit'],bins=[14681.4,70000,150000,192261.83])
conditions = [
    startup['State'].str.contains('California'),
    startup['State'].str.contains('Florida'),
    startup['State'].str.contains('New York')]
choices= ['1','2','3']
startup['choices']=np.select(conditions,choices,default='Other')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
startup = pd.DataFrame({'R&D Spend':[165349.2,162597.7,153441.51,144372.41,142107.34,
                                     131876.9,134615.46,130298.13,120542.52,123334.88,
                                     101913.08,100671.96,93863.75,91992.39,119943.24,
                                     114523.61,78013.11,94657.16,91749.16,86419.7,
                                     76253.86,78389.47,73994.56,67532.53,77044.01,
                                     64664.71,75328.87,72107.6,66051.52,65605.48,
                                     61994.48,61136.38,63408.86,55493.95,46426.07,
                                     46014.02,28663.76,44069.95,20229.59,38558.51,
                                     28754.33,27892.92,23640.93,15505.73,22177.74,1000.23,
                                     1315.46,0,542.05,0]})
startup['log+1'] = (startup['R&D Spend']+1).transform(np.log)
#Negative Values Handling
startup['log'] = (startup['R&D Spend']-startup['R&D Spend'].min()+1).transform(np.log)
startup = pd.DataFrame({'Administration':[136897.8,151377.59,101145.55,118671.85,
                                          91391.77,99814.71,147198.87,145530.06,148718.95,
                                          108679.17,110594.11,91790.61,127320.38,135495.07,
                                          156547.42,122616.84,121597.55,145077.58,114175.79,
                                          153514.11,113867.3,153773.43,122782.75,105751.03,
                                          99281.34,139553.16,144135.98,127864.55,182645.56,
                                          153032.06,115641.28,152701.92,129219.61,103057.49,
                                          157693.92,85047.44,127056.21,51283.14,65947.93,
                                          82982.09,118546.05,84710.77,96189.63,127382.3,
                                          154806.14,124153.04,115816.21,135426.92,51743.15,116983.8]})
startup['log+1'] = (startup['Administration']+1).transform(np.log)
#Negative Values Handling
startup['log'] = (startup['Administration']-startup['Administration'].min()+1).transform(np.log)
startup = pd.DataFrame({'Marketing Spend':[471784.1,443898.53,407934.54,383199.62,366168.42,
                                           362861.36,127716.82,323876.68,311613.29,304981.62,
                                           229160.95,249744.55,249839.44,252664.93,256512.92,
                                           261776.23,264346.06,282574.31,294919.57,0,298664.47,
                                           299737.29,303319.26,304768.73,140574.81,137962.62,
                                           134050.07,353183.81,118148.2,107138.38,91131.24,
                                           88218.23,46085.25,214634.81,210797.67,205517.64,
                                           201126.82,197029.42,185265.1,174999.3,172795.67,
                                           164470.71,148001.11,35534.17,28334.72,1903.93,297114.46,0,0,45173.06]})
startup['log+1'] = (startup['Marketing Spend']+1).transform(np.log)
#Negative Values Handling
startup['log'] = (startup['Marketing Spend']-startup['Marketing Spend'].min()+1).transform(np.log)
startup = pd.DataFrame({'Profit':[192261.83,191792.06,191050.39,182901.99,166187.94,156991.12,
                                  156122.51,155752.6,152211.77,149759.96,146121.95,144259.4,
                                  141585.52,134307.35,132602.65,129917.04,126992.93,125370.37,
                                  124266.9,122776.86,118474.03,111313.02,110352.25,108733.99,
                                  108552.04,107404.34,105733.54,105008.31,103282.38,101004.64,
                                  99937.59,97483.56,97427.84,96778.92,96712.8,96479.51,90708.19,
                                  89949.14,81229.06,81005.76,78239.91,77798.83,71498.49,69758.98,
                                  65200.33,64926.08,49490.75,42559.73,35673.41,14681.4]})
startup['log+1'] = (startup['Profit']+1).transform(np.log)
#Negative Values Handling
startup['log'] = (startup['Profit']-startup['Profit'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(startup['State'])
startup = startup.join(encoded_columns).drop('State', axis=1)                                     
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = startup.groupby('Profit')
sums = grouped['Profit'].sum().add_suffix('_sum')
avgs = grouped['Profit'].mean().add_suffix('_avg')
####Categorical Column grouping
startup.groupby('Profit').agg(lambda x: x.value_counts().index[0])
#Pivot table Pandas Example
startup.pivot_table(index='Profit', columns='New York', values='Marketing Spend', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='Florida', values='Marketing Spend', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='California', values='Marketing Spend', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='New York', values='Administration', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='Florida', values='Administration', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='California', values='Administration', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='New York', values='R&D Spend', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='Florida', values='R&D Spend', aggfunc=np.sum, fill_value = 0)
startup.pivot_table(index='Profit', columns='California', values='R&D Spend', aggfunc=np.sum, fill_value = 0)
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(startup.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(startup.iloc[:,0:9])
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
X = startup.iloc[:,0:4]
Y = startup['Profit']
X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 10)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=10).fit(X_Train,Y_Train)
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
PCA_df = pd.concat([PCA_df, startup['Profit']], axis = 1)
PCA_df['Profit'] = LabelEncoder().fit_transform(PCA_df['Profit'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
Profit = [49, 48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,
          29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
colors = ['r', 'b']
for Profit, color in zip(Profit, colors):
    plt.scatter(PCA_df.loc[PCA_df['Profit'] == Profit, 'PC1'], 
                PCA_df.loc[PCA_df['Profit'] == Profit, 'PC2'], 
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