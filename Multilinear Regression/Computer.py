# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# reading a csv file using pandas library
comp=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Multilinear Regression\\Assignments\\Computer_Data.csv")
comp.columns
comp.drop(["Unnamed: 0"],axis=1)
#####Exploratory Data Analysis#########################################################
comp.mean() ## price - 2219.576610, speed-52.011024,hd-416.601694,ram-8.286947,screen-14.608723,ads-221.301007,trend-15.926985
comp.median() ###  price-2144.0, speed-50.0,hd-340.0,ram-8.0,screen-14.0,ads-246.0,trend-16.0
comp.mode() 
####Measures of Dispersion
comp.var() 
comp.std() ####  price-580.803956, speed-21.157735,hd-258.548445,ram-5.631099,screen-0.905115,ads-74.835284,trend-7.873984 
#### Calculate the range value
range1 = max(comp['price'])-min(comp['price'])  ### 4450 
range2 = max(comp['speed'])-min(comp['speed']) ### 75
range3 = max(comp['hd'])-min(comp['hd']) ### 2020
range4 = max(comp['ram'])-min(comp['ram']) ### 30
range5 = max(comp['screen'])-min(comp['screen']) ## 3
range6 = max(comp['ads'])-min(comp['ads'])  ## 300
range7 = max(comp['trend'])-min(comp['trend'])  ###34
### Calculate skewness and Kurtosis
comp.skew() 
comp.kurt() 
#### Visualizations
plt.hist(comp["price"])
plt.hist(comp["speed"])
plt.hist(comp["hd"])
plt.hist(comp["ram"])
plt.hist(comp["screen"])
plt.hist(comp["ads"])
plt.hist(comp["trend"])
plt.boxplot(comp["price"],0,"rs",0)
plt.boxplot(comp["speed"],0,"rs",0)
plt.boxplot(comp["hd"],0,"rs",0)
plt.boxplot(comp["ram"],0,"rs",0)
plt.boxplot(comp["screen"],0,"rs",0)
plt.boxplot(comp["ads"],0,"rs",0)
plt.boxplot(comp["trend"],0,"rs",0)
plt.plot(comp["speed"],comp["price"],"bo");plt.xlabel("speed");plt.ylabel("price")
plt.plot(comp["hd"],comp["price"],"bo");plt.xlabel("hd");plt.ylabel("price")
plt.plot(comp["ram"],comp["price"],"bo");plt.xlabel("ram");plt.ylabel("price")
plt.plot(comp["screen"],comp["price"],"bo");plt.xlabel("screen");plt.ylabel("price")
plt.plot(comp["ads"],comp["price"],"bo");plt.xlabel("ads");plt.ylabel("price")
plt.plot(comp["trend"],comp["price"],"bo");plt.xlabel("trend");plt.ylabel("price")
# table 
pd.crosstab(comp["speed"],comp["price"])
pd.crosstab(comp["hd"],comp["price"])
pd.crosstab(comp["ram"],comp["price"])
pd.crosstab(comp["screen"],comp["price"])
pd.crosstab(comp["ads"],comp["price"])
pd.crosstab(comp["trend"],comp["price"])
## Barplot
pd.crosstab(comp["speed"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["hd"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ram"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["screen"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ads"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["price"]).plot(kind = "bar",width=1.85)
import seaborn as sns 
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="speed",y="price",data=comp)
sns.boxplot(x="hd",y="price",data=comp)
sns.boxplot(x="ram",y="price",data=comp)
sns.boxplot(x="screen",y="price",data=comp)
sns.boxplot(x="ads",y="price",data=comp)
sns.boxplot(x="trend",y="price",data=comp)
sns.pairplot(comp.iloc[:,0:7]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(comp,hue="price",size=5)
comp["speed"].value_counts()
comp["hd"].value_counts()
comp["ram"].value_counts()
comp["screen"].value_counts()
comp["ads"].value_counts()
comp["trend"].value_counts()
comp["speed"].value_counts().plot(kind="pie")
comp["hd"].value_counts().plot(kind="pie")
comp["ram"].value_counts().plot(kind="pie")
comp["screen"].value_counts().plot(kind="pie")
comp["ads"].value_counts().plot(kind="pie")
comp["trend"].value_counts().plot(kind="pie")
sns.pairplot(comp,hue="price",size=4,diag_kind = "kde")
sns.FacetGrid(comp,hue="price").map(plt.scatter,"speed","price").add_legend()
sns.FacetGrid(comp,hue="price").map(plt.scatter,"hd","price").add_legend()
sns.FacetGrid(comp,hue="price").map(plt.scatter,"ram","price").add_legend()
sns.FacetGrid(comp,hue="price").map(plt.scatter,"screen","price").add_legend()
sns.FacetGrid(comp,hue="price").map(plt.scatter,"ads","price").add_legend()
sns.FacetGrid(comp,hue="price").map(plt.scatter,"trend","price").add_legend()
sns.FacetGrid(comp,hue="price").map(sns.kdeplot,"speed").add_legend()
sns.FacetGrid(comp,hue="price").map(sns.kdeplot,"hd").add_legend()
sns.FacetGrid(comp,hue="price").map(sns.kdeplot,"ram").add_legend()
sns.FacetGrid(comp,hue="price").map(sns.kdeplot,"screen").add_legend()
sns.FacetGrid(comp,hue="price").map(sns.kdeplot,"ads").add_legend()
sns.FacetGrid(comp,hue="price").map(sns.kdeplot,"trend").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(comp['speed'], dist="norm",plot=pylab)
stats.probplot(comp['hd'],dist="norm",plot=pylab)
stats.probplot(comp['ram'],dist="norm",plot=pylab)
stats.probplot(comp['screen'],dist="norm",plot=pylab)
stats.probplot(comp['ads'],dist="norm",plot=pylab)
stats.probplot(comp['trend'],dist="norm",plot=pylab)
# ppf => Percent point function 
####speed
stats.norm.ppf(0.975,52.011024,21.157735)# similar to qnorm in R ---- 93.48
# cdf => cumulative distributive function 
stats.norm.cdf(comp["speed"],52.011024,21.157735) # similar to pnorm in R 
#### price
stats.norm.ppf(0.975,2219.576610,580.803956)# similar to qnorm in R ---- 6421.739
# cdf => cumulative distributive function 
stats.norm.cdf(comp["price"],2219.576610,580.803956) # similar to pnorm in R 
#### hd
stats.norm.ppf(0.975,416.601694,258.548445)# similar to qnorm in R ---- 1082.9894
# cdf => cumulative distributive function 
stats.norm.cdf(comp["hd"],416.601694,258.548445) # similar to pnorm in R 
#### ram
stats.norm.ppf(0.975, 8.286947, 5.631099)# similar to qnorm in R ---- 23.966
# cdf => cumulative distributive function 
stats.norm.cdf(comp["ram"], 8.286947, 5.631099) # similar to pnorm in R 
#### screen
stats.norm.ppf(0.975, 14.608723, 0.905115)# similar to qnorm in R ---- 23.966
# cdf => cumulative distributive function 
stats.norm.cdf(comp["screen"], 14.608723, 0.905115) # similar to pnorm in R 
###### ads
stats.norm.ppf(0.975, 221.301007, 74.835284)# similar to qnorm in R ---- 23.966
# cdf => cumulative distributive function 
stats.norm.cdf(comp["ads"], 221.301007, 74.835284)
### trend
stats.norm.ppf(0.975, 15.926985, 7.873984)# similar to qnorm in R ---- 23.966
# cdf => cumulative distributive function 
stats.norm.cdf(comp["trend"], 15.926985, 7.873984)
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
comp.corr(method = "pearson")
comp.corr(method = "kendall")
comp["price"].corr(comp["speed"]) # # correlation value between X and Y -- 0.3009
comp["price"].corr(comp["hd"])  ### 0.43
comp["price"].corr(comp["ram"]) ### 0.622
comp["price"].corr(comp["screen"])  ## 0.29
comp["price"].corr(comp["ads"])  ###  0.05
comp["price"].corr(comp["trend"])   ## -0.199
np.corrcoef(comp["speed"],comp["price"])
np.corrcoef(comp["hd"],comp["price"])
np.corrcoef(comp["ram"],comp["price"])
np.corrcoef(comp["screen"],comp["price"])
np.corrcoef(comp["ads"],comp["price"])
np.corrcoef(comp["trend"],comp["price"])
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(comp['speed'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(comp['hd'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(comp['ram'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(comp['screen'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(comp['ads'])
normalized_E = preprocessing.normalize([e_array])
f_array = np.array(comp['trend'])
normalized_F = preprocessing.normalize([f_array])
g_array = np.array(comp['price'])
normalized_G = preprocessing.normalize([g_array])
# to get top 6 rows
comp.head(10) # to get top n rows use cars.head(10)
comp.tail(10)
# Correlation matrix 
comp.corr()
# We see there exists High collinearity between input variables especially between
# [Profit & R&D Spend] , [Profit & Marketing Spend],[R&D Spend & Marketing Spend]
## so there exists collinearity problem
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(comp)
pd.tools.plotting.scatter_matrix(comp) ##-> also used for plotting all in one graph
#### preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
# Preparing model                  
ml1 = smf.ols('comp["price"]~ comp["speed"]+comp["hd"]+comp["ram"]+comp["screen"]+comp["ads"]+comp["trend"]',data=comp).fit() # regression model
# Getting coefficients of variables               
ml1.params
ml1.summary()
##All coefficients are significant having p-value < 0.05 and R-squared = 0.712
### Lets do some transformation
ml1_up = smf.ols('comp["price"]~ comp["speed"]+comp["hd"]+comp["ram"]+comp["screen"]+np.sqrt(comp["ads"])+comp["trend"]',data=comp).fit() # regression model
# Getting coefficients of variables               
ml1_up.params
ml1_up.summary()
### After doing transformation R-squared value increased to 0.713
# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1_up)
# index 1441,1701,3784,4478,5346,5435,5961,6253 is showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals
comp_new=comp.drop(comp.index[[1441,1701,3784,4478,5346,5435,5961,6253]],axis=0)
ml_upnew = smf.ols('comp_new["price"]~ comp_new["speed"]+comp_new["hd"]+comp_new["ram"]+comp_new["screen"]+np.sqrt(comp_new["ads"])+comp_new["trend"]',data = comp_new).fit()    
# Getting coefficients of variables        
ml_upnew.params
# Summary
ml_upnew.summary() # 0.714
### After removing influential variables R-squared value increased to 0.714
# Confidence values 99%
print(ml_upnew.conf_int(0.01)) # 99% confidence level
# Predicted values of MPG 
price_pred = ml_upnew.predict(comp_new[['speed','hd','ram','screen','ads','trend']])
price_pred
comp_new.head()
# calculating VIF's values of independent variables
rsq_sp = smf.ols('comp_new["speed"]~comp_new["hd"]+comp_new["ram"]+comp_new["screen"]+np.sqrt(comp_new["ads"])+comp_new["trend"]',data=comp_new).fit().rsquared  
vif_sp = 1/(1-rsq_sp) #1.26
rsq_hd = smf.ols('comp_new["hd"]~comp_new["speed"]+comp_new["ram"]+comp_new["screen"]+np.sqrt(comp_new["ads"])+comp_new["trend"]',data=comp_new).fit().rsquared  
vif_hd = 1/(1-rsq_hd) # 4.13
rsq_ram = smf.ols('comp_new["ram"]~comp_new["speed"]+comp_new["hd"]+comp_new["screen"]+np.sqrt(comp_new["ads"])+comp_new["trend"]',data=comp_new).fit().rsquared  
vif_ram = 1/(1-rsq_ram) # 2.88
rsq_sc = smf.ols('comp_new["screen"]~comp_new["speed"]+comp_new["ram"]+comp_new["hd"]+np.sqrt(comp_new["ads"])+comp_new["trend"]',data=comp_new).fit().rsquared  
vif_sc = 1/(1-rsq_sc) # 1.078
rsq_ads = smf.ols('comp_new["ads"]~comp_new["speed"]+comp_new["ram"]+comp_new["screen"]+np.sqrt(comp_new["hd"])+comp_new["trend"]',data=comp_new).fit().rsquared  
vif_ads = 1/(1-rsq_ads) # 1.14
rsq_tre = smf.ols('comp_new["trend"]~comp_new["speed"]+comp_new["ram"]+comp_new["screen"]+np.sqrt(comp_new["ads"])+comp_new["hd"]',data=comp_new).fit().rsquared  
vif_tre = 1/(1-rsq_tre) # 1.82
# Storing vif values in a data frame
d1 = {'Variables':['comp_new["speed"]','comp_new["hd"]','comp_new["ram"]','comp_new["screen"]','comp_new["ads"]','comp_new["trend"]'],
      'VIF':[vif_sp,vif_hd,vif_ram,vif_sc,vif_ads,vif_tre]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame 
### All have VIF < 10
# Added varible plot 
sm.graphics.plot_partregress_grid(ml_upnew)
# added varible plot is not showing any significance 
# final model
final_ml= smf.ols('comp_new["price"]~ comp_new["speed"]+comp_new["hd"]+comp_new["ram"]+comp_new["screen"]+np.sqrt(comp_new["ads"])+comp_new["trend"]',data = comp_new).fit()
final_ml.params
final_ml.summary() # 0.714
price_pred = final_ml.predict(comp_new)
import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)
######  Linearity #########
# Observed values VS Fitted values
plt.scatter(comp_new["price"],price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
# Residuals VS Fitted Values 
plt.scatter(price_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
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
plt.scatter(price_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
comp_train,comp_test  = train_test_split(comp_new,test_size = 0.2) # 20% size
# preparing the model on train data 
model_train = smf.ols('comp_train["price"]~ comp_train["speed"]+comp_train["hd"]+comp_train["ram"]+comp_train["screen"]+np.sqrt(comp_train["ads"])+comp_train["trend"]',data=comp_train).fit()
# train_data prediction
train_pred = model_train.predict(comp_train)
# train residual values 
train_resid  = train_pred - comp_train["price"]
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
model_test = smf.ols('comp_test["price"]~ comp_test["speed"]+comp_test["hd"]+comp_test["ram"]+comp_test["screen"]+np.sqrt(comp_test["ads"])+comp_test["trend"]',data=comp_test).fit()
# prediction on test data set 
test_pred = model_test.predict(comp_test)
# test residual values 
test_resid  = test_pred - comp_test["price"]
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
comp = comp[comp.columns[comp.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
comp = comp.loc[comp.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
comp = comp.fillna(comp.median())
comp['State'].fillna(comp['State'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##comp['column_name'].fillna(comp['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = comp['price'].mean () + comp['price'].std () * factor   ### 3961.988
lower_lim = comp['price'].mean () - comp['price'].std () * factor   ### 477.165
comp = comp[(comp['price'] < upper_lim) & (comp['price'] > lower_lim)]
upper_lim = comp['speed'].mean () + comp['speed'].std () * factor  #### 115.42
lower_lim = comp['speed'].mean () - comp['speed'].std () * factor  ## -11.59
comp = comp[(comp['speed'] < upper_lim) & (comp['speed'] > lower_lim)]
upper_lim = comp['hd'].mean () + comp['hd'].std () * factor  ### 1187.28
lower_lim = comp['hd'].mean () - comp['hd'].std () * factor  ## -357.38
comp = comp[(comp['hd'] < upper_lim) & (comp['hd'] > lower_lim)]
upper_lim = comp['ram'].mean () + comp['ram'].std () * factor  ### 25.05
lower_lim = comp['ram'].mean () - comp['ram'].std () * factor  ##-155845.83
comp = comp[(comp['ram'] < upper_lim) & (comp['ram'] > lower_lim)]
upper_lim = comp['screen'].mean () + comp['screen'].std () * factor  ###577896.03
lower_lim = comp['screen'].mean () - comp['screen'].std () * factor  ##-155845.83
comp = comp[(comp['screen'] < upper_lim) & (comp['screen'] > lower_lim)]
upper_lim = comp['ads'].mean () + comp['ads'].std () * factor  ###577896.03
lower_lim = comp['ads'].mean () - comp['ads'].std () * factor  ##-155845.83
comp = comp[(comp['ads'] < upper_lim) & (comp['ads'] > lower_lim)]
upper_lim = comp['trend'].mean () + comp['trend'].std () * factor  ###577896.03
lower_lim = comp['trend'].mean () - comp['trend'].std () * factor  ##-155845.83
comp = comp[(comp['trend'] < upper_lim) & (comp['trend'] > lower_lim)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim = comp['price'].quantile(.95)
lower_lim = comp['price'].quantile(.05)
comp = comp[(comp['price'] < upper_lim) & (comp['price'] > lower_lim)]
upper_lim = comp['speed'].quantile(.95)
lower_lim = comp['speed'].quantile(.05)
comp = comp[(comp['speed'] < upper_lim) & (comp['speed'] > lower_lim)]
upper_lim = comp['hd'].quantile(.95)
lower_lim = comp['hd'].quantile(.05)
comp = comp[(comp['hd'] < upper_lim) & (comp['hd'] > lower_lim)]
upper_lim = comp['ram'].quantile(.95)
lower_lim = comp['ram'].quantile(.05)
comp = comp[(comp['ram'] < upper_lim) & (comp['ram'] > lower_lim)]
upper_lim = comp['screen'].quantile(.95)
lower_lim = comp['screen'].quantile(.05)
comp = comp[(comp['screen'] < upper_lim) & (comp['screen'] > lower_lim)]
upper_lim = comp['ads'].quantile(.95)
lower_lim = comp['ads'].quantile(.05)
comp = comp[(comp['ads'] < upper_lim) & (comp['ads'] > lower_lim)]
upper_lim = comp['trend'].quantile(.95)
lower_lim = comp['trend'].quantile(.05)
comp = comp[(comp['trend'] < upper_lim) & (comp['trend'] > lower_lim)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
comp.loc[(comp['price'] > upper_lim)] = upper_lim
comp.loc[(comp['price'] < lower_lim)] = lower_lim
comp.loc[(comp['speed'] > upper_lim)] = upper_lim
comp.loc[(comp['speed'] < lower_lim)] = lower_lim
comp.loc[(comp['hd'] > upper_lim)] = upper_lim
comp.loc[(comp['hd'] < lower_lim)] = lower_lim
comp.loc[(comp['ram'] > upper_lim)] = upper_lim
comp.loc[(comp['ram'] < lower_lim)] = lower_lim
comp.loc[(comp['screen'] > upper_lim)] = upper_lim
comp.loc[(comp['screen'] < lower_lim)] = lower_lim
comp.loc[(comp['ads'] > upper_lim)] = upper_lim
comp.loc[(comp['ads'] < lower_lim)] = lower_lim
comp.loc[(comp['trend'] > upper_lim)] = upper_lim
comp.loc[(comp['trend'] < lower_lim)] = lower_lim
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
comp['bin1'] = pd.cut(comp['price'], bins=[949,2000,4000,5399], labels=["Low", "Mid", "High"])
comp['bin2'] = pd.cut(comp['speed'], bins=[25,50,75,100], labels=["Low", "Mid", "High"])
comp['bin3'] = pd.cut(comp['hd'], bins=[80,700,1400,2100], labels=["Low", "Mid", "High"])
comp['bin4'] = pd.cut(comp['ram'],bins=[2,12,22,32],labels=["Low", "Mid", "High"])
comp['bin5'] = pd.cut(comp['screen'],bins=[14,15,16,17],labels=["Low", "Mid", "High"])
comp['bin6'] = pd.cut(comp['ads'],bins=[39,139,239,339],labels=["Low", "Mid", "High"])
comp['bin7'] = pd.cut(comp['trend'],bins=[1,14,25,35],labels=["Low", "Mid", "High"])
conditions = [
    comp['cd'].str.contains('no'),
    comp['cd'].str.contains('yes')]
choices= ['1','2']
comp['choices']=np.select(conditions,choices,default='Other')
conditions1 = [
    comp['multi'].str.contains('no'),
    comp['multi'].str.contains('yes')]
choices1= ['1','2']
comp['choices']=np.select(conditions1,choices1,default='Other')
conditions2 = [
    comp['premium'].str.contains('no'),
    comp['premium'].str.contains('yes')]
choices2= ['1','2']
comp['choices']=np.select(conditions2,choices2,default='Other')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
comp = pd.DataFrame({'price':comp.iloc[:,1]})
comp['log+1'] = (comp['price']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['price']-comp['price'].min()+1).transform(np.log)
comp = pd.DataFrame({'speed':comp.iloc[:,2]})
comp['log+1'] = (comp['speed']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['speed']-comp['speed'].min()+1).transform(np.log)
comp = pd.DataFrame({'hd':comp.iloc[:,3]})
comp['log+1'] = (comp['hd']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['hd']-comp['hd'].min()+1).transform(np.log)
comp = pd.DataFrame({'ram':comp.iloc[:,4]})
comp['log+1'] = (comp['ram']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['ram']-comp['ram'].min()+1).transform(np.log)
comp = pd.DataFscreene({'screen':comp.iloc[:,5]})
comp['log+1'] = (comp['screen']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['screen']-comp['screen'].min()+1).transform(np.log)
comp = pd.DataFadse({'ads':comp.iloc[:,9]})
comp['log+1'] = (comp['ads']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['ads']-comp['ads'].min()+1).transform(np.log)
comp = pd.DataFtrende({'trend':comp.iloc[:,10]})
comp['log+1'] = (comp['trend']+1).transform(np.log)
#Negative Values Handling
comp['log'] = (comp['trend']-comp['trend'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(comp['multi'])
comp = comp.join(encoded_columns.add_suffix('_multi')).drop('multi', axis=1)      
encoded_columns_1 = pd.get_dummies(comp['premium'])
comp = comp.join(encoded_columns_1.add_suffix('_premium')).drop('premium', axis=1)  
encoded_columns_2 = pd.get_dummies(comp['cd'])
comp = comp.join(encoded_columns_2.add_suffix('_cd')).drop('cd', axis=1)                              
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = comp.groupby('price')
sums = grouped['price'].sum().add_suffix('_sum')
avgs = grouped['price'].mean().add_suffix('_avg')
####Categorical Column grouping
comp.groupby('price').agg(lambda x: x.value_counts().index[0]).drop(["Unnamed: 0"],axis=1)
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(comp.iloc[:,0:2])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(comp.iloc[:,0:9])
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
X1 = comp.iloc[:,2:6]
X2 = comp.iloc[:,9:11]
X = pd.concat([X1,X2],axis=1)
Y = comp['price']
##X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 6259)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=6259).fit(X_Train,Y_Train)
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
PCA_df = pd.concat([PCA_df, comp['price']], axis = 1)
PCA_df['price'] = LabelEncoder().fit_transform(PCA_df['price'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
price = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for price, color in zip(price, colors):
    plt.scatter(PCA_df.loc[PCA_df['price'] == price, 'PC1'], 
                PCA_df.loc[PCA_df['price'] == price, 'PC2'], 
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
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=3000)
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
X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=6259)
autoencoder.fit(X1, Y1,epochs=6259,batch_size=6259,shuffle=True,verbose = 7000,validation_data=(X2, Y2))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)