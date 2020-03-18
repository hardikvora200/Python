import pandas as pd
import numpy as np
import seaborn as sns
##import scipy.stats as st
import matplotlib.pyplot as plt# reading a csv file using pandas library
plastic=pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\PlasticSales.xlsx")
##plastic1.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
plastic.columns 
plastic['month'] = pd.DatetimeIndex(plastic['Month']).month
plastic.head()
month = pd.get_dummies(plastic['month'], prefix_sep='_')
plastic1 = pd.concat([plastic,month],axis=1)
plastic1.rename(columns={1: "Jan", 2: "Feb", 3: "Mar",4: "Apr",5: "may",6: "June",
                         7: "July",8: "Aug", 9: "Sep",10: "Oct",11: "Nov",12: "Dec"}, inplace = True, errors="raise")
plastic1["t"] = np.arange(1,61)
plastic1["t_square"] = plastic1["t"]*plastic1["t"]
plastic1["log_Sales"] = np.log(plastic1["Sales"])
plastic1.Sales.plot()
##plastic1.drop(["a"],axis=1,inplace=True)
##plastic1.columns
# To get the count of null values in the data 
plastic1.isnull().sum()
plastic1.isna()
plastic1.shape # 60 18 => Before dropping null values
# To drop null values ( dropping rows)
plastic1.dropna().shape # 96 17 => After dropping null values
#####Exploratory Data Analysis#########################################################
plastic1.mean() #### Sales - 1162.366667, month - 6.500, (Jan -Dec) - 0.083333,t- 30.5, t_square -1230.166667, log_Sales - 7.031121
plastic1.median() 
plastic1.mode() 
####Measures of Dispersion
plastic1.var() 
plastic1.std() #### Sales - 266.431469, month - 3.481184,(Jan-Dec)-0.278718,t-17.464249,t_square - 1099.1013, log_Sales - 0.238154  
#### Calculate the range value
range1 = max(plastic1['Sales'])-min(plastic1['Sales'])  ### 940
range2 = max(plastic1['log_Sales'])-min(plastic1['log_Sales']) ### 0.854
range3 = max(plastic1['t'])-min(plastic1['t']) ### 95
range4 = max(plastic1['t_square'])-min(plastic1['t_square'])  ##  9215
### Calculate skewness and Kurtosis
plastic1.skew()
plastic1.kurt() 
####Graphidelivery_time Representation 
plt.hist(plastic1["Sales"])
plt.hist(plastic1["log_Sales"])
plt.hist(plastic1["t"])
plt.hist(plastic1["t_square"])
plt.boxplot(plastic1["Sales"],0,"rs",0)
plt.boxplot(plastic1["log_Sales"],0,"rs",0)
plt.boxplot(plastic1["t"],0,"rs",0)
plt.boxplot(plastic1["t_square"],0,"rs",0)
plt.plot(plastic1["log_Sales"],plastic1["Sales"],"bo");plt.xlabel("log_Sales");plt.ylabel("Sales")
plt.plot(plastic1["t"],plastic1["Sales"],"bo");plt.xlabel("t");plt.ylabel("Sales")
plt.plot(plastic1["t_square"],plastic1["Sales"],"bo");plt.xlabel("t_square");plt.ylabel("Sales")
# table 
pd.crosstab(plastic1["log_Sales"],plastic1["Sales"])
pd.crosstab(plastic1["t"],plastic1["Sales"])
pd.crosstab(plastic1["t_square"],plastic1["Sales"])
## Barplot
pd.crosstab(plastic1["log_Sales"],plastic1["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(plastic1["t"],plastic1["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(plastic1["t_square"],plastic1["Sales"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Sales",data=plastic1,palette="hls")
sns.countplot(x="log_Sales",data=plastic1,palette="hls")
sns.countplot(x="t",data=plastic1,palette="hls")
sns.countplot(x="t_square",data=plastic1,palette="hls")
sns.boxplot(x="log_Sales",y="Sales",data=plastic1,palette="hls")
sns.boxplot(x="t",y="Sales",data=plastic1,palette="hls")
sns.boxplot(x="t_square",y="Sales",data=plastic1,palette="hls")
sns.pairplot(plastic1.iloc[:,0:17]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(plastic1,hue="Sales",size=2)
plastic1["Sales"].value_counts()
plastic1["log_Sales"].value_counts()
plastic1["t"].value_counts()
plastic1["t_square"].value_counts()
plastic1["Sales"].value_counts().plot(kind = "pie")
plastic1["log_Sales"].value_counts().plot(kind = "pie")
plastic1["t"].value_counts().plot(kind = "pie")
plastic1["t_square"].value_counts().plot(kind = "pie")
sns.pairplot(plastic1,hue="Sales",size=4,diag_kind = "kde")
sns.pairplot(plastic1,hue="log_Sales",size=4,diag_kind = "kde")
sns.pairplot(plastic1,hue="t",size=4,diag_kind = "kde")
sns.pairplot(plastic1,hue="t_square",size=4,diag_kind = "kde")
sns.FacetGrid(plastic1,hue="Sales").map(plt.scatter,"log_Sales","Sales").add_legend()
sns.FacetGrid(plastic1,hue="Sales").map(plt.scatter,"t","Sales").add_legend()
sns.FacetGrid(plastic1,hue="Sales").map(plt.scatter,"t_square","Sales").add_legend()
sns.FacetGrid(plastic1,hue="log_Sales").map(plt.scatter,"Sales","log_Sales").add_legend()
sns.FacetGrid(plastic1,hue="log_Sales").map(plt.scatter,"t","log_Sales").add_legend()
sns.FacetGrid(plastic1,hue="log_Sales").map(plt.scatter,"t_square","log_Sales").add_legend()
sns.FacetGrid(plastic1,hue="t").map(plt.scatter,"Sales","t").add_legend()
sns.FacetGrid(plastic1,hue="t").map(plt.scatter,"log_Sales","t").add_legend()
sns.FacetGrid(plastic1,hue="t").map(plt.scatter,"t_square","t").add_legend()
sns.FacetGrid(plastic1,hue="t_square").map(plt.scatter,"Sales","t_square").add_legend()
sns.FacetGrid(plastic1,hue="t_square").map(plt.scatter,"log_Sales","t_square").add_legend()
sns.FacetGrid(plastic1,hue="t_square").map(plt.scatter,"t","t_square").add_legend()
sns.FacetGrid(plastic1,hue="Sales").map(sns.kdeplot,"log_Sales").add_legend()
sns.FacetGrid(plastic1,hue="Sales").map(sns.kdeplot,"t").add_legend()
sns.FacetGrid(plastic1,hue="Sales").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(plastic1,hue="log_Sales").map(sns.kdeplot,"Sales").add_legend()
sns.FacetGrid(plastic1,hue="log_Sales").map(sns.kdeplot,"t").add_legend()
sns.FacetGrid(plastic1,hue="log_Sales").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(plastic1,hue="t").map(sns.kdeplot,"Sales").add_legend()
sns.FacetGrid(plastic1,hue="t").map(sns.kdeplot,"log_Sales").add_legend()
sns.FacetGrid(plastic1,hue="t").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(plastic1,hue="t_square").map(sns.kdeplot,"Sales").add_legend()
sns.FacetGrid(plastic1,hue="t_square").map(sns.kdeplot,"log_Sales").add_legend()
sns.FacetGrid(plastic1,hue="t_square").map(sns.kdeplot,"t").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab
# Checking Whether data is normally distributed
stats.probplot(plastic1['Sales'],dist="norm",plot=pylab)
stats.probplot(np.log(plastic1['Sales']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(plastic1['Sales']),dist="norm",plot=pylab)
stats.probplot((plastic1['Sales'] * plastic1['Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['Sales'])*np.exp(plastic1['Sales']),dist="norm",plot=pylab)
reci_1=1/plastic1['Sales']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((plastic1['Sales'] * plastic1['Sales'])+plastic1['Sales']),dist="norm",plot=pylab)
stats.probplot(plastic1['log_Sales'],dist="norm",plot=pylab)
stats.probplot(np.log(plastic1['log_Sales']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(plastic1['log_Sales']),dist="norm",plot=pylab)
stats.probplot((plastic1['log_Sales'] * plastic1['log_Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['log_Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['log_Sales'])*np.exp(plastic1['log_Sales']),dist="norm",plot=pylab)
reci_2=1/plastic1['log_Sales']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((plastic1['log_Sales'] * plastic1['log_Sales'])+plastic1['log_Sales']),dist="norm",plot=pylab)
stats.probplot(plastic1['t'],dist="norm",plot=pylab)
stats.probplot(np.log(plastic1['t']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(plastic1['t']),dist="norm",plot=pylab)
stats.probplot((plastic1['t'] * plastic1['t']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['t']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['t'])*np.exp(plastic1['t']),dist="norm",plot=pylab)
reci_3=1/plastic1['t']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((plastic1['t'] * plastic1['t'])+plastic1['t']),dist="norm",plot=pylab)
stats.probplot(plastic1['t_square'],dist="norm",plot=pylab)
stats.probplot(np.log(plastic1['t_square']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(plastic1['t_square']),dist="norm",plot=pylab)
stats.probplot((plastic1['t_square'] * plastic1['t_square']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['t_square']),dist="norm",plot=pylab)
stats.probplot(np.exp(plastic1['t_square'])*np.exp(plastic1['t_square']),dist="norm",plot=pylab)
reci_4=1/plastic1['t_square']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((plastic1['t_square'] * plastic1['t_square'])+plastic1['t_square']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Sales
stats.norm.ppf(0.975,1162.366667,266.431469)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(plastic1["Sales"],1162.366667,266.431469) # similar to pnorm in R 
#### log_Sales
stats.norm.ppf(0.975,7.031121,0.238154)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(plastic1["log_Sales"],7.031121,0.238154) # similar to pnorm in R 
#### t
stats.norm.ppf(0.975,30.5,17.464249)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(plastic1["t"],30.5,17.464249) # similar to pnorm in R 
#### t_square
stats.norm.ppf(0.975, 1230.166667,1099.1013)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(plastic1["t_square"],1230.166667,1099.1013) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
plastic1.corr(method = "pearson")
plastic1.corr(method = "kendall")
# to get top 6 rows
plastic1.head(40) # to get top n rows use cars.head(10)
plastic1.tail(10)
##Normalization
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
### Normalized data frame (considering the numerical part of data)
df_norm = norm_func(plastic1.iloc[:,0:])
# Scatter plot between the variables along with histograms
##plastic1.drop(['Month'],axis=1,inplace=True)
sns.pairplot(plastic1)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
##import statsmodels.tsa.statespace as tm_models
#from sm.tsa.statespace import sa
# Converting the normal index of plastic1 to time stamp 
plastic1.index = pd.to_datetime(plastic1.Month,format="%")
plastic1.Sales.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
###plastic1["Date"] = pd.to_datetime(plastic1.Month,format="%b-%y")
plastic1["Date"] = pd.DatetimeIndex(plastic1["Month"]).date ## date extraction
# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 
plastic1["month"] = pd.DatetimeIndex(plastic1["Month"]).month # month extraction
plastic1["year"] = pd.DatetimeIndex(plastic1["Month"]).year # year extraction
# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=plastic1,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")
# Boxplot for ever
sns.boxplot(x="month",y="Sales",data=plastic1)
sns.boxplot(x="year",y="Sales",data=plastic1)
# Line plot for Ridership based on year  and for each month
sns.lineplot(x="year",y="Sales",hue="month",data=plastic1)
# moving average for the time series to understand better about the trend character in Amtrak
plastic1.Sales.plot(label="org")
for i in range(2,27,6):
    plastic1["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic1.Sales,model="additive",period=3)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(plastic1.Sales,model="multiplicative",period=4)
decompose_ts_mul.plot()
# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(plastic1.Sales,lags=10)
tsa_plots.plot_pacf(plastic1.Sales,lags=10)
# lets verify on Train and Test data 
Train = plastic1.head(48)
Test = plastic1.tail(12)# to change the index value in pandas data frame 
# Creating a function to calculate the MAPE value for test data 
##plastic1.drop(['Month'],axis=1,inplace=True)
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)
##############################SMOOTHING TECHNIQUES ##########################################
################################### Simple Exponential Method#######################################################
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
pred_ses_int=pd.Series(pred_ses).astype(int)
MAPE(pred_ses_int,Test.Sales) # 17.04
###################################### Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
pred_hw_int=pd.Series(pred_hw).astype(int)
MAPE(pred_hw_int,Test.Sales) # 102.03
#################################### Holts winter exponential smoothing ##########################
hwe_model = ExponentialSmoothing(Train["Sales"],seasonal_periods=12).fit()
pred_hwe = hwe_model.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_int=pd.Series(pred_hwe).astype(int)
MAPE(pred_hwe_int,Test.Sales)   
#################################### Holts winter exponential smoothing with additive seasonality and additive trend##########################
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_add_add_int=pd.Series(pred_hwe_add_add).astype(int)
MAPE(pred_hwe_add_add_int,Test.Sales) # 14.71
################################## Holts winter exponential smoothing with multiplicative seasonality and additive trend####################
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_mul_add_int=pd.Series(pred_hwe_mul_add).astype(int)
MAPE(pred_hwe_mul_add_int,Test.Sales) # 14.83
# Lets us use auto_arima from p
from pmdarima.arima import auto_arima
auto_arima_model = auto_arima(Train["Sales"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)    #### Time - 0.281 seconds, Total fit time: 109.464 seconds
auto_arima_model.summary() # SARIMAX(0, 1, 1)x(1, 1, 1, 12)
# AIC ==> 362.145, BIC ==> 368.367, HQIC ==> 364.293 
# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )
# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=12))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Sales)  #### 17.33
# Using Sarimax from statsmodels 
# As we do not have automatic function in indetifying the 
# best p,d,q combination 
# iterate over multiple combinations and return the best the combination
# For sarimax we require p,d,q and P,D,Q 
import statsmodels.api as sm
from itertools import combinations
def combinations_l(arr,r):
    return list(combinations(arr,r))
if __name__ == '__main__':
    arr = [1,2,3,4,5,6,7,8,9,10]
    r = 3
print(combinations_l(arr,r))
def combinations_u(arr1,r1):
    return list(combinations(arr1,r1))
if __name__ == '__main__':
    arr1 = [1,2,3,4,5,6,7,8,9,10,11]
    r1 = 3
print(combinations_u(arr1,r1))
def combinations_z(arr2,r2):
    return list(combinations(arr2,r2))
if __name__ == '__main__':
    arr2 = [1,2,3,4,5,6,7,8,9,10,11,12]
    r2 = 4
print(combinations_z(arr2,r2))
x=combinations_z(arr2, r2)
new_combination = [];
for i in x:
    if i[-1] == 12:
        new_combination.append(i)
print(new_combination)
results_sarima = []
best_aic = float("inf")
for i in combinations_l(arr,r):
    print(i)
    for j in combinations_z(arr2,r2):
        print(j)
        try:
            model_sarima = sm.tsa.statespace.SARIMAX(Train["Sales"],order = i,seasonal_order=j).fit(disp=-1)
        except:
            continue
        aic = model_sarima.aic
        if aic < best_aic:
            best_model = model_sarima
            best_aic = aic
            best_l = i
            best_u = j
        results_sarima.append([i,j,model_sarima.aic])
result_sarima_table = pd.DataFrame(results_sarima)
result_sarima_table.columns = ["paramaters_l","parameters_j","aic"]
result_sarima_table = result_sarima_table.sort_values(by="aic",ascending=True).reset_index(drop=True)
best_fit_model = sm.tsa.statespace.SARIMAX(Train["Sales"],order = (8,9,10),seasonal_order=(9,10,11,12)).fit(disp=-1)
best_fit_model.summary()
best_fit_model.aic 
srma_pred = best_fit_model.predict(start = Test.index[0],end = Test.index[-1])
plastic1["srma_pred"] = srma_pred
# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Sales"], label='Train',color="black")
plt.plot(Test.index, Test["Sales"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")
plt.plot(pred_hwe_mul_add.index,srma_pred,label="Auto_Sarima",color="purple")
plt.legend(loc='best')
# Models and their MAPE values
model_mapes = pd.DataFrame(columns=["model_name","mape"])
model_mapes["model_name"] = [""]
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))
Train1= plastic1.head(48)
Test = plastic1.tail(12)
###################### DATA DRIVEN APPROACHES #########################################
#################### Linear ################################
import statsmodels.formula.api as smf 
linear_model = smf.ols('Train1["Sales"]~Train1["t"]',data=Train1).fit()
pred_linear =  linear_model.predict(Train1)
rmse_linear = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_linear))**2))
rmse_linear ### 208.51
linear_model_test = smf.ols('Test["Sales"]~Test["t"]',data=Test).fit()
pred_linear_test = linear_model_test.predict(Test)
rmse_linear_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear_test))**2))
rmse_linear_test ### 233.09
##################### Exponential ##############################
Exp = smf.ols('Train1["log_Sales"]~Train1["t"]',data=Train1).fit()
pred_Exp = Exp.predict(Train1)
rmse_Exp = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp ### 208.95
Exp_test = smf.ols('Test["log_Sales"]~Test["t"]',data=Test).fit()
pred_Exp_test = Exp_test.predict(Test)
rmse_Exp_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp_test)))**2))
rmse_Exp_test  ## 234.86
#################### Quadratic ###############################
Quad = smf.ols('Train1["Sales"]~Train1["t"]+Train1["t_square"]',data=Train1).fit()
pred_Quad = Quad.predict(Train1)
rmse_Quad = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_Quad))**2))
rmse_Quad  ### 207.89
Quad_test = smf.ols('Test["Sales"]~(Test["t"]+Test["t_square"])',data=Test).fit()
pred_Quad_test = Quad_test.predict(Test)
rmse_Quad_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad_test))**2))
rmse_Quad_test  ### 87.52
################### Additive seasonality ########################
add_sea = smf.ols('Train1["Sales"]~Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1.iloc[:,7]+Train1["June"]+Train1["July"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data=Train1).fit()
pred_add_sea = add_sea.predict(Train1)
rmse_add_sea = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea  ### 122.46
add_sea_test = smf.ols('Test["Sales"]~Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,7]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data=Test).fit()
pred_add_sea_test = add_sea_test.predict(Test)
rmse_add_sea_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_test))**2))
rmse_add_sea_test  ### 4.59
################### Additive seasonality Linear ########################
add_sea_li = smf.ols('Train1["Sales"]~Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1.iloc[:,7]+Train1["June"]+Train1["July"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data=Train1).fit()
pred_add_sea_li = add_sea_li.predict(Train1)
rmse_add_sea_li = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_add_sea_li))**2))
rmse_add_sea_li 
add_sea_test_li = smf.ols('Test["Sales"]~Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,7]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data=Test).fit()
pred_add_sea_test_li = add_sea_test_li.predict(Test)
rmse_add_sea_test_li = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_test_li))**2))
rmse_add_sea_test_li 
################## Additive Seasonality Quadratic ############################
add_sea_Quad = smf.ols('Train1["Sales"]~Train1["t"]+Train1["t_square"]+Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1.iloc[:,7]+Train1["June"]+Train1["July"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data=Train).fit()
pred_add_sea_quad = add_sea_Quad.predict(Train1)
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #### 33.03
add_sea_Quad_test = smf.ols('Test["Sales"]~Test["t"]+Test["t_square"]+Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data=Test).fit()
pred_add_sea_quad_test = add_sea_Quad_test.predict(Test)
rmse_add_sea_quad_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad_test))**2))
rmse_add_sea_quad_test #### 4.29
################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('Train1["log_Sales"]~Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1.iloc[:,7]+Train1["June"]+Train1["July"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data = Train1).fit()
pred_Mult_sea = Mul_sea.predict(Train1)
rmse_Mult_sea = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea  #### 122.67
Mul_sea_test = smf.ols('Test["log_Sales"]~Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data = Test).fit()
pred_Mult_sea_test = Mul_sea_test.predict(Test)
rmse_Mult_sea_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea_test)))**2))
rmse_Mult_sea_test  #### 93.27
##################Multiplicative Additive Seasonality ###########
Mul_Add_sea = smf.ols('Train1["log_Sales"]~Train1["t"]+Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1.iloc[:,7]+Train1["June"]+Train1["July"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data = Train1).fit()
pred_Mult_add_sea = Mul_Add_sea.predict(Train1)
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Train1['log_Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea   ##### 1142.74
Mul_Add_sea_test = smf.ols('Test["log_Sales"]~Test["t"]+Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data = Test).fit()
pred_Mult_add_sea_test = Mul_Add_sea_test.predict(Test)
rmse_Mult_add_sea_test = np.sqrt(np.mean((np.array(Test['log_Sales'])-np.array(np.exp(pred_Mult_add_sea_test)))**2))
rmse_Mult_add_sea_test   ##### 1337.35
################## Testing #######################################
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far 
# Predicting new values 
predict_data = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Predict_new_plastic.csv")
model_full = smf.ols('predict_data["Sales"]~predict_data["t"]+predict_data["t_square"]+predict_data["Jan"]+predict_data["Feb"]+predict_data["Mar"]+predict_data["Apr"]+predict_data["May"]+predict_data["June"]+predict_data["July"]+predict_data["Aug"]+predict_data["Sep"]+predict_data["Oct"]+predict_data["Nov"]',data=plastic1).fit()
pred_new  = model_full.predict(predict_data)
pred_new
predict_data["forecasted_Sales"] = pd.Series(pred_new)
