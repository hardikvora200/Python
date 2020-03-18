import pandas as pd
import numpy as np
import seaborn as sns
##import scipy.stats as st
import matplotlib.pyplot as plt# reading a csv file using pandas library
airline=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Airlinedata.csv")
##airline.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
airline.columns 
##'Month', 'Passengers', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
##       'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'log_Passengers', 't', 't_square'
airline.Passengers.plot()
##airline.drop(["a"],axis=1,inplace=True)
##airline.columns
# To get the count of null values in the data 
airline.isnull().sum()
airline.isna()
airline.shape # 96 17 => Before dropping null values
# To drop null values ( dropping rows)
airline.dropna().shape # 96 17 => After dropping null values
#####Exploratory Data Analysis#########################################################
airline.mean() ## Passengers-213.708333,(Jan-Dec)-0.083333,log_Passengers-5.309322,t-48.5,t_square-3120.166667
airline.median() 
airline.mode() 
####Measures of Dispersion
airline.var() #### 5172.229825
airline.std() ## Passengers-71.918216,log_Passengers-0.335338,t-27.856777,t_square-2788.897895
#### Calculate the range value
range1 = max(airline['Passengers'])-min(airline['Passengers'])  ### 309
range2 = max(airline['log_Passengers'])-min(airline['log_Passengers']) ### 1.38
range3 = max(airline['t'])-min(airline['t']) ### 95
range4 = max(airline['t_square'])-min(airline['t_square'])  ##  9215
### Calculate skewness and Kurtosis
airline.skew()
airline.kurt() 
####Graphidelivery_time Representation 
plt.hist(airline["Passengers"])
plt.hist(airline["log_Passengers"])
plt.hist(airline["t"])
plt.hist(airline["t_square"])
plt.boxplot(airline["Passengers"],0,"rs",0)
plt.boxplot(airline["log_Passengers"],0,"rs",0)
plt.boxplot(airline["t"],0,"rs",0)
plt.boxplot(airline["t_square"],0,"rs",0)
plt.plot(airline["log_Passengers"],airline["Passengers"],"bo");plt.xlabel("log_Passengers");plt.ylabel("Passengers")
plt.plot(airline["t"],airline["Passengers"],"bo");plt.xlabel("t");plt.ylabel("Passengers")
plt.plot(airline["t_square"],airline["Passengers"],"bo");plt.xlabel("t_square");plt.ylabel("Passengers")
# table 
pd.crosstab(airline["log_Passengers"],airline["Passengers"])
pd.crosstab(airline["t"],airline["Passengers"])
pd.crosstab(airline["t_square"],airline["Passengers"])
## Barplot
pd.crosstab(airline["log_Passengers"],airline["Passengers"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["t"],airline["Passengers"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["t_square"],airline["Passengers"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Passengers",data=airline,palette="hls")
sns.countplot(x="log_Passengers",data=airline,palette="hls")
sns.countplot(x="t",data=airline,palette="hls")
sns.countplot(x="t_square",data=airline,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="log_Passengers",y="Passengers",data=airline,palette="hls")
sns.boxplot(x="t",y="Passengers",data=airline,palette="hls")
sns.boxplot(x="t_square",y="Passengers",data=airline,palette="hls")
sns.pairplot(airline.iloc[:,0:17]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(airline,hue="Passengers",size=2)
airline["Passengers"].value_counts()
airline["log_Passengers"].value_counts()
airline["t"].value_counts()
airline["t_square"].value_counts()
airline["Passengers"].value_counts().plot(kind = "pie")
airline["log_Passengers"].value_counts().plot(kind = "pie")
airline["t"].value_counts().plot(kind = "pie")
airline["t_square"].value_counts().plot(kind = "pie")
sns.pairplot(airline,hue="Passengers",size=4,diag_kind = "kde")
sns.pairplot(airline,hue="log_Passengers",size=4,diag_kind = "kde")
sns.pairplot(airline,hue="t",size=4,diag_kind = "kde")
sns.pairplot(airline,hue="t_square",size=4,diag_kind = "kde")
sns.FacetGrid(airline,hue="Passengers").map(plt.scatter,"log_Passengers","Passengers").add_legend()
sns.FacetGrid(airline,hue="Passengers").map(plt.scatter,"t","Passengers").add_legend()
sns.FacetGrid(airline,hue="Passengers").map(plt.scatter,"t_square","Passengers").add_legend()
sns.FacetGrid(airline,hue="log_Passengers").map(plt.scatter,"Passengers","log_Passengers").add_legend()
sns.FacetGrid(airline,hue="log_Passengers").map(plt.scatter,"t","log_Passengers").add_legend()
sns.FacetGrid(airline,hue="log_Passengers").map(plt.scatter,"t_square","log_Passengers").add_legend()
sns.FacetGrid(airline,hue="t").map(plt.scatter,"Passengers","t").add_legend()
sns.FacetGrid(airline,hue="t").map(plt.scatter,"log_Passengers","t").add_legend()
sns.FacetGrid(airline,hue="t").map(plt.scatter,"t_square","t").add_legend()
sns.FacetGrid(airline,hue="t_square").map(plt.scatter,"Passengers","t_square").add_legend()
sns.FacetGrid(airline,hue="t_square").map(plt.scatter,"log_Passengers","t_square").add_legend()
sns.FacetGrid(airline,hue="t_square").map(plt.scatter,"t","t_square").add_legend()
sns.FacetGrid(airline,hue="Passengers").map(sns.kdeplot,"log_Passengers").add_legend()
sns.FacetGrid(airline,hue="Passengers").map(sns.kdeplot,"t").add_legend()
sns.FacetGrid(airline,hue="Passengers").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(airline,hue="log_Passengers").map(sns.kdeplot,"Passengers").add_legend()
sns.FacetGrid(airline,hue="log_Passengers").map(sns.kdeplot,"t").add_legend()
sns.FacetGrid(airline,hue="log_Passengers").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(airline,hue="t").map(sns.kdeplot,"Passengers").add_legend()
sns.FacetGrid(airline,hue="t").map(sns.kdeplot,"log_Passengers").add_legend()
sns.FacetGrid(airline,hue="t").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(airline,hue="t_square").map(sns.kdeplot,"Passengers").add_legend()
sns.FacetGrid(airline,hue="t_square").map(sns.kdeplot,"log_Passengers").add_legend()
sns.FacetGrid(airline,hue="t_square").map(sns.kdeplot,"t").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab
# Checking Whether data is normally distributed
stats.probplot(airline['Passengers'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Passengers']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Passengers']),dist="norm",plot=pylab)
stats.probplot((airline['Passengers'] * airline['Passengers']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Passengers']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Passengers'])*np.exp(airline['Passengers']),dist="norm",plot=pylab)
reci_1=1/airline['Passengers']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((airline['Passengers'] * airline['Passengers'])+airline['Passengers']),dist="norm",plot=pylab)
stats.probplot(airline['log_Passengers'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['log_Passengers']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['log_Passengers']),dist="norm",plot=pylab)
stats.probplot((airline['log_Passengers'] * airline['log_Passengers']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['log_Passengers']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['log_Passengers'])*np.exp(airline['log_Passengers']),dist="norm",plot=pylab)
reci_2=1/airline['log_Passengers']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((airline['log_Passengers'] * airline['log_Passengers'])+airline['log_Passengers']),dist="norm",plot=pylab)
stats.probplot(airline['t'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['t']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['t']),dist="norm",plot=pylab)
stats.probplot((airline['t'] * airline['t']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['t']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['t'])*np.exp(airline['t']),dist="norm",plot=pylab)
reci_3=1/airline['t']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((airline['t'] * airline['t'])+airline['t']),dist="norm",plot=pylab)
stats.probplot(airline['t_square'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['t_square']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['t_square']),dist="norm",plot=pylab)
stats.probplot((airline['t_square'] * airline['t_square']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['t_square']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['t_square'])*np.exp(airline['t_square']),dist="norm",plot=pylab)
reci_4=1/airline['t_square']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((airline['t_square'] * airline['t_square'])+airline['t_square']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Passengers
stats.norm.ppf(0.975,213.708333, 71.918216)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Passengers"],213.708333,71.918216) # similar to pnorm in R 
#### log_Passengers
stats.norm.ppf(0.975,5.309322,0.335338)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(airline["log_Passengers"],5.309322,0.335338) # similar to pnorm in R 
#### t
stats.norm.ppf(0.975,48.5,27.856777)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(airline["t"],48.5,27.856777) # similar to pnorm in R 
#### t_square
stats.norm.ppf(0.975, 3120.166667,2788.897895)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(airline["t_square"], 3120.166667,2788.897895) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
airline.corr(method = "pearson")
airline.corr(method = "kendall")
# to get top 6 rows
airline.head(40) # to get top n rows use cars.head(10)
airline.tail(10)
### Normalization
##def norm_func(i):
  ##  x = (i-i.mean())/(i.std())
 ##   return (x)
### Normalized data frame (considering the numerical part of data)
##df_norm = norm_func(airline.iloc[:,0:])
# Scatter plot between the variables along with histograms
##airline.drop(['Month'],axis=1,inplace=True)
sns.pairplot(airline)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
##import statsmodels.tsa.statespace as tm_models
#from sm.tsa.statespace import sa
# Converting the normal index of airline to time stamp 
airline.index = pd.to_datetime(airline.Month,format="%")
airline.Passengers.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
###airline["Date"] = pd.to_datetime(airline.Month,format="%b-%y")
airline["Date"] = pd.DatetimeIndex(airline["Month"]).date ## date extraction
# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 
airline["month"] = pd.DatetimeIndex(airline["Month"]).month # month extraction
airline["year"] = pd.DatetimeIndex(airline["Month"]).year # year extraction
# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=airline,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")
# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=airline)
sns.boxplot(x="year",y="Passengers",data=airline)
# Line plot for Ridership based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=airline)
# moving average for the time series to understand better about the trend character in Amtrak
airline.Passengers.plot(label="org")
for i in range(2,27,6):
    airline["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(airline.Passengers,model="additive",period=3)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(airline.Passengers,model="multiplicative",period=4)
decompose_ts_mul.plot()
# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(airline.Passengers,lags=10)
tsa_plots.plot_pacf(airline.Passengers)
# lets verify on Train and Test data 
Train = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Airlines+Data.xlsx")
Test = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Airlines+Data_Test.xlsx")
# to change the index value in pandas data frame 
# Creating a function to calculate the MAPE value for test data 
##airline.drop(['Month'],axis=1,inplace=True)
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)
############################## SMOOTHING TECHNIQUES ####################################
################################## Simple Exponential Method###########################
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
pred_ses_int=pd.Series(pred_ses).astype(int)
MAPE(pred_ses_int,Test.Passengers) # 59.13
########################################### Holt method #########################################
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
pred_hw_int=pd.Series(pred_hw).astype(int)
MAPE(pred_hw_int,Test.Passengers) # 58.53
#################################### Holts winter exponential smoothing ##########################
hwe_model = ExponentialSmoothing(Train["Passengers"],seasonal_periods=12).fit()
pred_hwe = hwe_model.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_int=pd.Series(pred_hwe).astype(int)
MAPE(pred_hwe_int,Test.Sales) 
################### Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_add_add_int=pd.Series(pred_hwe_add_add).astype(int)
MAPE(pred_hwe_add_add_int,Test.Passengers) # 58.17
#################### Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_mul_add_int=pd.Series(pred_hwe_mul_add).astype(int)
MAPE(pred_hwe_mul_add_int,Test.Passengers) # 58.02
# Lets us use auto_arima from p
from pmdarima.arima import auto_arima
auto_arima_model = auto_arima(Train["Passengers"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
auto_arima_model.summary() # SARIMAX(0, 1, 3)x(1, 1, [], 12)
# AIC ==> 616.109,# BIC ==>  630.622, HQIC ==> 621.939 
# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )
# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=16))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Passengers)  #### 26.26
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
            model_sarima = sm.tsa.statespace.SARIMAX(Train["Passengers"],order = i,seasonal_order=j).fit(disp=-1)
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
airline["srma_pred"] = srma_pred
# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
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
Train1=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Airlinedata.csv")
Test = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Airlines+Data_Test.xlsx")
#################### Linear ################################
import statsmodels.formula.api as smf 
linear_model = smf.ols('Train1["Passengers"]~Train1["t"]',data=Train1).fit()
pred_linear =  linear_model.predict(Train1)
rmse_linear = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(pred_linear))**2))
rmse_linear ### 30.54
linear_model_test = smf.ols('Test["Passengers"]~Test["t1"]',data=Test).fit()
pred_linear_test = linear_model_test.predict(Test)
rmse_linear_test = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear_test))**2))
rmse_linear_test ### 43.09
##################### Exponential ##############################
Exp = smf.ols('Train1["log_Passengers"]~Train1["t"]',data=Train1).fit()
pred_Exp = Exp.predict(Train1)
rmse_Exp = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp ### 29.64
Exp_test = smf.ols('Test["log(Passengers)"]~Test["t1"]',data=Test).fit()
pred_Exp_test = Exp_test.predict(Test)
rmse_Exp_test = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp_test)))**2))
rmse_Exp_test  ## 306.61
#################### Quadratic ###############################
Quad = smf.ols('Train1["Passengers"]~Train1["t"]+Train1["t_square"]',data=Train1).fit()
pred_Quad = Quad.predict(Train1)
rmse_Quad = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad  ### 29.59
Quad_test = smf.ols('Test["Passengers"]~(Test["t1"]+Test["t1^2"])',data=Test).fit()
pred_Quad_test = Quad_test.predict(Test)
rmse_Quad_test = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad_test))**2))
rmse_Quad_test  ### 38.91
################### Additive seasonality ########################
add_sea = smf.ols('Train1["Passengers"]~Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1["May"]+Train1["Jun"]+Train1["Jul"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data=Train1).fit()
pred_add_sea = add_sea.predict(Train1)
rmse_add_sea = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea  ### 66.42
add_sea_test = smf.ols('Test["Passengers"]~Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data=Test).fit()
pred_add_sea_test = add_sea_test.predict(Test)
rmse_add_sea_test = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_test))**2))
rmse_add_sea_test  ### 12.26
################### Additive seasonality Linear ########################
add_sea_li = smf.ols('Train1["Passengers"]~Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1["May"]+Train1["Jun"]+Train1["Jul"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data=Train1).fit()
pred_add_sea_li = add_sea_li.predict(Train1)
rmse_add_sea_li = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(pred_add_sea_li))**2))
rmse_add_sea_li  
add_sea_test_li = smf.ols('Test["Passengers"]~Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data=Test).fit()
pred_add_sea_test_li = add_sea_test_li.predict(Test)
rmse_add_sea_test_li = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_test_li))**2))
rmse_add_sea_test_li  
################## Additive Seasonality Quadratic ############################
add_sea_Quad = smf.ols('Train1["Passengers"]~Train1["t"]+Train1["t_square"]+Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1["May"]+Train1["Jun"]+Train1["Jul"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data=Train).fit()
pred_add_sea_quad = add_sea_Quad.predict(Train1)
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad #### 13.48
add_sea_Quad_test = smf.ols('Test["Passengers"]~Test["t1"]+Test["t1^2"]+Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data=Test).fit()
pred_add_sea_quad_test = add_sea_Quad_test.predict(Test)
rmse_add_sea_quad_test = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad_test))**2))
rmse_add_sea_quad_test #### 0.94
################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('Train1["log_Passengers"]~Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1["May"]+Train1["Jun"]+Train1["Jul"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data = Train1).fit()
pred_Mult_sea = Mul_sea.predict(Train1)
rmse_Mult_sea = np.sqrt(np.mean((np.array(Train1['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea  #### 67.21
Mul_sea_test = smf.ols('Test["log(Passengers)"]~Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data = Test).fit()
pred_Mult_sea_test = Mul_sea_test.predict(Test)
rmse_Mult_sea_test = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea_test)))**2))
rmse_Mult_sea_test  #### 306.49
##################Multiplicative Additive Seasonality ###########
Mul_Add_sea = smf.ols('Train1["log_Passengers"]~Train1["t"]+Train1["Jan"]+Train1["Feb"]+Train1["Mar"]+Train1["Apr"]+Train1["May"]+Train1["Jun"]+Train1["Jul"]+Train1["Aug"]+Train1["Sep"]+Train1["Oct"]+Train1["Nov"]',data = Train1).fit()
pred_Mult_add_sea = Mul_Add_sea.predict(Train1)
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Train1['log_Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea   ##### 219.72
Mul_Add_sea_test = smf.ols('Test["log(Passengers)"]~Test["t1"]+Test["Jan"]+Test["Feb"]+Test["Mar"]+Test["Apr"]+Test.iloc[:,6]+Test["June"]+Test["July"]+Test["Aug"]+Test["Sep"]+Test["Oct"]+Test["Nov"]',data = Test).fit()
pred_Mult_add_sea_test = Mul_Add_sea_test.predict(Test)
rmse_Mult_add_sea_test = np.sqrt(np.mean((np.array(Test['log(Passengers)'])-np.array(np.exp(pred_Mult_add_sea_test)))**2))
rmse_Mult_add_sea_test   ##### 9.66
################## Testing #######################################
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 
predict_data = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\Predict_new_airline.xlsx")
model_full = smf.ols('predict_data["Passengers"]~predict_data["t"]+predict_data["t_square"]+predict_data["Jan"]+predict_data["Feb"]+predict_data["Mar"]+predict_data["Apr"]+predict_data["May"]+predict_data["Jun"]+predict_data["Jul"]+predict_data["Aug"]+predict_data["Sep"]+predict_data["Oct"]+predict_data["Nov"]',data=predict_data).fit()
pred_new  = model_full.predict(predict_data)
pred_new
predict_data["forecasted_Passengers"] = pd.Series(pred_new)
