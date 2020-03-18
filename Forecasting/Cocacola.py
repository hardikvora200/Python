import pandas as pd
import numpy as np
import seaborn as sns
##import scipy.stats as st
import matplotlib.pyplot as plt# reading a csv file using pandas library
cocacola = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Forecasting\\Assignments\\CocaCola_Sales_Rawdata.xlsx")
##cocacola.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
cocacola.columns 
##cocacola['Quarter'] = pd.(cocacola['Quarter']).quarter
cocacola.head()
# new data frame with split value columns 
new = cocacola["Quarter"].str.split("_", n = 1, expand = True) 
# making separate first name column from new data frame 
cocacola["quarter"]= new[0]  
# making separate last name column from new data frame 
cocacola["Year"]= new[1] 
# Dropping old Name columns 
##cocacola.drop(columns =["Q4"], inplace = True) 
cocacola['Quarter'] = cocacola['quarter'].str.cat(cocacola['Year'], sep ="_") 
cocacola['Q1'] = [1 if x =='Q1' else 0 for x in cocacola['quarter']] 
cocacola["Q2"] = [1 if x =='Q2' else 0 for x in cocacola['quarter']]
cocacola["Q3"] = [1 if x =='Q3' else 0 for x in cocacola['quarter']]
cocacola["Q4"] = [1 if x =='Q4' else 0 for x in cocacola['quarter']]
##cocacola = pd.concat([cocacola,quarter],axis=1)
cocacola["t"] = np.arange(1,43)
cocacola["t_square"] = cocacola["t"]*cocacola["t"]
cocacola["log_Sales"] = np.log(cocacola["Sales"])
cocacola.Sales.plot()
##cocacola.drop(["a"],axis=1,inplace=True)
##cocacola.columns
# To get the count of null values in the data 
cocacola.isnull().sum()
cocacola.isna()
cocacola.shape # 60 18 => Before dropping null values
# To drop null values ( dropping rows)
cocacola.dropna().shape # 96 17 => After dropping null values
#####Exploratory Data Analysis#########################################################
cocacola.mean() #### Sales - 2994.353308,Q1-0.2619,Q2-0.2619,Q3-0.238,Q4-0.238,t- 21.5, t_square -609.166667, log_Sales - 7.954004
cocacola.median() 
cocacola.mode() 
####Measures of Dispersion
cocacola.var() 
cocacola.std() #### Sales - 977.930896,Q1-0.445001,Q2-0.445001,Q3-0.431081,Q4-0.431081,t-12.267844,t_square - 543.997396, log_Sales - 0.320225  
#### Calculate the range value
range1 = max(cocacola['Sales'])-min(cocacola['Sales'])  ### 3705.18
range2 = max(cocacola['log_Sales'])-min(cocacola['log_Sales']) ### 1.22
range3 = max(cocacola['t'])-min(cocacola['t']) ### 41
range4 = max(cocacola['t_square'])-min(cocacola['t_square'])  ##  1763
### Calculate skewness and Kurtosis
cocacola.skew()
cocacola.kurt() 
####Graphidelivery_time Representation 
plt.hist(cocacola["Sales"])
plt.hist(cocacola["log_Sales"])
plt.hist(cocacola["t"])
plt.hist(cocacola["t_square"])
plt.boxplot(cocacola["Sales"],0,"rs",0)
plt.boxplot(cocacola["log_Sales"],0,"rs",0)
plt.boxplot(cocacola["t"],0,"rs",0)
plt.boxplot(cocacola["t_square"],0,"rs",0)
plt.plot(cocacola["log_Sales"],cocacola["Sales"],"bo");plt.xlabel("log_Sales");plt.ylabel("Sales")
plt.plot(cocacola["t"],cocacola["Sales"],"bo");plt.xlabel("t");plt.ylabel("Sales")
plt.plot(cocacola["t_square"],cocacola["Sales"],"bo");plt.xlabel("t_square");plt.ylabel("Sales")
# table 
pd.crosstab(cocacola["log_Sales"],cocacola["Sales"])
pd.crosstab(cocacola["t"],cocacola["Sales"])
pd.crosstab(cocacola["t_square"],cocacola["Sales"])
## Barplot
pd.crosstab(cocacola["log_Sales"],cocacola["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(cocacola["t"],cocacola["Sales"]).plot(kind = "bar", width = 1.85)
pd.crosstab(cocacola["t_square"],cocacola["Sales"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Sales",data=cocacola,palette="hls")
sns.countplot(x="log_Sales",data=cocacola,palette="hls")
sns.countplot(x="t",data=cocacola,palette="hls")
sns.countplot(x="t_square",data=cocacola,palette="hls")
sns.boxplot(x="log_Sales",y="Sales",data=cocacola,palette="hls")
sns.boxplot(x="t",y="Sales",data=cocacola,palette="hls")
sns.boxplot(x="t_square",y="Sales",data=cocacola,palette="hls")
sns.pairplot(cocacola.iloc[:,0:17]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(cocacola,hue="Sales",size=2)
cocacola["Sales"].value_counts()
cocacola["log_Sales"].value_counts()
cocacola["t"].value_counts()
cocacola["t_square"].value_counts()
cocacola["Sales"].value_counts().plot(kind = "pie")
cocacola["log_Sales"].value_counts().plot(kind = "pie")
cocacola["t"].value_counts().plot(kind = "pie")
cocacola["t_square"].value_counts().plot(kind = "pie")
sns.pairplot(cocacola,hue="Sales",size=4,diag_kind = "kde")
sns.pairplot(cocacola,hue="log_Sales",size=4,diag_kind = "kde")
sns.pairplot(cocacola,hue="t",size=4,diag_kind = "kde")
sns.pairplot(cocacola,hue="t_square",size=4,diag_kind = "kde")
sns.FacetGrid(cocacola,hue="Sales").map(plt.scatter,"log_Sales","Sales").add_legend()
sns.FacetGrid(cocacola,hue="Sales").map(plt.scatter,"t","Sales").add_legend()
sns.FacetGrid(cocacola,hue="Sales").map(plt.scatter,"t_square","Sales").add_legend()
sns.FacetGrid(cocacola,hue="log_Sales").map(plt.scatter,"Sales","log_Sales").add_legend()
sns.FacetGrid(cocacola,hue="log_Sales").map(plt.scatter,"t","log_Sales").add_legend()
sns.FacetGrid(cocacola,hue="log_Sales").map(plt.scatter,"t_square","log_Sales").add_legend()
sns.FacetGrid(cocacola,hue="t").map(plt.scatter,"Sales","t").add_legend()
sns.FacetGrid(cocacola,hue="t").map(plt.scatter,"log_Sales","t").add_legend()
sns.FacetGrid(cocacola,hue="t").map(plt.scatter,"t_square","t").add_legend()
sns.FacetGrid(cocacola,hue="t_square").map(plt.scatter,"Sales","t_square").add_legend()
sns.FacetGrid(cocacola,hue="t_square").map(plt.scatter,"log_Sales","t_square").add_legend()
sns.FacetGrid(cocacola,hue="t_square").map(plt.scatter,"t","t_square").add_legend()
sns.FacetGrid(cocacola,hue="Sales").map(sns.kdeplot,"log_Sales").add_legend()
sns.FacetGrid(cocacola,hue="Sales").map(sns.kdeplot,"t").add_legend()
sns.FacetGrid(cocacola,hue="Sales").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(cocacola,hue="log_Sales").map(sns.kdeplot,"Sales").add_legend()
sns.FacetGrid(cocacola,hue="log_Sales").map(sns.kdeplot,"t").add_legend()
sns.FacetGrid(cocacola,hue="log_Sales").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(cocacola,hue="t").map(sns.kdeplot,"Sales").add_legend()
sns.FacetGrid(cocacola,hue="t").map(sns.kdeplot,"log_Sales").add_legend()
sns.FacetGrid(cocacola,hue="t").map(sns.kdeplot,"t_square").add_legend()
sns.FacetGrid(cocacola,hue="t_square").map(sns.kdeplot,"Sales").add_legend()
sns.FacetGrid(cocacola,hue="t_square").map(sns.kdeplot,"log_Sales").add_legend()
sns.FacetGrid(cocacola,hue="t_square").map(sns.kdeplot,"t").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab
# Checking Whether data is normally distributed
stats.probplot(cocacola['Sales'],dist="norm",plot=pylab)
stats.probplot(np.log(cocacola['Sales']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(cocacola['Sales']),dist="norm",plot=pylab)
stats.probplot((cocacola['Sales'] * cocacola['Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['Sales'])*np.exp(cocacola['Sales']),dist="norm",plot=pylab)
reci_1=1/cocacola['Sales']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((cocacola['Sales'] * cocacola['Sales'])+cocacola['Sales']),dist="norm",plot=pylab)
stats.probplot(cocacola['log_Sales'],dist="norm",plot=pylab)
stats.probplot(np.log(cocacola['log_Sales']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(cocacola['log_Sales']),dist="norm",plot=pylab)
stats.probplot((cocacola['log_Sales'] * cocacola['log_Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['log_Sales']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['log_Sales'])*np.exp(cocacola['log_Sales']),dist="norm",plot=pylab)
reci_2=1/cocacola['log_Sales']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((cocacola['log_Sales'] * cocacola['log_Sales'])+cocacola['log_Sales']),dist="norm",plot=pylab)
stats.probplot(cocacola['t'],dist="norm",plot=pylab)
stats.probplot(np.log(cocacola['t']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(cocacola['t']),dist="norm",plot=pylab)
stats.probplot((cocacola['t'] * cocacola['t']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['t']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['t'])*np.exp(cocacola['t']),dist="norm",plot=pylab)
reci_3=1/cocacola['t']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((cocacola['t'] * cocacola['t'])+cocacola['t']),dist="norm",plot=pylab)
stats.probplot(cocacola['t_square'],dist="norm",plot=pylab)
stats.probplot(np.log(cocacola['t_square']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(cocacola['t_square']),dist="norm",plot=pylab)
stats.probplot((cocacola['t_square'] * cocacola['t_square']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['t_square']),dist="norm",plot=pylab)
stats.probplot(np.exp(cocacola['t_square'])*np.exp(cocacola['t_square']),dist="norm",plot=pylab)
reci_4=1/cocacola['t_square']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((cocacola['t_square'] * cocacola['t_square'])+cocacola['t_square']),dist="norm",plot=pylab)
# ppf => Percent point function 
##cocacola.mean() #### Sales - 2994.353308, Q1_86-Q4_95 - 0.023810,t- 21.5, t_square -609.166667, log_Sales - 7.954004
##cocacola.std() #### Sales - 977.930896,Q1_86-Q4_95-0.154303,t-12.267844,t_square - 543.997396, log_Sales - 0.320225  
#### Sales
stats.norm.ppf(0.975,2994.353308,977.930896)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(cocacola["Sales"],2994.353308,977.930896) # similar to pnorm in R 
#### log_Sales
stats.norm.ppf(0.975,7.954004,0.320225)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(cocacola["log_Sales"],7.954004,0.320225) # similar to pnorm in R 
#### t
stats.norm.ppf(0.975,21.5,12.267844)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(cocacola["t"],21.5,12.267844) # similar to pnorm in R 
#### t_square
stats.norm.ppf(0.975, 609.166667,543.997396)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(cocacola["t_square"],609.166667,543.997396) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
cocacola.corr(method = "pearson")
cocacola.corr(method = "kendall")
# to get top 6 rows
cocacola.head(40) # to get top n rows use cars.head(10)
cocacola.tail(10)
### Normalization
def norm_func(i):
      x = (i-i.mean())/(i.std())
      return (x)
##Normalized data frame (considering the numerical part of data)
df_norm = norm_func(cocacola.iloc[:,0:])
# Scatter plot between the variables along with histograms
##cocacola.drop(['Month'],axis=1,inplace=True)
sns.pairplot(cocacola)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
# Converting the normal index of cocacola to time stamp 
##cocacola.index = pd.to_datetime(cocacola.Quarter,format="%")
cocacola.Sales.plot() # time series plot 
# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_quarter_Q1 = pd.pivot_table(data=cocacola,values="Sales",index="Q1",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_quarter_Q1,annot=True,fmt="g")
heatmap_y_quarter_Q2 = pd.pivot_table(data=cocacola,values="Sales",index="Q2",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_quarter_Q2,annot=True,fmt="g")
heatmap_y_quarter_Q3 = pd.pivot_table(data=cocacola,values="Sales",index="Q3",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_quarter_Q3,annot=True,fmt="g")
heatmap_y_quarter_Q4 = pd.pivot_table(data=cocacola,values="Sales",index="Q4",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_quarter_Q4,annot=True,fmt="g")
# Boxplot for ever
sns.boxplot(x="Q1",y="Sales",data=cocacola)
sns.boxplot(x="Q2",y="Sales",data=cocacola)
sns.boxplot(x="Q3",y="Sales",data=cocacola)
sns.boxplot(x="Q4",y="Sales",data=cocacola)
sns.boxplot(x="Q1",y="log_Sales",data=cocacola)
sns.boxplot(x="Q2",y="log_Sales",data=cocacola)
sns.boxplot(x="Q3",y="log_Sales",data=cocacola)
sns.boxplot(x="Q4",y="log_Sales",data=cocacola)
# moving average for the time series to understand better about the trend character in Amtrak
cocacola.Sales.plot(label="org")
for i in range(2,27,6):
    cocacola["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cocacola.Sales,model="additive",period=3)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cocacola.Sales,model="multiplicative",period=4)
decompose_ts_mul.plot()
# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(cocacola.Sales,lags=10)
tsa_plots.plot_pacf(cocacola.Sales,lags=10)
# lets verify on Train and Test data 
Train = cocacola.head(36)
Test = cocacola.tail(6)# to change the index value in pandas data frame 
# Creating a function to calculate the MAPE value for test data 
##cocacola.drop(['Month'],axis=1,inplace=True)
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)
############# SMOOTHING TECHNIQUES #######################################################
################################### Simple Exponential Method#######################################################
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
pred_ses_int=pd.Series(pred_ses).astype(int)
MAPE(pred_ses_int,Test.Sales)  ### 11.63
#################################### Holt method############################################ 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
pred_hw_int=pd.Series(pred_hw).astype(int)
MAPE(pred_hw_int,Test.Sales)   ###### 9.89
#################################### Holts winter exponential smoothing ##########################
hwe_model = ExponentialSmoothing(Train["Sales"],seasonal_periods=4).fit()
pred_hwe = hwe_model.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_int=pd.Series(pred_hwe).astype(int)
MAPE(pred_hwe_int,Test.Sales)  #### 11.62 
#################################### Holts winter exponential smoothing with additive seasonality and additive trend##########################
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_add_add_int=pd.Series(pred_hwe_add_add).astype(int)
MAPE(pred_hwe_add_add_int,Test.Sales)   ###### 3.59
################################## Holts winter exponential smoothing with multiplicative seasonality and additive trend####################
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
pred_hwe_mul_add_int=pd.Series(pred_hwe_mul_add).astype(int)
MAPE(pred_hwe_mul_add_int,Test.Sales)   #### 1.83
# Lets us use auto_arima from p
from pmdarima.arima import auto_arima
auto_arima_model = auto_arima(Train["Sales"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=4,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)    #### Time - 0.281 seconds, Total fit time: 109.464 seconds
auto_arima_model.summary() # SARIMAX(3, 1, 0)x(0, 1, [1], 4)
# AIC ==> 411.763, BIC ==> 420.367, HQIC ==> 414.568 
# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )
# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=6))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Sales)  #### 3.97
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
    arr = [1,2]
    r = 2
print(combinations_l(arr,r)) ######[(1,2)]
def combinations_u(arr1,r1):
    return list(combinations(arr1,r1))
if __name__ == '__main__':
    arr1 = [1,2,3]
    r1 = 3
print(combinations_u(arr1,r1)) ### [(1,2,3)]
def combinations_z(arr2,r2):
    return list(combinations(arr2,r2))
if __name__ == '__main__':
    arr2 = [1,2,3,4]
    r2 = 4
print(combinations_z(arr2,r2))   ##### [(1,2,3,4)]
x=combinations_z(arr2, r2)
new_combination = [];
for i in x:
    if i[-1] == 4:
        new_combination.append(i)
print(new_combination)
results_sarima = []
best_aic = float("inf")
for i in combinations_l(arr1,r1):
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
result_sarima_table = pd.DataFrame(results_sarima)  ### i - (1,2,3),j-(1,2,3,4), model_sarima.aic - 369.755105
result_sarima_table.columns = ["paramaters_l","parameters_j","aic"]
result_sarima_table = result_sarima_table.sort_values(by="aic",ascending=True).reset_index(drop=True)
best_fit_model = sm.tsa.statespace.SARIMAX(Train["Sales"],order = (1,2,3),seasonal_order=(1,2,3,4)).fit(disp=-1)
best_fit_model.summary()
best_fit_model.aic 
srma_pred = best_fit_model.predict(start = Test.index[0],end = Test.index[-1])
cocacola["srma_pred"] = srma_pred
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
Train1= cocacola.head(36)
Test = cocacola.tail(6)
################################# DATA DRIVEN APPROACHES ###############################
#################### Linear ################################
import statsmodels.formula.api as smf 
linear_model = smf.ols('Train1["Sales"]~Train1["t"]',data=Train1).fit()
pred_linear =  linear_model.predict(Train1)
rmse_linear = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_linear))**2))
rmse_linear #### 336.951
linear_model_test = smf.ols('Test["Sales"]~Test["t"]',data=Test).fit()
pred_linear_test = linear_model_test.predict(Test)
rmse_linear_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear_test))**2))
rmse_linear_test    ##### 440.28
##################### Exponential ##############################
Exp = smf.ols('Train1["log_Sales"]~Train1["t"]',data=Train1).fit()
pred_Exp = Exp.predict(Train1)
rmse_Exp = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp   #### 303.99
Exp_test = smf.ols('Test["log_Sales"]~Test["t"]',data=Test).fit()
pred_Exp_test = Exp_test.predict(Test)
rmse_Exp_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp_test)))**2))
rmse_Exp_test  #### 440.99
#################### Quadratic ###############################
Quad = smf.ols('Train1["Sales"]~Train1["t"]+Train1["t_square"]',data=Train1).fit()
pred_Quad = Quad.predict(Train1)
rmse_Quad = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_Quad))**2))
rmse_Quad  ####### 272.88
Quad_test = smf.ols('Test["Sales"]~(Test["t"]+Test["t_square"])',data=Test).fit()
pred_Quad_test = Quad_test.predict(Test)
rmse_Quad_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad_test))**2))
rmse_Quad_test  ###### 439.70
################### Additive seasonality ########################
add_sea = smf.ols('Train1["Sales"]~Train1["Q1"]+Train1["Q2"]+Train1["Q3"]+Train1["Q4"]',data=Train1).fit()
pred_add_sea = add_sea.predict(Train1)
rmse_add_sea = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea ########## 705.03
add_sea_test = smf.ols('Test["Sales"]~Test["Q1"]+Test["Q2"]+Test["Q3"]+Test["Q4"]',data=Test).fit()
pred_add_sea_test = add_sea_test.predict(Test)
rmse_add_sea_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_test))**2))
rmse_add_sea_test  ######## 134.19
################### Additive seasonality Linear########################
add_sea_li = smf.ols('Train1["Sales"]~Train1["t"]+Train1["Q1"]+Train1["Q2"]+Train1["Q3"]+Train1["Q4"]',data=Train1).fit()
pred_add_sea_li = add_sea_li.predict(Train1)
rmse_add_sea_li = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_add_sea_li))**2))
rmse_add_sea_li ########## 248.42
add_sea_li_test = smf.ols('Test["Sales"]~Test["t"]+Test["Q1"]+Test["Q2"]+Test["Q3"]+Test["Q4"]',data=Test).fit()
pred_add_sea_li_test = add_sea_li_test.predict(Test)
rmse_add_sea_li_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_li_test))**2))
rmse_add_sea_li_test  ######## 4.69
################## Additive Seasonality Quadratic ############################
add_sea_Quad = smf.ols('Train1["Sales"]~Train1["t"]+Train1["t_square"]+Train1["Q1"]+Train1["Q2"]+Train1["Q3"]+Train1["Q4"]',data=Train).fit()
pred_add_sea_quad = add_sea_Quad.predict(Train1)
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  ######## 147.49
add_sea_Quad_test = smf.ols('Test["Sales"]~Test["t"]+Test["t_square"]+Test["Q1"]+Test["Q2"]+Test["Q3"]+Test["Q4"]',data=Test).fit()
pred_add_sea_quad_test = add_sea_Quad_test.predict(Test)
rmse_add_sea_quad_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad_test))**2))
rmse_add_sea_quad_test  ########## 7.83
################## Multiplicative Seasonality ##################
Mul_sea = smf.ols('Train1["log_Sales"]~Train1["Q1"]+Train1["Q2"]+Train1["Q3"]+Train1["Q4"]',data = Train1).fit()
pred_Mult_sea = Mul_sea.predict(Train1)
rmse_Mult_sea = np.sqrt(np.mean((np.array(Train1['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea  ##### 710.27
Mul_sea_test = smf.ols('Test["log_Sales"]~Test["Q1"]+Test["Q2"]+Test["Q3"]+Test["Q4"]',data = Test).fit()
pred_Mult_sea_test = Mul_sea_test.predict(Test)
rmse_Mult_sea_test = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea_test)))**2))
rmse_Mult_sea_test  ##### 134.21
##################Multiplicative Additive Seasonality ###########
Mul_Add_sea = smf.ols('Train1["log_Sales"]~Train1["t"]+Train1["Q1"]+Train1["Q2"]+Train1["Q3"]+Train1["Q4"]',data = Train1).fit()
pred_Mult_add_sea = Mul_Add_sea.predict(Train1)
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Train1['log_Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea   ##### 2798.63
Mul_Add_sea_test = smf.ols('Test["log_Sales"]~Test["t"]+Test["Q1"]+Test["Q2"]+Test["Q3"]+Test["Q4"]',data = Test).fit()
pred_Mult_add_sea_test = Mul_Add_sea_test.predict(Test)
rmse_Mult_add_sea_test = np.sqrt(np.mean((np.array(Test['log_Sales'])-np.array(np.exp(pred_Mult_add_sea_test)))**2))
rmse_Mult_add_sea_test  ###### 4595.012 
################## Testing #######################################
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_li","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_li,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far 
# Predicting new values 
predict_data = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr\\Forecasting\\Cocacola_new.xlsx")
model_full = smf.ols('predict_data["Sales"]~predict_data["t"]+predict_data["t_square"]+predict_data["Q1"]+predict_data["Q2"]+predict_data["Q3"]+predict_data["Q4"]',data=cocacola).fit()
pred_new  = model_full.predict(predict_data)
pred_new
predict_data["forecasted_Sales"] = pd.Series(pred_new)

        