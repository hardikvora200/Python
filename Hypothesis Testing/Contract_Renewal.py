import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
##install plotly package 
##import plotly.plotly as py
##import plotly.graph_objs as go
####from plotly.tools import FigureFactory as FF
data_supp=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\ContractRenewal_Data(unstacked).csv")
data_supp.columns="SupplierA","SupplierB","SupplierC"
##First check for Normality of all 3 suppliers:-
##H0:- Data are Normal , Ha: Data are not Normal
#Normality test 
SupA=stats.shapiro(data_supp.SupplierA)   
SupA_pValue=SupA[1]
print("p-value is: "+str(SupA_pValue))
SupB=stats.shapiro(data_supp.SupplierB)
SupB_pValue=SupB[1]
print("p-value is: "+str(SupB_pValue))
SupC=stats.shapiro(data_supp.SupplierC)
SupC_pValue=SupC[1]
print("p-value is: "+str(SupC_pValue))
###P-value in above 3 cases > 0.05 i.e. P High H0 Fly which implies Data are Normal
##Now we need to check for variance test,
##H0:- Variance of all 3 suppliers are same
##Ha:- Variance of at least 1 supplier is different
#Varience Test 
scipy.stats.levene(data_supp.SupplierA, data_supp.SupplierB)
scipy.stats.levene(data_supp.SupplierB, data_supp.SupplierC)
scipy.stats.levene(data_supp.SupplierC, data_supp.SupplierA)
##P-value  > 0.05 i.e. Variance of TAT of all 3 suppliers are same
#One-Way Anova
mod=smf.ols('SupplierA~SupplierB+SupplierC',data=data_supp).fit()
aov_table=sm.stats.anova_lm(mod,type=2)
print(aov_table)
mod1=smf.ols('SupplierB~SupplierA+SupplierC',data=data_supp).fit()
aov_table1=sm.stats.anova_lm(mod1,type=2)
print(aov_table1)
mod2=smf.ols('SupplierC~SupplierA+SupplierB',data=data_supp).fit()
aov_table2=sm.stats.anova_lm(mod2,type=2)
print(aov_table2)
## Ho- Average transaction time across 3 suppliers are same
##Ha- Average transaction time of atleast 1 supplier is different
### But as we have seen P>0.05 which means P High Ho fly
#### Accept Ho
### CMO needs to renew contracts of all 3 suppliers