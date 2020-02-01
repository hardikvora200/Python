import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
LabTAT=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\Assignments\\LabTAT.csv")
## Doing Normality test 
### Ho: Data are normal
##### We consider Ha: Data are not normal
Laboratory_1=stats.shapiro(LabTAT.Laboratory_1)
Laboratory_1_pValue=Laboratory_1[1]
print("p-value is: "+str(Laboratory_1_pValue))
##  p-value is: 0.5506953597068787
Laboratory_2=stats.shapiro(LabTAT.Laboratory_2)
Laboratory_2_pValue=Laboratory_2[1]
print("p-value is: "+str(Laboratory_2_pValue))
###  p-value is: 0.8637524843215942
Laboratory_3=stats.shapiro(LabTAT.Laboratory_3)
Laboratory_3_pValue=Laboratory_3[1]
print("p-value is: "+str(Laboratory_3_pValue))
### p-value is: 0.4205053448677063
Laboratory_4=stats.shapiro(LabTAT.Laboratory_4)
Laboratory_4_pValue=Laboratory_4[1]
print("p-value is: "+str(Laboratory_4_pValue))
### p-value is: 0.6618951559066772
#### As p-value >0.05 for all 4 laboratories i.e. P High H0 Fly which implies Data are Normal 
##  we can proceed with the model 
#Variance test 
####### H0:- Variance of TAT of all 4 laboratories are same
## Ha:- Variance of TAT of at least 1 laboratory is different
scipy.stats.levene(LabTAT.Laboratory_1, LabTAT.Laboratory_2)
###LeveneResult(statistic=3.5495027780905763, pvalue=0.06078228171776711)
scipy.stats.levene(LabTAT.Laboratory_2, LabTAT.Laboratory_3)
### LeveneResult(statistic=0.9441465124387124, pvalue=0.33220021420602397)
scipy.stats.levene(LabTAT.Laboratory_3, LabTAT.Laboratory_4)
##### LeveneResult(statistic=2.037958464521512, pvalue=0.15472618294425391)
scipy.stats.levene(LabTAT.Laboratory_4, LabTAT.Laboratory_1)
### LeveneResult(statistic=1.5000140718506723, pvalue=0.22188001348277267)
## Now, we need to check for ANOVA- One way
#### H0:- Average TAT of all 4 laboratory is same , 
#####  Ha:- Average TAT of at least 1 laboratory is different
mod=smf.ols('Laboratory_4~Laboratory_1+Laboratory_2+Laboratory_3',data=LabTAT).fit()
aov_table=sm.stats.anova_lm(mod,type=3)
print(aov_table)
### P High Ho fly ==>Accept Ho