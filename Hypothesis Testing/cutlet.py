import pandas as pd
import scipy 
from scipy import stats
cutlet=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\Assignments\\Cutlets.csv")
## Doing Normality test 
### Ho: Data are normal
##### We consider Ha: Data are not normal
UnitA=stats.shapiro(cutlet.Unit_A)
UnitA_pValue=UnitA[1]
print("p-value is: "+str(UnitA_pValue))
UnitB=stats.shapiro(cutlet.Unit_B)
UnitB_pValue=UnitB[1]
print("p-value is: "+str(UnitB_pValue))
#we can proceed with the model 
#Variance test 
## H0: [sigma(UnitA)]^2 = [sigma(UnitB)]^2
## Ha: Variance of Unit A is not equal to variance of Unit B
scipy.stats.levene(cutlet.Unit_A, cutlet.Unit_B)
### LeveneResult(statistic=0.665089763863238, pvalue=0.4176162212502553)
## Now compare means using 2 Sample t test assuming equal variance
#### H0: Mean(Unit A) = Mean(Unit B) , Ha: Mean(Unit A) != Mean(Unit B)
#2 Sample T test 
scipy.stats.ttest_ind(cutlet.Unit_A, cutlet.Unit_B)
scipy.stats.ttest_ind(cutlet.Unit_A, cutlet.Unit_B,equal_var = True)
#####  Ttest_indResult(statistic=0.7228688704678061, pvalue=0.4722394724599501)
## Mean(Unit A) = Mean(Unit B)