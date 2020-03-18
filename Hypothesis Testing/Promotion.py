import pandas as pd
import scipy 
from scipy import stats
pr=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\Promotion.csv")
#Ho: Avg of purchases made by FIW < = Avg purchases made by SC =>default/ current/ no action
#Ha: Avg of purchases made by FIW > Avg purchases made by SC =>take action 
#Doing Normality test 
#We consider Ho: Data are normal
#We consider Ha: Data are not normal
Promotion=stats.shapiro(pr.InterestRateWaiver)
Promotion_pValue=Promotion[1]
print("p-value is: "+str(Promotion_pValue))    ###0.224
SDPromotion=stats.shapiro(pr.StandardPromotion)
SDPromotion_pValue=Promotion[1]
print("p-value is: "+str(SDPromotion_pValue))  ###0.224
#### P High Ho flywhich implies data are normal
#we can proceed with the model 
#Variance test 
### Ho = sigma2(Interest Rate Waiver)=sigma2(StandardPromotion)
### Ha = sigma2(Interest Rate Waiver)<>sigma2(StandardPromotion)
scipy.stats.levene(pr.InterestRateWaiver, pr.StandardPromotion)
### statistic=1.1334674473666406, pvalue=0.2875528565130808 
### P High Ho fly which implies variance is same for both
#2 Sample T test 
#### Ho = Mean(InterestRateWaiver)=Mean(StandardPromotion)
#### Ha = Mean(InterestRateWaiver)<>Mean(StandardPromotion)
scipy.stats.ttest_ind(pr.InterestRateWaiver,pr.StandardPromotion)
### statistic=2.260425163136941, pvalue=0.02422584468584312
scipy.stats.ttest_ind(pr.InterestRateWaiver,pr.StandardPromotion,equal_var = True)
### statistic=2.260425163136941, pvalue=0.02422584468584312
### As p-value = 0.024 < 0.05 which implies P low Ho go that means accept Ha
### Now which one is better
#### Ho = Mean(InterestRateWaiver)<=Mean(StandardPromotion)
#### Ha = Mean(InterestRateWaiver)>=Mean(StandardPromotion)
scipy.stats.ttest_ind(pr.InterestRateWaiver,pr.StandardPromotion)
### p-value = 0.012 which implies go with Interest Rate Waiver Schme