import numpy as np
import pandas as pd
faltoon=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\Assignments\\Faltoons.csv")
from statsmodels.stats.proportion import proportions_ztest
#we do the cross table and see How many adults or children are purchasing
tab = faltoon.groupby(['Weekdays', 'Weekend']).size()
###Weekdays  Weekend
##Female    Female     167   
##          Male       12             
##Male      Female      66   
##          Male        47            
count = np.array([167,66]) #How many females coming to store during weekdays as well as weekends
nobs = np.array([179,113]) #Total number of males and females are there 
##H0:- Proportion of male and females walking in to the store are same
##Ha:- Proportion of male and female walking in to the store are different
stat, pval = proportions_ztest(count, nobs,alternative='two-sided') 
#Alternative The alternative hypothesis can be either two-sided or one of the one- sided tests
#smaller means that the alternative hypothesis is prop < value
#larger means prop > value.
print('{0:0.3f}'.format(pval)) ## 0.00
# two. sided -> means checking for equal proportions of Males and Females walking in the store
# p-value = 4.769008605701679e-13 < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 
##Now H0:- (Proportion)male <= (Proportion)female
##         Ha:- (Proportion)male > (Proportion)female

stat, pval = proportions_ztest(count, nobs,alternative='larger')
print('{0:0.3f}'.format(pval))
# p-value = 0.00 < 0.05 accept null hypothesis 
# so (Proportion)male <= (Proportion)female
# So Fantalooon should continue to open store during weekdays and weekends
