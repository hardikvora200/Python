import pandas as pd
import scipy
#Chi-Square test 
BuyerRatio=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\Assignments\\BuyerRatio.csv")
BuyerRatio.head()
df = pd.DataFrame(BuyerRatio)
## H0:- Proportions of male-female buyer rations is similar in across four regions
## Ha:- Proportion of male-female buyer rations is different in at least one region
scipy.stats.chisquare(BuyerRatio["East"].value_counts())
scipy.stats.chisquare(BuyerRatio["West"].value_counts())
scipy.stats.chisquare(BuyerRatio["North"].value_counts())
scipy.stats.chisquare(BuyerRatio["South"].value_counts())
cont=pd.crosstab(BuyerRatio["East"],BuyerRatio["West"])
Chisquares_results=scipy.stats.chi2_contingency(cont)    
Chi_pValue=Chisquares_results[1]
print("p-value is: "+str(Chi_pValue))
cont1=pd.crosstab(BuyerRatio["North"],BuyerRatio["South"])
Chisquares_results1=scipy.stats.chi2_contingency(cont1) 
Chi_pValue1=Chisquares_results1[1]
print("p-value is: "+str(Chi_pValue1))   
cont2=pd.crosstab(BuyerRatio["East"],BuyerRatio["North"])
Chisquares_results2=scipy.stats.chi2_contingency(cont2)   
Chi_pValue2=Chisquares_results2[1]
print("p-value is: "+str(Chi_pValue2))   
cont3=pd.crosstab(BuyerRatio["East"],BuyerRatio["South"])
Chisquares_results3=scipy.stats.chi2_contingency(cont3)  
Chi_pValue3=Chisquares_results3[1]
print("p-value is: "+str(Chi_pValue3)) 
cont4=pd.crosstab(BuyerRatio["West"],BuyerRatio["North"])
Chisquares_results4=scipy.stats.chi2_contingency(cont4)  
Chi_pValue4=Chisquares_results4[1]
print("p-value is: "+str(Chi_pValue4)) 
cont5=pd.crosstab(BuyerRatio["West"],BuyerRatio["South"])
Chisquares_results5=scipy.stats.chi2_contingency(cont5)  
Chi_pValue5=Chisquares_results5[1]
print("p-value is: "+str(Chi_pValue5)) 
###P-value(Pearson) > 0.05 that implies Accept H0 which states that Proportion of male-female buyer rations is similar across 4 regions