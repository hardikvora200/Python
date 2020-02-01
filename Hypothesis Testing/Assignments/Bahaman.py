import pandas as pd
from scipy import stats
import statsmodels.api as sm

#Chi-Square test 
#Importing the data set of bahaman 
Bahaman=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\Bahaman.csv")
count=pd.crosstab(Bahaman["Defective"],Bahaman["Country"])
print("count")
###
Chisquares_results=scipy.stats.chi2_contingency(count)
Chi_pValue=Chisquares_results[1]
print("p-value is: "+str(Chi_pValue)) ###0.63
### Accept Null Hypothesis which means all countries have equal proportions of defective
