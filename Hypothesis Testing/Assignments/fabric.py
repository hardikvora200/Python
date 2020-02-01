#importing the pacakages which are required 
import pandas as pd
import statsmodels.api as sm
fabric=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Hypothesis Testing\\fabric.csv")
fabric_len=fabric['Fabric_length']
mu=fabric_len.mean() ###155.064
##sigma=fabric_len.std() ###5.64
### Ho- Average Fabric length= Average Fabric Length Historical
### Ha- Average Fabric length!=Average Fabric length Historical
#doing Normality test 
data=stats.shapiro(fabric.Fabric_length)
pValue=data[1]
print("p-value is: "+str(pValue))  #### 0.15 > 0.05

#Doing Anderson Darling test
from scipy.stats import anderson
result = anderson(data)
#### Fabric length is following the normal distribution
#### Here Sigma is known, so we will go with 1 Sample Z test
### H0=Average Fabric Length Current <= Average Fabric Length Historical
### Ha=Average Fabric Length Current > Average Fabric Length Historical
## 1 Sample Z test
z_crirical = 1.96 ### alpha level of 0.05
N=25
SE=sigma/np.sqrt(25) ## 1.1280939677172293
z_stat=(fabric_len-mu)/SE
sum(z_stat)
stats.norm.cdf(fabric_len,mu,sigma)
## P-value > 0.05 => P High Ho Fly
##=>Average Fabric Length Current <= Average Fabric Length Historical
#### Machines used are not acceptable..give it for maintenance service