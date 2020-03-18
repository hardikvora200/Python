import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pylab
import seaborn as sns
comp = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Basic Statistics Level 1\\Computer_Data.csv")
comp
comp.drop(["Unnamed: 0"],axis=1,inplace = True)
comp.mean() ## price - 2219.576610, speed-52.011024,hd-416.601694,ram-8.286947,screen-14.608723,ads-221.301007,trend-15.926985
comp.median() ###  price-2144.0, speed-50.0,hd-340.0,ram-8.0,screen-14.0,ads-246.0,trend-16.0
comp.mode() 
####Measures of Dispersion
comp.var() 
comp.std() ####  price-580.803956, speed-21.157735,hd-258.548445,ram-5.631099,screen-0.905115,ads-74.835284,trend-7.873984 
### Calculate skewness and Kurtosis
comp.skew() 
comp.kurt() 
#### Calculate the range value
range1 = max(comp['price'])-min(comp['price'])  ### 4450 
range2 = max(comp['speed'])-min(comp['speed']) ### 75
range3 = max(comp['hd'])-min(comp['hd']) ### 2020
range4 = max(comp['ram'])-min(comp['ram']) ### 30
range5 = max(comp['screen'])-min(comp['screen']) ## 3
range6 = max(comp['ads'])-min(comp['ads'])  ## 300
range7 = max(comp['trend'])-min(comp['trend'])  ###34
plt.hist(comp["price"])
plt.hist(comp["speed"])
plt.hist(comp["hd"])
plt.hist(comp["ram"])
plt.hist(comp["screen"])
plt.hist(comp["ads"])
plt.hist(comp["trend"])
plt.boxplot(comp["price"],0,"rs",0)
plt.boxplot(comp["speed"],0,"rs",0)
plt.boxplot(comp["hd"],0,"rs",0)
plt.boxplot(comp["ram"],0,"rs",0)
plt.boxplot(comp["screen"],0,"rs",0)
plt.boxplot(comp["ads"],0,"rs",0)
plt.boxplot(comp["trend"],0,"rs",0)
plt.plot(comp["speed"],comp["price"],"bo");plt.xlabel("speed");plt.ylabel("price")
plt.plot(comp["hd"],comp["price"],"bo");plt.xlabel("hd");plt.ylabel("price")
plt.plot(comp["ram"],comp["price"],"bo");plt.xlabel("ram");plt.ylabel("price")
plt.plot(comp["screen"],comp["price"],"bo");plt.xlabel("screen");plt.ylabel("price")
plt.plot(comp["ads"],comp["price"],"bo");plt.xlabel("ads");plt.ylabel("price")
plt.plot(comp["trend"],comp["price"],"bo");plt.xlabel("trend");plt.ylabel("price")
plt.plot(comp["hd"],comp["speed"],"bo");plt.xlabel("hd");plt.ylabel("speed")
plt.plot(comp["ram"],comp["speed"],"bo");plt.xlabel("ram");plt.ylabel("speed")
plt.plot(comp["screen"],comp["speed"],"bo");plt.xlabel("screen");plt.ylabel("speed")
plt.plot(comp["ads"],comp["speed"],"bo");plt.xlabel("ads");plt.ylabel("speed")
plt.plot(comp["trend"],comp["speed"],"bo");plt.xlabel("trend");plt.ylabel("speed")
plt.plot(comp["ram"],comp["hd"],"bo");plt.xlabel("ram");plt.ylabel("hd")
plt.plot(comp["screen"],comp["hd"],"bo");plt.xlabel("screen");plt.ylabel("hd")
plt.plot(comp["ads"],comp["hd"],"bo");plt.xlabel("ads");plt.ylabel("hd")
plt.plot(comp["trend"],comp["hd"],"bo");plt.xlabel("trend");plt.ylabel("hd")
plt.plot(comp["screen"],comp["ram"],"bo");plt.xlabel("screen");plt.ylabel("ram")
plt.plot(comp["ads"],comp["ram"],"bo");plt.xlabel("ads");plt.ylabel("ram")
plt.plot(comp["trend"],comp["ram"],"bo");plt.xlabel("trend");plt.ylabel("ram")
plt.plot(comp["ads"],comp["screen"],"bo");plt.xlabel("ads");plt.ylabel("screen")
plt.plot(comp["trend"],comp["screen"],"bo");plt.xlabel("trend");plt.ylabel("screen")
plt.plot(comp["trend"],comp["ads"],"bo");plt.xlabel("trend");plt.ylabel("ads")
## Barplot
pd.crosstab(comp["speed"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["hd"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ram"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["screen"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ads"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["price"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["hd"],comp["speed"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ram"],comp["speed"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["screen"],comp["speed"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ads"],comp["speed"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["speed"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ram"],comp["hd"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["screen"],comp["hd"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ads"],comp["hd"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["hd"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["screen"],comp["ram"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ads"],comp["ram"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["ram"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["ads"],comp["screen"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["screen"]).plot(kind = "bar",width=1.85)
pd.crosstab(comp["trend"],comp["ads"]).plot(kind = "bar",width=1.85)   
#########Box plot
sns.boxplot(x="speed",y="price",data=comp)
sns.boxplot(x="hd",y="price",data=comp)
sns.boxplot(x="ram",y="price",data=comp)
sns.boxplot(x="screen",y="price",data=comp)
sns.boxplot(x="ads",y="price",data=comp)
sns.boxplot(x="trend",y="price",data=comp)
sns.boxplot(x="hd",y="speed",data=comp)
sns.boxplot(x="ram",y="speed",data=comp)
sns.boxplot(x="screen",y="speed",data=comp)
sns.boxplot(x="ads",y="speed",data=comp)
sns.boxplot(x="trend",y="speed",data=comp)
sns.boxplot(x="ram",y="hd",data=comp)
sns.boxplot(x="screen",y="hd",data=comp)
sns.boxplot(x="ads",y="hd",data=comp)
sns.boxplot(x="trend",y="hd",data=comp)
sns.boxplot(x="screen",y="ram",data=comp)
sns.boxplot(x="ads",y="ram",data=comp)
sns.boxplot(x="trend",y="ram",data=comp)
sns.boxplot(x="ads",y="screen",data=comp)
sns.boxplot(x="trend",y="screen",data=comp)
sns.boxplot(x="trend",y="ads",data=comp)
# histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(comp.iloc[:,0:7]) 
# Checking Whether data is normally distributed
stats.probplot(comp['speed'],dist="norm",plot=pylab)
stats.probplot(np.log(comp['speed']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(comp['speed']),dist="norm",plot=pylab)
stats.probplot((comp['speed'] * comp['speed']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['speed']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['speed'])*np.exp(comp['speed']),dist="norm",plot=pylab)
reci_1=1/comp['speed']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((comp['speed'] * comp['speed'])+comp['speed']),dist="norm",plot=pylab)
############################################################################################
stats.probplot(comp['hd'],dist="norm",plot=pylab)
stats.probplot(np.log(comp['hd']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(comp['hd']),dist="norm",plot=pylab)
stats.probplot((comp['hd'] * comp['hd']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['hd']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['hd'])*np.exp(comp['hd']),dist="norm",plot=pylab)
reci_2=1/comp['hd']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((comp['hd'] * comp['hd'])+comp['hd']),dist="norm",plot=pylab)
#######################################################################################
stats.probplot(comp['ram'],dist="norm",plot=pylab)
stats.probplot(np.log(comp['ram']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(comp['ram']),dist="norm",plot=pylab)
stats.probplot((comp['ram'] * comp['ram']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['ram']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['ram'])*np.exp(comp['ram']),dist="norm",plot=pylab)
reci_3=1/comp['ram']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((comp['ram'] * comp['ram'])+comp['ram']),dist="norm",plot=pylab)
######################################################################################
stats.probplot(comp['screen'],dist="norm",plot=pylab)
stats.probplot(np.log(comp['screen']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(comp['screen']),dist="norm",plot=pylab)
stats.probplot((comp['screen'] * comp['screen']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['screen']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['screen'])*np.exp(comp['screen']),dist="norm",plot=pylab)
reci_4=1/comp['screen']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((comp['screen'] * comp['screen'])+comp['screen']),dist="norm",plot=pylab)
############################################################################################
stats.probplot(comp['ads'],dist="norm",plot=pylab)
stats.probplot(np.log(comp['ads']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(comp['ads']),dist="norm",plot=pylab)
stats.probplot((comp['ads'] * comp['ads']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['ads']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['ads'])*np.exp(comp['ads']),dist="norm",plot=pylab)
reci_5=1/comp['ads']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((comp['ads'] * comp['ads'])+comp['ads']),dist="norm",plot=pylab)
###########################################################################################
stats.probplot(comp['trend'],dist="norm",plot=pylab)
stats.probplot(np.log(comp['trend']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(comp['trend']),dist="norm",plot=pylab)
stats.probplot((comp['trend'] * comp['trend']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['trend']),dist="norm",plot=pylab)
stats.probplot(np.exp(comp['trend'])*np.exp(comp['trend']),dist="norm",plot=pylab)
reci_6=1/comp['trend']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((comp['trend'] * comp['trend'])+comp['trend']),dist="norm",plot=pylab)
##########################################################################################
##########################################################################################
Q7 = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Basic Statistics Level 1\\Q7.csv")
Q7
Q7.drop(["Unnamed: 0"],axis=1,inplace = True)
Q7.mean() #### Points - 3.596563, Score- 3.217250, Weigh - 17.84875
Q7.median() #### Points - 3.695, Score- 3.325, Weigh - 17.71
Q7.mode() 
####Measures of Dispersion
Q7.var() ##### Points - 0.285881, Score - 0.957379, Weigh - 3.193166
Q7.std() ####  Points - 0.534679, Score - 0.978457, Weigh - 1.786943
### Calculate skewness and Kurtosis
Q7.skew() ##### Points - 0.292780, Score - 0.465916, Weigh - 0.406347
Q7.kurt() ##### Points - -0.450432, Score - 0.416595, Weigh - 0.864931
#### Calculate the range value
range1 = max(Q7['Points'])-min(Q7['Points'])  ### 2.17
range2 = max(Q7['Score'])-min(Q7['Score']) ### 3.91
range3 = max(Q7['Weigh'])-min(Q7['Weigh']) ### 8.39
plt.hist(Q7["Points"])
plt.hist(Q7["Score"])
plt.hist(Q7["Weigh"])
plt.boxplot(Q7["Points"],0,"rs",0)
plt.boxplot(Q7["Score"],0,"rs",0)
plt.boxplot(Q7["Weigh"],0,"rs",0)
plt.plot(Q7["Score"],Q7["Points"],"bo");plt.xlabel("Score");plt.ylabel("Points")
plt.plot(Q7["Weigh"],Q7["Points"],"bo");plt.xlabel("Weigh");plt.ylabel("Points")
plt.plot(Q7["Weigh"],Q7["Score"],"bo");plt.xlabel("Weigh");plt.ylabel("Score")
## Barplot
pd.crosstab(Q7["Score"],Q7["Points"]).plot(kind = "bar",width=1.85)
pd.crosstab(Q7["Weigh"],Q7["Points"]).plot(kind = "bar",width=1.85)
pd.crosstab(Q7["Weigh"],Q7["Score"]).plot(kind = "bar",width=1.85)
#########Box plot
sns.boxplot(x="Score",y="Points",data=Q7)
sns.boxplot(x="Weigh",y="Points",data=Q7)
sns.boxplot(x="Weigh",y="Score",data=Q7)
# histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(Q7.iloc[:,0:3])
####### Checking whether data is normally distributed
stats.probplot(Q7['Points'],dist="norm",plot=pylab)
stats.probplot(np.log(Q7['Points']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q7['Points']),dist="norm",plot=pylab)
stats.probplot((Q7['Points'] * Q7['Points']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q7['Points']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q7['Points'])*np.exp(Q7['Points']),dist="norm",plot=pylab)
reci_1=1/Q7['Points']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((Q7['Points'] * Q7['Points'])+Q7['Points']),dist="norm",plot=pylab)
stats.probplot(Q7['Score'],dist="norm",plot=pylab)
stats.probplot(np.log(Q7['Score']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q7['Score']),dist="norm",plot=pylab)
stats.probplot((Q7['Score'] * Q7['Score']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q7['Score']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q7['Score'])*np.exp(Q7['Score']),dist="norm",plot=pylab)
reci_2=1/Q7['Score']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((Q7['Score'] * Q7['Score'])+Q7['Score']),dist="norm",plot=pylab)
stats.probplot(Q7['Weigh'],dist="norm",plot=pylab)
stats.probplot(np.log(Q7['Weigh']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q7['Weigh']),dist="norm",plot=pylab)
stats.probplot((Q7['Weigh'] * Q7['Weigh']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q7['Weigh']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q7['Weigh'])*np.exp(Q7['Weigh']),dist="norm",plot=pylab)
reci_3=1/Q7['Weigh']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((Q7['Weigh'] * Q7['Weigh'])+Q7['Weigh']),dist="norm",plot=pylab)
####################################################################################
####################################################################################
Q9_a = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Basic Statistics Level 1\\Q9_a.csv")
Q9_a
Q9_a.drop(["Index"],axis=1,inplace = True)
Q9_a.mean() #### speed - 15.4, dist - 42.98
Q9_a.median() #### speed - 15.0, dist - 36.0
Q9_a.mode()  ### speed - 20, dist - 26
####Measures of Dispersion
Q9_a.var() ##### speed - 27.959184, dist - 664.060816
Q9_a.std() ####  speed - 5.287644, dist - 25.769377
### Calculate skewness and Kurtosis
Q9_a.skew() ##### speed - -0.117510, dist - 0.806895
Q9_a.kurt() ##### speed - -0.508994, dist - 0.405053
#### Calculate the range value
range1 = max(Q9_a['speed'])-min(Q9_a['speed'])  ### 21
range2 = max(Q9_a['dist'])-min(Q9_a['dist']) ### 118
plt.hist(Q9_a["speed"])
plt.hist(Q9_a["dist"])
plt.boxplot(Q9_a["speed"],0,"rs",0)
plt.boxplot(Q9_a["dist"],0,"rs",0)
plt.plot(Q9_a["speed"],Q9_a["dist"],"bo");plt.xlabel("speed");plt.ylabel("dist")
## Barplot
pd.crosstab(Q9_a["speed"],Q9_a["dist"]).plot(kind = "bar",width=1.85)
########Box plot
sns.boxplot(x="speed",y="dist",data=Q9_a)
# histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(Q9_a.iloc[:,0:2])
####### Checking whether data is normally distributed
stats.probplot(Q9_a['speed'],dist="norm",plot=pylab)
stats.probplot(np.log(Q9_a['speed']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q9_a['speed']),dist="norm",plot=pylab)
stats.probplot((Q9_a['speed'] * Q9_a['speed']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_a['speed']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_a['speed'])*np.exp(Q9_a['speed']),dist="norm",plot=pylab)
reci_1=1/Q9_a['speed']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((Q9_a['speed'] * Q9_a['speed'])+Q9_a['speed']),dist="norm",plot=pylab)
stats.probplot(Q9_a['dist'],dist="norm",plot=pylab)
stats.probplot(np.log(Q9_a['dist']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q9_a['dist']),dist="norm",plot=pylab)
stats.probplot((Q9_a['dist'] * Q9_a['dist']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_a['dist']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_a['dist'])*np.exp(Q9_a['dist']),dist="norm",plot=pylab)
reci_2=1/Q9_a['dist']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((Q9_a['dist'] * Q9_a['dist'])+Q9_a['dist']),dist="norm",plot=pylab)
####################################################################################
####################################################################################
Q9_b = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Basic Statistics Level 1\\Q9_b.csv")
Q9_b
Q9_b.drop(["Unnamed: 0"],axis=1,inplace = True)
Q9_b.mean() #### SP - 121.540272, WT - 32.412577
Q9_b.median() #### SP - 118.734518, WT - 32.734518
Q9_b.mode()  
####Measures of Dispersion
Q9_b.var() ##### SP - 201.113002, WT - 56.142247
Q9_b.std() ####  SP - 14.181432, WT - 7.492813
### Calculate skewness and Kurtosis
Q9_b.skew() ##### SP - 1.611450, WT - -0.614753
Q9_b.kurt() ##### SP - 2.977329, WT - 0.950291
#### Calculate the range value
range1 = max(Q9_b['SP'])-min(Q9_b['SP'])  ### 70.03
range2 = max(Q9_b['WT'])-min(Q9_b['WT']) ### 37.28
plt.hist(Q9_b["SP"])
plt.hist(Q9_b["WT"])
plt.boxplot(Q9_b["SP"],0,"rs",0)
plt.boxplot(Q9_b["WT"],0,"rs",0)
plt.plot(Q9_b["SP"],Q9_b["WT"],"bo");plt.xlabel("SP");plt.ylabel("WT")
## Barplot
pd.crosstab(Q9_b["SP"],Q9_b["WT"]).plot(kind = "bar",width=1.85)
########Box plot
sns.boxplot(x="SP",y="WT",data=Q9_b)
# histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(Q9_b.iloc[:,0:2])
####### Checking whether data is normally WTributed
stats.probplot(Q9_b['SP'],dist="norm",plot=pylab)
stats.probplot(np.log(Q9_b['SP']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q9_b['SP']),dist="norm",plot=pylab)
stats.probplot((Q9_b['SP'] * Q9_b['SP']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_b['SP']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_b['SP'])*np.exp(Q9_b['SP']),dist="norm",plot=pylab)
reci_1=1/Q9_b['SP']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((Q9_b['SP'] * Q9_b['SP'])+Q9_b['SP']),dist="norm",plot=pylab)
stats.probplot(Q9_b['WT'],dist="norm",plot=pylab)
stats.probplot(np.log(Q9_b['WT']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(Q9_b['WT']),dist="norm",plot=pylab)
stats.probplot((Q9_b['WT'] * Q9_b['WT']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_b['WT']),dist="norm",plot=pylab)
stats.probplot(np.exp(Q9_b['WT'])*np.exp(Q9_b['WT']),dist="norm",plot=pylab)
reci_2=1/Q9_b['WT']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((Q9_b['WT'] * Q9_b['WT'])+Q9_b['WT']),dist="norm",plot=pylab)
####################################################################################
####################################################################################
wc = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Basic Statistics Level 1\\wc-at.csv")
wc
wc.mean() #### Waist - 91.901835, AT - 101.894037
wc.median() #### Waist - 90.80, AT - 96.54
wc.mode()  
####Measures of Dispersion
wc.var() ##### Waist - 183.849626, AT - 3282.689835
wc.std() ####  Waist - 13.559116, AT - 57.294763
### Calculate skewness and Kurtosis
wc.skew() ##### Waist - 0.134056, AT - 0.584869
wc.kurt() ##### Waist - -1.102667, AT - -0.285576
#### Calculate the range value
range1 = max(wc['Waist'])-min(wc['Waist'])  ### 57.5
range2 = max(wc['AT'])-min(wc['AT']) ### 241.56
plt.hist(wc["Waist"])
plt.hist(wc["AT"])
plt.boxplot(wc["Waist"],0,"rs",0)
plt.boxplot(wc["AT"],0,"rs",0)
plt.plot(wc["Waist"],wc["AT"],"bo");plt.xlabel("Waist");plt.ylabel("AT")
## Barplot
pd.crosstab(wc["Waist"],wc["AT"]).plot(kind = "bar",width=1.85)
########Box plot
sns.boxplot(x="Waist",y="AT",data=wc)
# histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(wc.iloc[:,0:2])
####### Checking whether data is normally ATributed
stats.probplot(wc['Waist'],dist="norm",plot=pylab)
stats.probplot(np.log(wc['Waist']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wc['Waist']),dist="norm",plot=pylab)
stats.probplot((wc['Waist'] * wc['Waist']),dist="norm",plot=pylab)
stats.probplot(np.exp(wc['Waist']),dist="norm",plot=pylab)
stats.probplot(np.exp(wc['Waist'])*np.exp(wc['Waist']),dist="norm",plot=pylab)
reci_1=1/wc['Waist']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((wc['Waist'] * wc['Waist'])+wc['Waist']),dist="norm",plot=pylab)
stats.probplot(wc['AT'],dist="norm",plot=pylab)
stats.probplot(np.log(wc['AT']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wc['AT']),dist="norm",plot=pylab)
stats.probplot((wc['AT'] * wc['AT']),dist="norm",plot=pylab)
stats.probplot(np.exp(wc['AT']),dist="norm",plot=pylab)
stats.probplot(np.exp(wc['AT'])*np.exp(wc['AT']),dist="norm",plot=pylab)
reci_2=1/wc['AT']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((wc['AT'] * wc['AT'])+wc['AT']),dist="norm",plot=pylab)
