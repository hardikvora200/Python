import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
# reading a csv file using pandas library
glass=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\KNN\\Assignments\\glass.csv")
glass.columns
# To get the count of null values in the data 
glass.isnull().sum()
glass.shape # 214 10 => Before dropping null values
# To drop null values ( dropping rows)
glass.dropna().shape # 214 10 => After dropping null values
#####Exploratory Data Analysis#########################################################
glass.mean() ## RI-1.518365,Na-13.407850,Mg-2.684533,Al-1.444907,
### Si-72.650935,K-0.497056,Ca-8.956963,Ba-0.175047,Fe-0.057009,Type-2.780374
glass.median() 
glass.mode() 
####Measures of Dispersion
glass.var() 
glass.std() ##RI-0.003037,Na-0.816604,Mg-1.442408,Al-0.499270,Si-0.774546,
###K-0.652192,Ca-1.423153,Ba-0.497219,Fe-0.097439,Type-2.103739
#### Calculate the range value
range1 = max(glass['RI'])-min(glass['RI'])  ### 0.0227
range2 = max(glass['Na'])-min(glass['Na']) ### 6.65
range3 = max(glass['Mg'])-min(glass['Mg']) ### 4.49
range4 = max(glass['Al'])-min(glass['Al']) ### 3.21
range5 = max(glass['Si'])-min(glass['Si'])  ### 5.59
range6 = max(glass['K'])-min(glass['K']) ### 6.21
range7 = max(glass['Ca'])-min(glass['Ca']) ### 10.76
range8 = max(glass['Ba'])-min(glass['Ba']) ### 3.15
range9 = max(glass['Fe'])-min(glass['Fe'])  ### 0.51
range10 = max(glass['Type'])-min(glass['Type']) ### 6
### Calculate skewness and Kurtosis
glass.skew() 
glass.kurt() 
#### Various graphs 
plt.hist(glass["RI"])
plt.hist(glass["Na"])
plt.hist(glass["Mg"])
plt.hist(glass["Al"])
plt.hist(glass["Si"])
plt.hist(glass["K"])
plt.hist(glass["Ca"])
plt.hist(glass["Ba"])
plt.hist(glass["Fe"])
plt.hist(glass["Type"])
plt.boxplot(glass["RI"],0,"rs",0)
plt.boxplot(glass["Na"],0,"rs",0)
plt.boxplot(glass["Mg"],0,"rs",0)
plt.boxplot(glass["Al"],0,"rs",0)
plt.boxplot(glass["Si"],0,"rs",0)
plt.boxplot(glass["K"],0,"rs",0)
plt.boxplot(glass["Ca"],0,"rs",0)
plt.boxplot(glass["Ba"],0,"rs",0)
plt.boxplot(glass["Fe"],0,"rs",0)
plt.boxplot(glass["Type"],0,"rs",0)
# table 
pd.crosstab(glass["RI"],glass["Na"])
pd.crosstab(glass["RI"],glass["Mg"])
pd.crosstab(glass["RI"],glass["Al"])
pd.crosstab(glass["RI"],glass["Si"])
pd.crosstab(glass["RI"],glass["K"])
pd.crosstab(glass["RI"],glass["Ca"])
pd.crosstab(glass["RI"],glass["Ba"])
pd.crosstab(glass["RI"],glass["Fe"])
pd.crosstab(glass["RI"],glass["Type"])
pd.crosstab(glass["Na"],glass["Mg"])
pd.crosstab(glass["Na"],glass["Al"])
pd.crosstab(glass["Na"],glass["Si"])
pd.crosstab(glass["Na"],glass["K"])
pd.crosstab(glass["Na"],glass["Ca"])
pd.crosstab(glass["Na"],glass["Ba"])
pd.crosstab(glass["Na"],glass["Fe"])
pd.crosstab(glass["Na"],glass["Type"])
pd.crosstab(glass["Mg"],glass["Al"])
pd.crosstab(glass["Mg"],glass["Si"])
pd.crosstab(glass["Mg"],glass["K"])
pd.crosstab(glass["Mg"],glass["Ca"])
pd.crosstab(glass["Mg"],glass["Ba"])
pd.crosstab(glass["Mg"],glass["Fe"])
pd.crosstab(glass["Mg"],glass["Type"])
pd.crosstab(glass["Al"],glass["Si"])
pd.crosstab(glass["Al"],glass["K"])
pd.crosstab(glass["Al"],glass["Ca"])
pd.crosstab(glass["Al"],glass["Ba"])
pd.crosstab(glass["Al"],glass["Fe"])
pd.crosstab(glass["Al"],glass["Type"])
pd.crosstab(glass["Si"],glass["K"])
pd.crosstab(glass["Si"],glass["Ca"])
pd.crosstab(glass["Si"],glass["Ba"])
pd.crosstab(glass["Si"],glass["Fe"])
pd.crosstab(glass["Si"],glass["Type"])
pd.crosstab(glass["K"],glass["Ca"])
pd.crosstab(glass["K"],glass["Ba"])
pd.crosstab(glass["K"],glass["Fe"])
pd.crosstab(glass["K"],glass["Type"])
pd.crosstab(glass["Ca"],glass["Ba"])
pd.crosstab(glass["Ca"],glass["Fe"])
pd.crosstab(glass["Ca"],glass["Type"])
pd.crosstab(glass["Ba"],glass["Fe"])
pd.crosstab(glass["Ba"],glass["Type"])
pd.crosstab(glass["Fe"],glass["Type"])
## Barplot
pd.crosstab(glass["RI"],glass["Na"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Mg"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Al"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Si"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["K"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Ca"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["RI"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Mg"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Al"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Si"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["K"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Ca"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Na"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["Al"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["Si"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["K"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["Ca"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Mg"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Al"],glass["Si"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Al"],glass["K"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Al"],glass["Ca"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Al"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Al"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Al"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Si"],glass["K"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Si"],glass["Ca"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Si"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Si"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Si"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["K"],glass["Ca"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["K"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["K"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["K"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Ca"],glass["Ba"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Ca"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Ca"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Ba"],glass["Fe"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Ba"],glass["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(glass["Fe"],glass["Type"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="RI",data=glass,palette="hls")
sns.countplot(x="Na",data=glass,palette="hls")
sns.countplot(x="Mg",data=glass,palette="hls")
sns.countplot(x="Al",data=glass,palette="hls")
sns.countplot(x="Si",data=glass,palette="hls")
sns.countplot(x="K",data=glass,palette="hls")
sns.countplot(x="Ca",data=glass,palette="hls")
sns.countplot(x="Ba",data=glass,palette="hls")
sns.countplot(x="Fe",data=glass,palette="hls")
sns.countplot(x="Type",data=glass,palette="hls")
###Boxplot
sns.boxplot(x="RI",y="Na",data=glass,palette="hls")
sns.boxplot(x="RI",y="Mg",data=glass,palette="hls")
sns.boxplot(x="RI",y="Al",data=glass,palette="hls")
sns.boxplot(x="RI",y="Si",data=glass,palette="hls")
sns.boxplot(x="RI",y="K",data=glass,palette="hls")
sns.boxplot(x="RI",y="Ca",data=glass,palette="hls")
sns.boxplot(x="RI",y="Ba",data=glass,palette="hls")
sns.boxplot(x="RI",y="Fe",data=glass,palette="hls")
sns.boxplot(x="RI",y="Type",data=glass,palette="hls")
sns.boxplot(x="Na",y="Mg",data=glass,palette="hls")
sns.boxplot(x="Na",y="Al",data=glass,palette="hls")
sns.boxplot(x="Na",y="Si",data=glass,palette="hls")
sns.boxplot(x="Na",y="K",data=glass,palette="hls")
sns.boxplot(x="Na",y="Ca",data=glass,palette="hls")
sns.boxplot(x="Na",y="Ba",data=glass,palette="hls")
sns.boxplot(x="Na",y="Fe",data=glass,palette="hls")
sns.boxplot(x="Na",y="Type",data=glass,palette="hls")
sns.boxplot(x="Mg",y="Al",data=glass,palette="hls")
sns.boxplot(x="Mg",y="Si",data=glass,palette="hls")
sns.boxplot(x="Mg",y="K",data=glass,palette="hls")
sns.boxplot(x="Mg",y="Ca",data=glass,palette="hls")
sns.boxplot(x="Mg",y="Ba",data=glass,palette="hls")
sns.boxplot(x="Mg",y="Fe",data=glass,palette="hls")
sns.boxplot(x="Mg",y="Type",data=glass,palette="hls")
sns.boxplot(x="Al",y="Si",data=glass,palette="hls")
sns.boxplot(x="Al",y="K",data=glass,palette="hls")
sns.boxplot(x="Al",y="Ca",data=glass,palette="hls")
sns.boxplot(x="Al",y="Ba",data=glass,palette="hls")
sns.boxplot(x="Al",y="Fe",data=glass,palette="hls")
sns.boxplot(x="Al",y="Type",data=glass,palette="hls")
sns.boxplot(x="Si",y="K",data=glass,palette="hls")
sns.boxplot(x="Si",y="Ca",data=glass,palette="hls")
sns.boxplot(x="Si",y="Ba",data=glass,palette="hls")
sns.boxplot(x="Si",y="Fe",data=glass,palette="hls")
sns.boxplot(x="Si",y="Type",data=glass,palette="hls")
sns.boxplot(x="K",y="Ca",data=glass,palette="hls")
sns.boxplot(x="K",y="Ba",data=glass,palette="hls")
sns.boxplot(x="K",y="Fe",data=glass,palette="hls")
sns.boxplot(x="K",y="Type",data=glass,palette="hls")
sns.boxplot(x="Ca",y="Ba",data=glass,palette="hls")
sns.boxplot(x="Ca",y="Fe",data=glass,palette="hls")
sns.boxplot(x="Ca",y="Type",data=glass,palette="hls")
sns.boxplot(x="Ba",y="Fe",data=glass,palette="hls")
sns.boxplot(x="Ba",y="Type",data=glass,palette="hls")
sns.boxplot(x="Fe",y="Type",data=glass,palette="hls")
sns.pairplot(glass.iloc[:,0:10]) # histogram of each column and scatter plot of each variable with respect to other columns
glass["RI"].value_counts()
glass["Na"].value_counts()
glass["Mg"].value_counts()
glass["Al"].value_counts()
glass["Si"].value_counts()
glass["K"].value_counts()
glass["Ca"].value_counts()
glass["Ba"].value_counts()
glass["Fe"].value_counts()
glass["Type"].value_counts()
glass["RI"].value_counts().plot(kind = "pie")
glass["Na"].value_counts().plot(kind = "pie")
glass["Mg"].value_counts().plot(kind = "pie")
glass["Al"].value_counts().plot(kind = "pie")
glass["Si"].value_counts().plot(kind = "pie")
glass["K"].value_counts().plot(kind = "pie")
glass["Ca"].value_counts().plot(kind = "pie")
glass["Ba"].value_counts().plot(kind = "pie")
glass["Fe"].value_counts().plot(kind = "pie")
glass["Type"].value_counts().plot(kind = "pie")
sns.pairplot(glass,hue="RI",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Na",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Mg",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Al",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Si",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="K",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Ca",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Ba",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Fe",height=1.2,diag_kind = "kde")
sns.pairplot(glass,hue="Type",height=1.2,diag_kind = "kde")
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Na","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Mg","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Al","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Si","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"K","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Ca","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Ba","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Fe","RI").add_legend()
sns.FacetGrid(glass,hue="RI").map(plt.scatter,"Type","RI").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Mg","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Al","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Si","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"K","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Ca","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Ba","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Fe","Na").add_legend()
sns.FacetGrid(glass,hue="Na").map(plt.scatter,"Type","Na").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"Al","Mg").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"Si","Mg").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"K","Mg").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"Ca","Mg").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"Ba","Mg").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"Fe","Mg").add_legend()
sns.FacetGrid(glass,hue="Mg").map(plt.scatter,"Type","Mg").add_legend()
sns.FacetGrid(glass,hue="Al").map(plt.scatter,"Si","Al").add_legend()
sns.FacetGrid(glass,hue="Al").map(plt.scatter,"K","Al").add_legend()
sns.FacetGrid(glass,hue="Al").map(plt.scatter,"Ca","Al").add_legend()
sns.FacetGrid(glass,hue="Al").map(plt.scatter,"Ba","Al").add_legend()
sns.FacetGrid(glass,hue="Al").map(plt.scatter,"Fe","Al").add_legend()
sns.FacetGrid(glass,hue="Al").map(plt.scatter,"Type","Al").add_legend()
sns.FacetGrid(glass,hue="Si").map(plt.scatter,"K","Si").add_legend()
sns.FacetGrid(glass,hue="Si").map(plt.scatter,"Ca","Si").add_legend()
sns.FacetGrid(glass,hue="Si").map(plt.scatter,"Ba","Si").add_legend()
sns.FacetGrid(glass,hue="Si").map(plt.scatter,"Fe","Si").add_legend()
sns.FacetGrid(glass,hue="Si").map(plt.scatter,"Type","Si").add_legend()
sns.FacetGrid(glass,hue="K").map(plt.scatter,"Ca","K").add_legend()
sns.FacetGrid(glass,hue="K").map(plt.scatter,"Ba","K").add_legend()
sns.FacetGrid(glass,hue="K").map(plt.scatter,"Fe","K").add_legend()
sns.FacetGrid(glass,hue="K").map(plt.scatter,"Type","K").add_legend()
sns.FacetGrid(glass,hue="Ca").map(plt.scatter,"Ba","Ca").add_legend()
sns.FacetGrid(glass,hue="Ca").map(plt.scatter,"Fe","Ca").add_legend()
sns.FacetGrid(glass,hue="Ca").map(plt.scatter,"Type","Ca").add_legend()
sns.FacetGrid(glass,hue="Ba").map(plt.scatter,"Fe","Ba").add_legend()
sns.FacetGrid(glass,hue="Ba").map(plt.scatter,"Type","Ba").add_legend()
sns.FacetGrid(glass,hue="Fe").map(plt.scatter,"Type","Fe").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Na").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Mg").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Al").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Si").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"K").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Ca").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="RI").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Mg").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Al").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Si").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"K").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Ca").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="Na").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"Al").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"Si").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"K").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"Ca").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="Mg").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Al").map(sns.kdeplot,"Si").add_legend()
sns.FacetGrid(glass,hue="Al").map(sns.kdeplot,"K").add_legend()
sns.FacetGrid(glass,hue="Al").map(sns.kdeplot,"Ca").add_legend()
sns.FacetGrid(glass,hue="Al").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="Al").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="Al").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Si").map(sns.kdeplot,"K").add_legend()
sns.FacetGrid(glass,hue="Si").map(sns.kdeplot,"Ca").add_legend()
sns.FacetGrid(glass,hue="Si").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="Si").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="Si").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="K").map(sns.kdeplot,"Ca").add_legend()
sns.FacetGrid(glass,hue="K").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="K").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="K").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Ca").map(sns.kdeplot,"Ba").add_legend()
sns.FacetGrid(glass,hue="Ca").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="Ca").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Ba").map(sns.kdeplot,"Fe").add_legend()
sns.FacetGrid(glass,hue="Ba").map(sns.kdeplot,"Type").add_legend()
sns.FacetGrid(glass,hue="Fe").map(sns.kdeplot,"Type").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import pylab   
# Checking Whether data is normally distributed
stats.probplot(glass['RI'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['RI']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['RI']),dist="norm",plot=pylab)
stats.probplot((glass['RI'] * glass['RI']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['RI']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['RI'])*np.exp(glass['RI']),dist="norm",plot=pylab)
reci_1=1/glass['RI']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((glass['RI'] * glass['RI'])+glass['RI']),dist="norm",plot=pylab)
stats.probplot(glass['Na'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Na']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Na']),dist="norm",plot=pylab)
stats.probplot((glass['Na'] * glass['Na']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Na']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Na'])*np.exp(glass['Na']),dist="norm",plot=pylab)
reci_2=1/glass['Na']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((glass['Na'] * glass['Na'])+glass['Na']),dist="norm",plot=pylab)
stats.probplot(glass['Mg'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Mg']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Mg']),dist="norm",plot=pylab)
stats.probplot((glass['Mg'] * glass['Mg']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Mg']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Mg'])*np.exp(glass['Mg']),dist="norm",plot=pylab)
reci_3=1/glass['Mg']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((glass['Mg'] * glass['Mg'])+glass['Mg']),dist="norm",plot=pylab)
stats.probplot(glass['Al'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Al']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Al']),dist="norm",plot=pylab)
stats.probplot((glass['Al'] * glass['Al']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Al']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Al'])*np.exp(glass['Al']),dist="norm",plot=pylab)
reci_4=1/glass['Al']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((glass['Al'] * glass['Al'])+glass['Al']),dist="norm",plot=pylab)
stats.probplot(glass['Si'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Si']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Si']),dist="norm",plot=pylab)
stats.probplot((glass['Si'] * glass['Si']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Si']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Si'])*np.exp(glass['Si']),dist="norm",plot=pylab)
reci_5=1/glass['Si']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((glass['Si'] * glass['Si'])+glass['Si']),dist="norm",plot=pylab)
stats.probplot(glass['K'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['K']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['K']),dist="norm",plot=pylab)
stats.probplot((glass['K'] * glass['K']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['K']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['K'])*np.exp(glass['K']),dist="norm",plot=pylab)
reci_6=1/glass['K']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((glass['K'] * glass['K'])+glass['K']),dist="norm",plot=pylab)
stats.probplot(glass['Ca'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Ca']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Ca']),dist="norm",plot=pylab)
stats.probplot((glass['Ca'] * glass['Ca']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Ca']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Ca'])*np.exp(glass['Ca']),dist="norm",plot=pylab)
reci_7=1/glass['Ca']
reci_7_2=reci_7 * reci_7
reci_7_4=reci_7_2 * reci_7_2
stats.probplot(reci_7*reci_7,dist="norm",plot=pylab)
stats.probplot(reci_7_2,dist="norm",plot=pylab)
stats.probplot(reci_7_4,dist="norm",plot=pylab)
stats.probplot(reci_7_4*reci_7_4,dist="norm",plot=pylab)
stats.probplot((reci_7_4*reci_7_4)*(reci_7_4*reci_7_4),dist="norm",plot=pylab)
stats.probplot(((glass['Ca'] * glass['Ca'])+glass['Ca']),dist="norm",plot=pylab)
stats.probplot(glass['Ba'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Ba']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Ba']),dist="norm",plot=pylab)
stats.probplot((glass['Ba'] * glass['Ba']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Ba']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Ba'])*np.exp(glass['Ba']),dist="norm",plot=pylab)
reci_8=1/glass['Ba']
reci_8_2=reci_8 * reci_8
reci_8_4=reci_8_2 * reci_8_2
stats.probplot(reci_8*reci_8,dist="norm",plot=pylab)
stats.probplot(reci_8_2,dist="norm",plot=pylab)
stats.probplot(reci_8_4,dist="norm",plot=pylab)
stats.probplot(reci_8_4*reci_8_4,dist="norm",plot=pylab)
stats.probplot((reci_8_4*reci_8_4)*(reci_8_4*reci_8_4),dist="norm",plot=pylab)
stats.probplot(((glass['Ba'] * glass['Ba'])+glass['Ba']),dist="norm",plot=pylab)
stats.probplot(glass['Fe'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Fe']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Fe']),dist="norm",plot=pylab)
stats.probplot((glass['Fe'] * glass['Fe']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Fe']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Fe'])*np.exp(glass['Fe']),dist="norm",plot=pylab)
reci_9=1/glass['Fe']
reci_9_2=reci_9 * reci_9
reci_9_4=reci_9_2 * reci_9_2
stats.probplot(reci_9*reci_9,dist="norm",plot=pylab)
stats.probplot(reci_9_2,dist="norm",plot=pylab)
stats.probplot(reci_9_4,dist="norm",plot=pylab)
stats.probplot(reci_9_4*reci_9_4,dist="norm",plot=pylab)
stats.probplot((reci_9_4*reci_9_4)*(reci_9_4*reci_9_4),dist="norm",plot=pylab)
stats.probplot(((glass['Fe'] * glass['Fe'])+glass['Fe']),dist="norm",plot=pylab)
stats.probplot(glass['Type'],dist="norm",plot=pylab)
stats.probplot(np.log(glass['Type']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(glass['Type']),dist="norm",plot=pylab)
stats.probplot((glass['Type'] * glass['Type']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Type']),dist="norm",plot=pylab)
stats.probplot(np.exp(glass['Type'])*np.exp(glass['Type']),dist="norm",plot=pylab)
reci_10=1/glass['Type']
reci_10_2=reci_10 * reci_10
reci_10_4=reci_10_2 * reci_10_2
stats.probplot(reci_10*reci_10,dist="norm",plot=pylab)
stats.probplot(reci_10_2,dist="norm",plot=pylab)
stats.probplot(reci_10_4,dist="norm",plot=pylab)
stats.probplot(reci_10_4*reci_10_4,dist="norm",plot=pylab)
stats.probplot((reci_10_4*reci_10_4)*(reci_10_4*reci_10_4),dist="norm",plot=pylab)
stats.probplot(((glass['Type'] * glass['Type'])+glass['Type']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### RI
stats.norm.ppf(0.975,1.518365,0.003037)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["RI"],1.518365,0.003037) # similar to pnorm in R           
#### Na
stats.norm.ppf(0.975,13.407850,0.816604)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Na"],13.407850,0.816604) # similar to pnorm in R 
#### Mg
stats.norm.ppf(0.975,2.684533,1.442408)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Mg"],2.684533,1.442408) # similar to pnorm in R 
#### Al
stats.norm.ppf(0.975,1.444907, 0.499270)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Al"], 1.444907, 0.499270) # similar to pnorm in R 
#### Si
stats.norm.ppf(0.975,72.650935,0.774546)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Si"], 72.650935,0.774546) # similar to pnorm in R 
#### K
stats.norm.ppf(0.975,0.497056,0.652192)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["K"], 0.497056, 0.652192) # similar to pnorm in R 
#### Ca
stats.norm.ppf(0.975,8.956963,1.423153)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Ca"],8.956963,1.423153) # similar to pnorm in R 
#### Ba
stats.norm.ppf(0.975,0.175047,0.497219)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Ba"],0.175047,0.497219) # similar to pnorm in R 
#### Fe
stats.norm.ppf(0.975,0.057009,0.097439)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Fe"],0.057009,0.097439) # similar to pnorm in R 
#### Type
stats.norm.ppf(0.975,2.780374,2.103739)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(glass["Type"],2.780374,2.103739) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
glass.corr(method = "pearson")
glass.corr(method = "kendall")
# to get top 30 rows
glass.head(30) # to get top n rows use glass.head(10)
glass.tail(10)
##############################################################
# Training and Test data using 
from sklearn.model_selection import train_test_split
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(glass.iloc[:,0:])
train,test = train_test_split(glass,test_size = 0.2) # 0.2 => 20 percent of entire data 
from sklearn.neighbors import KNeighborsClassifier as KNC
# for 4 nearest neighbours 
neigh = KNC(n_neighbors= 4)
# Fitting with training data 
neigh.fit(train.iloc[:,0:10],train.iloc[:,9])
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:10])==train.iloc[:,9])  ### 95.91%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:10])==test.iloc[:,9]) ### 93.02%
# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)
# fitting with training data
neigh.fit(train.iloc[:,0:10],train.iloc[:,9])
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:10])==train.iloc[:,9]) ### 95.32%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:10])==test.iloc[:,9]) #### 90.69%
# creating empty list variable 
acc = []
# running KNN algorithm for 2 to 50 nearest neighbours(even numbers) and 
# storing the accuracy values 

for i in range(2,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:10],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:10])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:10])==test.iloc[:,9])
    acc.append([train_acc,test_acc])
import matplotlib.pyplot as plt # library to do visualizations 
# train accuracy plot 
plt.plot(np.arange(2,50,2),[i[0] for i in acc],"bo-")
# test accuracy plot
plt.plot(np.arange(2,50,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])




