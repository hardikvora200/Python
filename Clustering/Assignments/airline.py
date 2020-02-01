import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
# reading a csv file using pandas library
airline=pd.read_excel("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Clustering\\Assignments\\EastWestAirlines.xlsx")
###airline.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
airline.columns
airline.drop(["ID#"],axis=1,inplace=True)
airline.columns
# To get the count of null values in the data 
airline.isnull().sum()
airline.shape # 3999 11 => Before dropping null values
# To drop null values ( dropping rows)
airline.dropna().shape # 50 4 => After dropping null values
#####Exploratory Data Analysis#########################################################
airline.mean() ## Balance-73601.327582,Qual_miles-144.114529,cc1_miles- 2.059515,cc2_miles-1.014504
###cc3_miles- 1.012253,Bonus_miles-17144.846212,Bonus_trans-11.601900,Flight_miles_12mo-460.055764
####Flight_trans_12- 1.373593,Days_since_enroll-4118.559390,Award- 0.370343
airline.median()
airline.mode()
####Measures of Dispersion
airline.var() 
airline.std() ## Balance-100775.664958,Qual_miles-773.663804,cc1_miles- 1.376919,cc2_miles-0.147650
###cc3_miles- 0.195241,Bonus_miles-24150.967826,Bonus_trans- 9.603810,Flight_miles_12mo-1400.209171
####Flight_trans_12-3.793172,Days_since_enroll-2065.134540,Award- 0.482957
#### Calculate the range value
range1 = max(airline['Balance'])-min(airline['Balance'])  ### 1704838
range2 = max(airline['Qual_miles'])-min(airline['Qual_miles']) ### 11148
range3 = max(airline['cc1_miles'])-min(airline['cc1_miles']) ### 4
range4 = max(airline['cc2_miles'])-min(airline['cc2_miles']) ### 2
range5 = max(airline['cc3_miles'])-min(airline['cc3_miles']) ### 4
range6 = max(airline['Bonus_miles'])-min(airline['Bonus_miles']) ### 263685
range7 = max(airline['Bonus_trans'])-min(airline['Bonus_trans']) ### 86
range8 = max(airline['Flight_miles_12mo'])-min(airline['Flight_miles_12mo']) ### 30817
range9 = max(airline['Flight_trans_12'])-min(airline['Flight_trans_12']) ### 53
range10 = max(airline['Days_since_enroll'])-min(airline['Days_since_enroll']) ### 8294
range11 = max(airline['Award'])-min(airline['Award']) ## 1
### Calculate skewness and Kurtosis
airline.skew() 
airline.kurt() 
#### Various graphs 
plt.hist(airline["Balance"])
plt.hist(airline["Qua1_miles"])
plt.hist(airline["cc1_miles"])
plt.hist(airline["cc2_miles"])
plt.hist(airline["cc3_miles"])
plt.hist(airline["Bonus_miles"])
plt.hist(airline["Bonus_trans"])
plt.hist(airline["Flight_miles_12mo"])
plt.hist(airline["Flight_trans_12"])
plt.hist(airline["Days_since_enroll"])
plt.hist(airline["Award"])
plt.boxplot(airline["Balance"],0,"rs",0)
plt.boxplot(airline["Qua1_miles"],0,"rs",0)
plt.boxplot(airline["cc1_miles"],0,"rs",0)
plt.boxplot(airline["cc2_miles"],0,"rs",0)
plt.boxplot(airline["cc3_miles"],0,"rs",0)
plt.boxplot(airline["Bonus_miles"],0,"rs",0)
plt.boxplot(airline["Bonus_trans"],0,"rs",0)
plt.boxplot(airline["Flight_miles_12mo"],0,"rs",0)
plt.boxplot(airline["Flight_trans_12"],0,"rs",0)
plt.boxplot(airline["Days_since_enroll"],0,"rs",0)
plt.boxplot(airline["Award"],0,"rs",0)
# table 
pd.crosstab(airline["Balance"],airline["Qua1_miles"])
pd.crosstab(airline["Balance"],airline["cc1_miles"])
pd.crosstab(airline["Balance"],airline["cc2_miles"])
pd.crosstab(airline["Balance"],airline["cc3_miles"])
pd.crosstab(airline["Balance"],airline["Bonus_miles"])
pd.crosstab(airline["Balance"],airline["Bonus_trans"])
pd.crosstab(airline["Balance"],airline["Flight_miles_12mo"])
pd.crosstab(airline["Balance"],airline["Flight_trans_12"])
pd.crosstab(airline["Balance"],airline["Days_since_enroll"])
pd.crosstab(airline["Balance"],airline["Award"])
pd.crosstab(airline["Qua1_miles"],airline["cc1_miles"])
pd.crosstab(airline["Qua1_miles"],airline["cc2_miles"])
pd.crosstab(airline["Qua1_miles"],airline["cc3_miles"])
pd.crosstab(airline["Qua1_miles"],airline["Bonus_miles"])
pd.crosstab(airline["Qua1_miles"],airline["Bonus_trans"])
pd.crosstab(airline["Qua1_miles"],airline["Flight_miles_12mo"])
pd.crosstab(airline["Qua1_miles"],airline["Flight_trans_12"])
pd.crosstab(airline["Qua1_miles"],airline["Days_since_enroll"])
pd.crosstab(airline["Qua1_miles"],airline["Award"])
pd.crosstab(airline["cc1_miles"],airline["cc2_miles"])
pd.crosstab(airline["cc1_miles"],airline["cc3_miles"])
pd.crosstab(airline["cc1_miles"],airline["Bonus_miles"])
pd.crosstab(airline["cc1_miles"],airline["Bonus_trans"])
pd.crosstab(airline["cc1_miles"],airline["Flight_miles_12mo"])
pd.crosstab(airline["cc1_miles"],airline["Flight_trans_12"])
pd.crosstab(airline["cc1_miles"],airline["Days_since_enroll"])
pd.crosstab(airline["cc1_miles"],airline["Award"])
pd.crosstab(airline["cc2_miles"],airline["cc3_miles"])
pd.crosstab(airline["cc2_miles"],airline["Bonus_miles"])
pd.crosstab(airline["cc2_miles"],airline["Bonus_trans"])
pd.crosstab(airline["cc2_miles"],airline["Flight_miles_12mo"])
pd.crosstab(airline["cc2_miles"],airline["Flight_trans_12"])
pd.crosstab(airline["cc2_miles"],airline["Days_since_enroll"])
pd.crosstab(airline["cc2_miles"],airline["Award"])
pd.crosstab(airline["cc3_miles"],airline["Bonus_miles"])
pd.crosstab(airline["cc3_miles"],airline["Bonus_trans"])
pd.crosstab(airline["cc3_miles"],airline["Flight_miles_12mo"])
pd.crosstab(airline["cc3_miles"],airline["Flight_trans_12"])
pd.crosstab(airline["cc3_miles"],airline["Days_since_enroll"])
pd.crosstab(airline["cc3_miles"],airline["Award"])
pd.crosstab(airline["Bonus_miles"],airline["Bonus_trans"])
pd.crosstab(airline["Bonus_miles"],airline["Flight_miles_12mo"])
pd.crosstab(airline["Bonus_miles"],airline["Flight_trans_12"])
pd.crosstab(airline["Bonus_miles"],airline["Days_since_enroll"])
pd.crosstab(airline["Bonus_miles"],airline["Award"])
pd.crosstab(airline["Bonus_trans"],airline["Flight_miles_12mo"])
pd.crosstab(airline["Bonus_trans"],airline["Flight_trans_12"])
pd.crosstab(airline["Bonus_trans"],airline["Days_since_enroll"])
pd.crosstab(airline["Bonus_trans"],airline["Award"])
pd.crosstab(airline["Flight_miles_12mo"],airline["Flight_trans_12"])
pd.crosstab(airline["Flight_miles_12mo"],airline["Days_since_enroll"])
pd.crosstab(airline["Flight_miles_12mo"],airline["Award"])
pd.crosstab(airline["Flight_trans_12"],airline["Days_since_enroll"])
pd.crosstab(airline["Flight_trans_12"],airline["Award"])
pd.crosstab(airline["Days_since_enroll"],airline["Award"])
## Barplot
pd.crosstab(airline["Balance"],airline["Qua1_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["cc1_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["cc2_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["cc3_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["Bonus_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["Bonus_trans"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Balance"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["cc1_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["cc2_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["cc3_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["Bonus_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["Bonus_trans"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Qua1_miles"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["cc2_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["cc3_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["Bonus_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["Bonus_trans"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc1_miles"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["cc3_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["Bonus_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["Bonus_trans"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc2_miles"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc3_miles"],airline["Bonus_miles"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc3_miles"],airline["Bonus_trans"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc3_miles"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc3_miles"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc3_miles"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["cc3_miles"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_miles"],airline["Bonus_trans"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_miles"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_miles"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_miles"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_miles"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_trans"],airline["Flight_miles_12mo"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_trans"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_trans"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Bonus_trans"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Flight_miles_12mo"],airline["Flight_trans_12"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Flight_miles_12mo"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Flight_miles_12mo"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Flight_trans_12"],airline["Days_since_enroll"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Flight_trans_12"],airline["Award"]).plot(kind = "bar", width = 1.85)
pd.crosstab(airline["Days_since_enroll"],airline["Award"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Balance",data=airline,palette="hls")
sns.countplot(x="Qual_miles",data=airline,palette="hls")
sns.countplot(x="cc1_miles",data=airline,palette="hls")
sns.countplot(x="cc2_miles",data=airline,palette="hls")
sns.countplot(x="cc3_miles",data=airline,palette="hls")
sns.countplot(x="Bonus_miles",data=airline,palette="hls")
sns.countplot(x="Bonus_trans",data=airline,palette="hls")
sns.countplot(x="Flight_miles_12mo",data=airline,palette="hls")
sns.countplot(x="Flight_trans_12",data=airline,palette="hls")
sns.countplot(x="Days_since_enroll",data=airline,palette="hls")
sns.countplot(x="Award",data=airline,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="Balance",y="Qual_miles",data=airline,palette="hls")
sns.boxplot(x="Balance",y="cc1_miles",data=airline,palette="hls")
sns.boxplot(x="Balance",y="cc2_miles",data=airline,palette="hls")
sns.boxplot(x="Balance",y="cc3_miles",data=airline,palette="hls")
sns.boxplot(x="Balance",y="Bonus_miles",data=airline,palette="hls")
sns.boxplot(x="Balance",y="Bonus_trans",data=airline,palette="hls")
sns.boxplot(x="Balance",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="Balance",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="Balance",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="Balance",y="Award",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="cc1_miles",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="cc2_miles",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="cc3_miles",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="Bonus_miles",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="Bonus_trans",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="Qual_miles",y="Award",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="cc2_miles",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="cc3_miles",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="Bonus_miles",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="Bonus_trans",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="cc1_miles",y="Award",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="cc3_miles",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="Bonus_miles",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="Bonus_trans",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="cc2_miles",y="Award",data=airline,palette="hls")
sns.boxplot(x="cc3_miles",y="Bonus_miles",data=airline,palette="hls")
sns.boxplot(x="cc3_miles",y="Bonus_trans",data=airline,palette="hls")
sns.boxplot(x="cc3_miles",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="cc3_miles",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="cc3_miles",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="cc3_miles",y="Award",data=airline,palette="hls")
sns.boxplot(x="Bonus_miles",y="Bonus_trans",data=airline,palette="hls")
sns.boxplot(x="Bonus_miles",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="Bonus_miles",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="Bonus_miles",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="Bonus_miles",y="Award",data=airline,palette="hls")
sns.boxplot(x="Bonus_trans",y="Flight_miles_12mo",data=airline,palette="hls")
sns.boxplot(x="Bonus_trans",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="Bonus_trans",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="Bonus_trans",y="Award",data=airline,palette="hls")
sns.boxplot(x="Flight_miles_12mo",y="Flight_trans_12",data=airline,palette="hls")
sns.boxplot(x="Flight_miles_12mo",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="Flight_miles_12mo",y="Award",data=airline,palette="hls")
sns.boxplot(x="Flight_trans_12",y="Days_since_enroll",data=airline,palette="hls")
sns.boxplot(x="Flight_trans_12",y="Award",data=airline,palette="hls")
sns.boxplot(x="Days_since_enroll",y="Award",data=airline,palette="hls")
sns.pairplot(airline.iloc[:,0:9]) # histogram of each column and scatter plot of each variable with respect to other columns
airline["Balance"].value_counts()
airline["Qual_miles"].value_counts()
airline["cc1_miles"].value_counts()
airline["cc2_miles"].value_counts()
airline["cc3_miles"].value_counts()
airline["Bonus_miles"].value_counts()
airline["Bonus_trans"].value_counts()
airline["Flight_miles_12mo"].value_counts()
airline["Flight_trans_12"].value_counts()
airline["Days_since_enroll"].value_counts()
airline["Award"].value_counts()
airline["Balance"].value_counts().plot(kind = "pie")
airline["Qual_miles"].value_counts().plot(kind = "pie")
airline["cc1_miles"].value_counts().plot(kind = "pie")
airline["cc2_miles"].value_counts().plot(kind = "pie")
airline["cc3_miles"].value_counts().plot(kind = "pie")
airline["Bonus_miles"].value_counts().plot(kind = "pie")
airline["Bonus_trans"].value_counts().plot(kind = "pie")
airline["Flight_miles_12mo"].value_counts().plot(kind = "pie")
airline["Flight_trans_12"].value_counts().plot(kind = "pie")
airline["Days_since_enroll"].value_counts().plot(kind = "pie")
airline["Award"].value_counts().plot(kind = "pie")
sns.pairplot(airline,hue="Balance",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Qual_miles",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="cc1_miles",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="cc2_miles",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="cc3_miles",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Bonus_miles",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Bonus_trans",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Flight_miles_12mo",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Flight_trans_12",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Days_since_enroll",height=1.2,diag_kind = "kde")
sns.pairplot(airline,hue="Award",height=1.2,diag_kind = "kde")
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Qual_miles","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"cc1_miles","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"cc2_miles","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"cc3_miles","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Bonus_miles","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Bonus_trans","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Flight_miles_12mo","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Flight_trans_12","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Days_since_enroll","Balance").add_legend()
sns.FacetGrid(airline,hue="Balance").map(plt.scatter,"Award","Balance").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Balance","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"cc1_miles","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"cc2_miles","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"cc3_miles","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Bonus_miles","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Bonus_trans","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Flight_miles_12mo","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Flight_trans_12","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Days_since_enroll","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="Qual_miles").map(plt.scatter,"Award","Qual_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Balance","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Qual_miles","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"cc2_miles","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"cc3_miles","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Bonus_miles","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Bonus_trans","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Flight_miles_12mo","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Flight_trans_12","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Days_since_enroll","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc1_miles").map(plt.scatter,"Award","cc1_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Balance","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Qual_miles","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"cc1_miles","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"cc3_miles","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Bonus_miles","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Bonus_trans","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Flight_miles_12mo","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Flight_trans_12","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Days_since_enroll","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc2_miles").map(plt.scatter,"Award","cc2_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Balance","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Qual_miles","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"cc1_miles","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"cc2_miles","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Bonus_miles","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Bonus_trans","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Flight_miles_12mo","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Flight_trans_12","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Days_since_enroll","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="cc3_miles").map(plt.scatter,"Award","cc3_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Balance","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Qual_miles","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"cc1_miles","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"cc2_miles","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"cc3_miles","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Bonus_trans","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Flight_miles_12mo","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Flight_trans_12","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Days_since_enroll","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_miles").map(plt.scatter,"Award","Bonus_miles").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Balance","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Qual_miles","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"cc1_miles","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"cc2_miles","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"cc3_miles","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Bonus_miles","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Flight_miles_12mo","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Flight_trans_12","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Days_since_enroll","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Bonus_trans").map(plt.scatter,"Award","Bonus_trans").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Balance","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Qual_miles","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"cc1_miles","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"cc2_miles","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"cc3_miles","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Bonus_miles","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Bonus_trans","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Flight_trans_12","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Days_since_enroll","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_miles_12mo").map(plt.scatter,"Award","Flight_miles_12mo").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Balance","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Qual_miles","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"cc1_miles","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"cc2_miles","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"cc3_miles","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Bonus_miles","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Bonus_trans","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Flight_miles_12mo","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Days_since_enroll","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Flight_trans_12").map(plt.scatter,"Award","Flight_trans_12").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Balance","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Qual_miles","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"cc1_miles","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"cc2_miles","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"cc3_miles","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Bonus_miles","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Bonus_trans","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Flight_miles_12mo","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Flight_trans_12","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Days_since_enroll").map(plt.scatter,"Award","Days_since_enroll").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Balance","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Qual_miles","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"cc1_miles","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"cc2_miles","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"cc3_miles","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Bonus_miles","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Bonus_trans","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Flight_miles_12mo","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Flight_trans_12","Award").add_legend()
sns.FacetGrid(airline,hue="Award").map(plt.scatter,"Days_since_enroll","Award").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import pylab   
# Checking Whether data is normally distributed
stats.probplot(airline['Balance'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Balance']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Balance']),dist="norm",plot=pylab)
stats.probplot((airline['Balance'] * airline['Balance']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Balance']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Balance'])*np.exp(airline['Balance']),dist="norm",plot=pylab)
reci_1=1/airline['Balance']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((airline['Balance'] * airline['Balance'])+airline['Balance']),dist="norm",plot=pylab)
stats.probplot(airline['Qual_miles'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Qual_miles']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Qual_miles']),dist="norm",plot=pylab)
stats.probplot((airline['Qual_miles'] * airline['Qual_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Qual_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Qual_miles'])*np.exp(airline['Qual_miles']),dist="norm",plot=pylab)
reci_2=1/airline['Qual_miles']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((airline['Qual_miles'] * airline['Qual_miles'])+airline['Qual_miles']),dist="norm",plot=pylab)
stats.probplot(airline['cc1_miles'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['cc1_miles']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['cc1_miles']),dist="norm",plot=pylab)
stats.probplot((airline['cc1_miles'] * airline['cc1_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['cc1_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['cc1_miles'])*np.exp(airline['cc1_miles']),dist="norm",plot=pylab)
reci_3=1/airline['cc1_miles']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((airline['cc1_miles'] * airline['cc1_miles'])+airline['cc1_miles']),dist="norm",plot=pylab)
stats.probplot(airline['cc2_miles'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['cc2_miles']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['cc2_miles']),dist="norm",plot=pylab)
stats.probplot((airline['cc2_miles'] * airline['cc2_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['cc2_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['cc2_miles'])*np.exp(airline['cc2_miles']),dist="norm",plot=pylab)
reci_4=1/airline['cc2_miles']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((airline['cc2_miles'] * airline['cc2_miles'])+airline['cc2_miles']),dist="norm",plot=pylab)
stats.probplot(airline['cc3_miles'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['cc3_miles']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['cc3_miles']),dist="norm",plot=pylab)
stats.probplot((airline['cc3_miles'] * airline['cc3_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['cc3_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['cc3_miles'])*np.exp(airline['cc3_miles']),dist="norm",plot=pylab)
reci_5=1/airline['cc3_miles']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((airline['cc3_miles'] * airline['cc3_miles'])+airline['cc3_miles']),dist="norm",plot=pylab)
stats.probplot(airline['Bonus_miles'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Bonus_miles']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Bonus_miles']),dist="norm",plot=pylab)
stats.probplot((airline['Bonus_miles'] * airline['Bonus_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Bonus_miles']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Bonus_miles'])*np.exp(airline['Bonus_miles']),dist="norm",plot=pylab)
reci_6=1/airline['Bonus_miles']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((airline['Bonus_miles'] * airline['Bonus_miles'])+airline['Bonus_miles']),dist="norm",plot=pylab)
stats.probplot(airline['Bonus_trans'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Bonus_trans']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Bonus_trans']),dist="norm",plot=pylab)
stats.probplot((airline['Bonus_trans'] * airline['Bonus_trans']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Bonus_trans']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Bonus_trans'])*np.exp(airline['Bonus_trans']),dist="norm",plot=pylab)
reci_7=1/airline['Bonus_trans']
reci_7_2=reci_7 * reci_7
reci_7_4=reci_7_2 * reci_7_2
stats.probplot(reci_7*reci_7,dist="norm",plot=pylab)
stats.probplot(reci_7_2,dist="norm",plot=pylab)
stats.probplot(reci_7_4,dist="norm",plot=pylab)
stats.probplot(reci_7_4*reci_7_4,dist="norm",plot=pylab)
stats.probplot((reci_7_4*reci_7_4)*(reci_7_4*reci_7_4),dist="norm",plot=pylab)
stats.probplot(((airline['Bonus_trans'] * airline['Bonus_trans'])+airline['Bonus_trans']),dist="norm",plot=pylab)
stats.probplot(airline['Flight_miles_12mo'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Flight_miles_12mo']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Flight_miles_12mo']),dist="norm",plot=pylab)
stats.probplot((airline['Flight_miles_12mo'] * airline['Flight_miles_12mo']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Flight_miles_12mo']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Flight_miles_12mo'])*np.exp(airline['Flight_miles_12mo']),dist="norm",plot=pylab)
reci_8=1/airline['Flight_miles_12mo']
reci_8_2=reci_8 * reci_8
reci_8_4=reci_8_2 * reci_8_2
stats.probplot(reci_8*reci_8,dist="norm",plot=pylab)
stats.probplot(reci_8_2,dist="norm",plot=pylab)
stats.probplot(reci_8_4,dist="norm",plot=pylab)
stats.probplot(reci_8_4*reci_8_4,dist="norm",plot=pylab)
stats.probplot((reci_8_4*reci_8_4)*(reci_8_4*reci_8_4),dist="norm",plot=pylab)
stats.probplot(((airline['Flight_miles_12mo'] * airline['Flight_miles_12mo'])+airline['Flight_miles_12mo']),dist="norm",plot=pylab)
stats.probplot(airline['Flight_trans_12'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Flight_trans_12']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Flight_trans_12']),dist="norm",plot=pylab)
stats.probplot((airline['Flight_trans_12'] * airline['Flight_trans_12']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Flight_trans_12']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Flight_trans_12'])*np.exp(airline['Flight_trans_12']),dist="norm",plot=pylab)
reci_9=1/airline['Flight_trans_12']
reci_9_2=reci_9 * reci_9
reci_9_4=reci_9_2 * reci_9_2
stats.probplot(reci_9*reci_9,dist="norm",plot=pylab)
stats.probplot(reci_9_2,dist="norm",plot=pylab)
stats.probplot(reci_9_4,dist="norm",plot=pylab)
stats.probplot(reci_9_4*reci_9_4,dist="norm",plot=pylab)
stats.probplot((reci_9_4*reci_9_4)*(reci_9_4*reci_9_4),dist="norm",plot=pylab)
stats.probplot(((airline['Flight_trans_12'] * airline['Flight_trans_12'])+airline['Flight_trans_12']),dist="norm",plot=pylab)
stats.probplot(airline['Days_since_enroll'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Days_since_enroll']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Days_since_enroll']),dist="norm",plot=pylab)
stats.probplot((airline['Days_since_enroll'] * airline['Days_since_enroll']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Days_since_enroll']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Days_since_enroll'])*np.exp(airline['Days_since_enroll']),dist="norm",plot=pylab)
reci_10=1/airline['Days_since_enroll']
reci_10_2=reci_10 * reci_10
reci_10_4=reci_10_2 * reci_10_2
stats.probplot(reci_10*reci_10,dist="norm",plot=pylab)
stats.probplot(reci_10_2,dist="norm",plot=pylab)
stats.probplot(reci_10_4,dist="norm",plot=pylab)
stats.probplot(reci_10_4*reci_10_4,dist="norm",plot=pylab)
stats.probplot((reci_10_4*reci_10_4)*(reci_10_4*reci_10_4),dist="norm",plot=pylab)
stats.probplot(((airline['Days_since_enroll'] * airline['Days_since_enroll'])+airline['Days_since_enroll']),dist="norm",plot=pylab)
stats.probplot(airline['Award'],dist="norm",plot=pylab)
stats.probplot(np.log(airline['Award']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(airline['Award']),dist="norm",plot=pylab)
stats.probplot((airline['Award'] * airline['Award']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Award']),dist="norm",plot=pylab)
stats.probplot(np.exp(airline['Award'])*np.exp(airline['Award']),dist="norm",plot=pylab)
reci_11=1/airline['Award']
reci_11_2=reci_11 * reci_11
reci_11_4=reci_11_2 * reci_11_2
stats.probplot(reci_11*reci_11,dist="norm",plot=pylab)
stats.probplot(reci_11_2,dist="norm",plot=pylab)
stats.probplot(reci_11_4,dist="norm",plot=pylab)
stats.probplot(reci_11_4*reci_11_4,dist="norm",plot=pylab)
stats.probplot((reci_11_4*reci_11_4)*(reci_11_4*reci_11_4),dist="norm",plot=pylab)
stats.probplot(((airline['Award'] * airline['Award'])+airline['Award']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Balance
stats.norm.ppf(0.975,73601.327582,100775.664958)# similar to qnorm in R ---- 271118
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Balance"],73601.327582,100775.664958) # similar to pnorm in R 
#### Qua1_miles
stats.norm.ppf(0.975,144.114529,773.663804)# similar to qnorm in R ---- 1660.4677209822555
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Qua1_miles"],144.114529,773.663804) # similar to pnorm in R 
#### cc1_miles
stats.norm.ppf(0.975,2.059515,1.376919)# similar to qnorm in R ----  4.758226649628907
# cdf => cumulative distributive function 
stats.norm.cdf(airline["cc1_miles"],2.059515,1.376919) # similar to pnorm in R 
#### cc2_miles
stats.norm.ppf(0.975, 1.014504, 0.147650)# similar to qnorm in R ---- 1.303892
# cdf => cumulative distributive function 
stats.norm.cdf(airline["cc2_miles"], 1.014504,0.147650) # similar to pnorm in R 
#### cc3_miles
stats.norm.ppf(0.975, 1.012253, 0.195241)# similar to qnorm in R ----  1.3949183283055848
# cdf => cumulative distributive function 
stats.norm.cdf(airline["cc3_miles"], 1.012253,0.195241) # similar to pnorm in R 
#### Bonus_miles
stats.norm.ppf(0.975, 17144.846212, 24150.967826)# similar to qnorm in R ----  64479.8733427456
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Bonus_miles"], 17144.846212,24150.967826) # similar to pnorm in R 
#### Bonus_trans
stats.norm.ppf(0.975, 11.601900, 9.603810)# similar to qnorm in R ----  30.425021714365617
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Bonus_trans"], 11.601900,9.603810) # similar to pnorm in R 
#### Flight_miles_12mo
stats.norm.ppf(0.975, 460.055764, 1400.209171)# similar to qnorm in R ----  3204.415309982686
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Flight_miles_12mo"], 460.055764,1400.209171) # similar to pnorm in R 
#### Flight_trans_12
stats.norm.ppf(0.975, 1.373593, 3.793172)# similar to qnorm in R ----   8.808073507165766
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Flight_trans_12"], 1.373593,3.793172) # similar to pnorm in R
#### Days_since_enroll
stats.norm.ppf(0.975, 4118.559390, 2065.134540)# similar to qnorm in R ----   8166.148711629692
# cdf => cumulative distributive function 
stats.norm.cdf(airline["Days_since_enroll"], 4118.559390,2065.134540) # similar to pnorm in R
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
airline.corr(method = "pearson")
airline.corr(method = "kendall")
# to get top 6 rows
airline.head(30) # to get top n rows use cars.head(10)
airline.tail(10)
# Correlation matrix 
airline.corr()
# Scatter plot between the variables along with histograms
sns.pairplot(airline)
##### Hierarchical clustering
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airline.iloc[:,0:])
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
type(df_norm)
###p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_complete.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,1:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data1.csv",encoding="utf-8")
#####################################################################
a = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    a,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 8 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_single	=AgglomerativeClustering(n_clusters=8,	linkage='single',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_single.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data2.csv",encoding="utf-8")
######################################################################
b = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    b,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_average =AgglomerativeClustering(n_clusters=6,linkage='average',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_average.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data3.csv",encoding="utf-8")
######################################################################
c = linkage(df_norm, method="centroid",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    c,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_centroid =AgglomerativeClustering(n_clusters=5,linkage='centroid',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_centroid.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data4.csv",encoding="utf-8")
##############################################################################
d = linkage(df_norm, method="complete",metric="manhattan")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    d,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hd_complete	=AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "manhattan").fit(df_norm) 
cluster_labels=pd.Series(hd_complete.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data5.csv",encoding="utf-8")
#####################################################################
e = linkage(df_norm, method="single",metric="manhattan")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    e,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 10 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
he_single = AgglomerativeClustering(n_clusters=10,linkage='single',affinity = "manhattan").fit(df_norm) 
cluster_labels=pd.Series(he_single.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data6.csv",encoding="utf-8")
######################################################################
f = linkage(df_norm, method="average",metric="manhattan")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    f,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 7 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hf_average =AgglomerativeClustering(n_clusters=7,linkage='average',affinity = "manhattan").fit(df_norm) 
cluster_labels=pd.Series(hf_average.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data7.csv",encoding="utf-8")
######################################################################
g = linkage(df_norm, method="centroid",metric="manhattan")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    g,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 9 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hg_centroid =AgglomerativeClustering(n_clusters=9,linkage='centroid',affinity = "manhattan").fit(df_norm) 
cluster_labels=pd.Series(hg_centroid.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data8.csv",encoding="utf-8")
##########################################################################
h = linkage(df_norm, method="complete",metric="maximum")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    h,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 11 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hh_complete	=AgglomerativeClustering(n_clusters=11,	linkage='complete',affinity = "maximum").fit(df_norm) 
cluster_labels=pd.Series(hh_complete.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data9.csv",encoding="utf-8")
#####################################################################
i = linkage(df_norm, method="single",metric="maximum")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    i,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hi_single = AgglomerativeClustering(n_clusters=5,linkage='single',affinity = "maximum").fit(df_norm) 
cluster_labels=pd.Series(hi_single.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data10.csv",encoding="utf-8")
######################################################################
j = linkage(df_norm, method="average",metric="maximum")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    j,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 9 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hj_average =AgglomerativeClustering(n_clusters=9,linkage='average',affinity = "maximum").fit(df_norm) 
cluster_labels=pd.Series(hj_average.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data11.csv",encoding="utf-8")
######################################################################
k = linkage(df_norm, method="centroid",metric="maximum")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    k,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 8 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hk_centroid =AgglomerativeClustering(n_clusters=8,linkage='centroid',affinity = "maximum").fit(df_norm) 
cluster_labels=pd.Series(hk_centroid.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data12.csv",encoding="utf-8")
##############################################################################
l = linkage(df_norm, method="complete",metric="binary")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    l,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hl_complete	=AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "binary").fit(df_norm) 
cluster_labels=pd.Series(hl_complete.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data13.csv",encoding="utf-8")
#####################################################################
m = linkage(df_norm, method="single",metric="binary")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    m,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hm_single = AgglomerativeClustering(n_clusters=4,linkage='single',affinity = "binary").fit(df_norm) 
cluster_labels=pd.Series(hm_single.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data14.csv",encoding="utf-8")
######################################################################
n = linkage(df_norm, method="average",metric="binary")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    n,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hn_average =AgglomerativeClustering(n_clusters=5,linkage='average',affinity = "binary").fit(df_norm) 
cluster_labels=pd.Series(hn_average.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data15.csv",encoding="utf-8")
######################################################################
o = linkage(df_norm, method="centroid",metric="binary")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    o,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
ho_centroid =AgglomerativeClustering(n_clusters=6,linkage='centroid',affinity = "binary").fit(df_norm) 
cluster_labels=pd.Series(ho_centroid.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data16.csv",encoding="utf-8")
####################################################################################
p = linkage(df_norm, method="complete",metric="canberra")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    p,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 7 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hp_complete	=AgglomerativeClustering(n_clusters=7,	linkage='complete',affinity = "canberra").fit(df_norm) 
cluster_labels=pd.Series(hp_complete.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data17.csv",encoding="utf-8")
#####################################################################
q = linkage(df_norm, method="single",metric="canberra")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    q,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hq_single = AgglomerativeClustering(n_clusters=6,linkage='single',affinity = "canberra").fit(df_norm) 
cluster_labels=pd.Series(hq_single.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data18.csv",encoding="utf-8")
######################################################################
r = linkage(df_norm, method="average",metric="canberra")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    r,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hr_average =AgglomerativeClustering(n_clusters=5,linkage='average',affinity = "canberra").fit(df_norm) 
cluster_labels=pd.Series(hr_average.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data19.csv",encoding="utf-8")
######################################################################
s = linkage(df_norm, method="centroid",metric="canberra")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    s,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hs_centroid =AgglomerativeClustering(n_clusters=4,linkage='centroid',affinity = "canberra").fit(df_norm) 
cluster_labels=pd.Series(hs_centroid.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data20.csv",encoding="utf-8")
######################################################################
t = linkage(df_norm, method="complete",metric="minkowski")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    t,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 7 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
ht_complete	=AgglomerativeClustering(n_clusters=7,	linkage='complete',affinity = "minkowski").fit(df_norm) 
cluster_labels=pd.Series(ht_complete.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data21.csv",encoding="utf-8")
#####################################################################
u = linkage(df_norm, method="single",metric="minkowski")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    u,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hu_single = AgglomerativeClustering(n_clusters=6,linkage='single',affinity = "minkowski").fit(df_norm) 
cluster_labels=pd.Series(hu_single.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data22.csv",encoding="utf-8")
######################################################################
v = linkage(df_norm, method="average",metric="minkowski")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    v,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 12 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hv_average =AgglomerativeClustering(n_clusters=12,linkage='average',affinity = "minkowski").fit(df_norm) 
cluster_labels=pd.Series(hv_average.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data23.csv",encoding="utf-8")
######################################################################
w = linkage(df_norm, method="centroid",metric="minkowski")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    w,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 6 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
hw_centroid =AgglomerativeClustering(n_clusters=6,linkage='centroid',affinity = "minkowski").fit(df_norm) 
cluster_labels=pd.Series(hw_centroid.labels_)
airline['clust']=cluster_labels # creating a  new column and assigning it to new column 
airline = airline.iloc[:,:]
airline.head()
# getting aggregate mean of each cluster
airline.iloc[:,:].groupby(airline.clust).median()
# creating a csv file 
airline.to_csv("airline_data24.csv",encoding="utf-8")
############### Lets see kmeans clustering ############################
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
# Generating random uniform numbers 
X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=5).fit(df_xy)
model1.labels_
model1.cluster_centers_
df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)
#####Our objective is to find what is happening behind the scenes
###Elbow curve and k~sqrt(n/2) will decide the 'k' value
###In this way the distance between each and every data point for each and every centroid
## of cluster is calculated
##To which cluster centroid, a specific data point is close to it, it will go and form a cluster with it
###### scree plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
###airline.drop(["clust"],axis=1,inplace=True)
model.fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
airline['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()
airline = airline.iloc[:,:]
airline.iloc[:,:].groupby(airline.clust).mean()
airline.to_csv("Airline_kmeans.csv")