import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# reading a csv file using pandas library
wine=pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\PCA\\Assignments\\wine.csv")
##wine.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
##wine.columns
##wine.drop(["a"],axis=1,inplace=True)
##wine.columns
# To get the count of null values in the data 
wine.isnull().sum() ### Checking for sum of null value
wine.shape # 178 14 => Before dropping null values
# To drop null values ( dropping rows)
wine.dropna().shape # 178 14 => After dropping null values
#####Exploratory Data Analysis#########################################################
wine.mean() ## Type - 1.938202, Alcohol-13.000618, Malic-2.336348,Ash-2.366517,Alcalinity-19.494944,
############## Magnesium-99.741573,Phenols-2.295112,Flavanoids-2.02927,Nonflavanoids-0.361854,
############## Proanthocyanins-1.590899,Color-5.058090,Hue-0.957449,Dilution-2.611685,Proline-746.893258
wine.median() ##Type - 2.000, Alcohol-13.050, Malic-1.865,Ash-2.360,Alcalinity-19.500,
############## Magnesium-98.000,Phenols-2.355,Flavanoids-2.135,Nonflavanoids-0.340,
############## Proanthocyanins-1.555,Color-4.690,Hue-0.965,Dilution-2.780,Proline-673.5
wine.mode() 
####Measures of Dispersion
wine.var() 
wine.std() ##Type - 0.775035, Alcohol-0.811827, Malic-1.117146,Ash-0.274344,Alcalinity-3.339564,
############## Magnesium-14.282484,Phenols-0.625851,Flavanoids-0.998859,Nonflavanoids-0.124453,
############## Proanthocyanins-0.572359,Color-2.318286,Hue-0.228572,Dilution-0.709990,Proline-314.907474
#### Calculate the range value
range1 = max(wine['Type'])-min(wine['Type'])  ### 2
range2 = max(wine['Alcohol'])-min(wine['Alcohol']) ### 3.8
range3 = max(wine['Malic'])-min(wine['Malic']) ### 5.06
range4 = max(wine['Ash'])-min(wine['Ash']) ### 1.869
range5 = max(wine['Alcalinity'])-min(wine['Alcalinity'])  ##  19.4
range6 = max(wine['Magnesium'])-min(wine['Magnesium']) ### 92
range7 = max(wine['Phenols'])-min(wine['Phenols']) #### 2.9
range8 = max(wine['Flavanoids'])-min(wine['Flavanoids']) #### 4.74
range9 = max(wine['Nonflavanoids'])-min(wine['Nonflavanoids']) ###0.53
range10 = max(wine['Proanthocyanins'])-min(wine['Proanthocyanins']) #### 3.17
range11 = max(wine['Color'])-min(wine['Color']) ### 11.72
range12 = max(wine['Hue'])-min(wine['Hue']) ### 1.23
range13 = max(wine['Dilution'])-min(wine['Dilution']) ### 2.73
range14 = max(wine['Proline'])-min(wine['Proline'])  #### 1402
### Calculate skewness and Kurtosis
wine.skew() 
wine.kurt() 
####Graphidelivery_time Representation 
plt.hist(wine["Type"])
plt.hist(wine["Alcohol"])
plt.hist(wine["Malic"])
plt.hist(wine["Ash"])
plt.hist(wine["Alcalinity"])
plt.hist(wine["Magnesium"])
plt.hist(wine["Phenols"])
plt.hist(wine["Flavanoids"])
plt.hist(wine["Nonflavanoids"])
plt.hist(wine["Proanthocyanins"])
plt.hist(wine["Color"])
plt.hist(wine["Hue"])
plt.hist(wine["Dilution"])
plt.hist(wine["Proline"])
plt.boxplot(wine["Type"],0,"rs",0)
plt.boxplot(wine["Alcohol"],0,"rs",0)
plt.boxplot(wine["Malic"],0,"rs",0)
plt.boxplot(wine["Ash"],0,"rs",0)
plt.boxplot(wine["Alcalinity"],0,"rs",0)
plt.boxplot(wine["Magnesium"],0,"rs",0)
plt.boxplot(wine["Phenols"],0,"rs",0)
plt.boxplot(wine["Flavanoids"],0,"rs",0)
plt.boxplot(wine["Nonflavanoids"],0,"rs",0)
plt.boxplot(wine["Proanthocyanins"],0,"rs",0)
plt.boxplot(wine["Color"],0,"rs",0)
plt.boxplot(wine["Hue"],0,"rs",0)
plt.boxplot(wine["Dilution"],0,"rs",0)
plt.boxplot(wine["Proline"],0,"rs",0)
plt.plot(wine["Alcohol"],wine["Type"],"bo");plt.xlabel("Alcohol");plt.ylabel("Type")
plt.plot(wine["Malic"],wine["Type"],"bo");plt.xlabel("Malic");plt.ylabel("Type")
plt.plot(wine["Ash"],wine["Type"],"bo");plt.xlabel("Ash");plt.ylabel("Type")
plt.plot(wine["Alcalinity"],wine["Type"],"bo");plt.xlabel("Alcalinity");plt.ylabel("Type")
plt.plot(wine["Magnesium"],wine["Type"],"bo");plt.xlabel("Magnesium");plt.ylabel("Type")
plt.plot(wine["Phenols"],wine["Type"],"bo");plt.xlabel("Phenols");plt.ylabel("Type")
plt.plot(wine["Flavanoids"],wine["Type"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Type")
plt.plot(wine["Nonflavanoids"],wine["Type"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Type")
plt.plot(wine["Proanthocyanins"],wine["Type"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Type")
plt.plot(wine["Color"],wine["Type"],"bo");plt.xlabel("Color");plt.ylabel("Type")
plt.plot(wine["Hue"],wine["Type"],"bo");plt.xlabel("Hue");plt.ylabel("Type")
plt.plot(wine["Dilution"],wine["Type"],"bo");plt.xlabel("Dilution");plt.ylabel("Type")
plt.plot(wine["Proline"],wine["Type"],"bo");plt.xlabel("Proline");plt.ylabel("Type")
plt.plot(wine["Malic"],wine["Alcohol"],"bo");plt.xlabel("Malic");plt.ylabel("Alcohol")
plt.plot(wine["Ash"],wine["Alcohol"],"bo");plt.xlabel("Ash");plt.ylabel("Alcohol")
plt.plot(wine["Alcalinity"],wine["Alcohol"],"bo");plt.xlabel("Alcalinity");plt.ylabel("Alcohol")
plt.plot(wine["Magnesium"],wine["Alcohol"],"bo");plt.xlabel("Magnesium");plt.ylabel("Alcohol")
plt.plot(wine["Phenols"],wine["Alcohol"],"bo");plt.xlabel("Phenols");plt.ylabel("Alcohol")
plt.plot(wine["Flavanoids"],wine["Alcohol"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Alcohol")
plt.plot(wine["Nonflavanoids"],wine["Alcohol"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Alcohol")
plt.plot(wine["Proanthocyanins"],wine["Alcohol"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Alcohol")
plt.plot(wine["Color"],wine["Alcohol"],"bo");plt.xlabel("Color");plt.ylabel("Alcohol")
plt.plot(wine["Hue"],wine["Alcohol"],"bo");plt.xlabel("Hue");plt.ylabel("Alcohol")
plt.plot(wine["Dilution"],wine["Alcohol"],"bo");plt.xlabel("Dilution");plt.ylabel("Alcohol")
plt.plot(wine["Proline"],wine["Alcohol"],"bo");plt.xlabel("Proline");plt.ylabel("Alcohol")
plt.plot(wine["Ash"],wine["Malic"],"bo");plt.xlabel("Ash");plt.ylabel("Malic")
plt.plot(wine["Alcalinity"],wine["Malic"],"bo");plt.xlabel("Alcalinity");plt.ylabel("Malic")
plt.plot(wine["Magnesium"],wine["Malic"],"bo");plt.xlabel("Magnesium");plt.ylabel("Malic")
plt.plot(wine["Phenols"],wine["Malic"],"bo");plt.xlabel("Phenols");plt.ylabel("Malic")
plt.plot(wine["Flavanoids"],wine["Malic"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Malic")
plt.plot(wine["Nonflavanoids"],wine["Malic"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Malic")
plt.plot(wine["Proanthocyanins"],wine["Malic"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Malic")
plt.plot(wine["Color"],wine["Malic"],"bo");plt.xlabel("Color");plt.ylabel("Malic")
plt.plot(wine["Hue"],wine["Malic"],"bo");plt.xlabel("Hue");plt.ylabel("Malic")
plt.plot(wine["Dilution"],wine["Malic"],"bo");plt.xlabel("Dilution");plt.ylabel("Malic")
plt.plot(wine["Proline"],wine["Malic"],"bo");plt.xlabel("Proline");plt.ylabel("Malic")
plt.plot(wine["Alcalinity"],wine["Ash"],"bo");plt.xlabel("Alcalinity");plt.ylabel("Ash")
plt.plot(wine["Magnesium"],wine["Ash"],"bo");plt.xlabel("Magnesium");plt.ylabel("Ash")
plt.plot(wine["Phenols"],wine["Ash"],"bo");plt.xlabel("Phenols");plt.ylabel("Ash")
plt.plot(wine["Flavanoids"],wine["Ash"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Ash")
plt.plot(wine["Nonflavanoids"],wine["Ash"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Ash")
plt.plot(wine["Proanthocyanins"],wine["Ash"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Ash")
plt.plot(wine["Color"],wine["Ash"],"bo");plt.xlabel("Color");plt.ylabel("Ash")
plt.plot(wine["Hue"],wine["Ash"],"bo");plt.xlabel("Hue");plt.ylabel("Ash")
plt.plot(wine["Dilution"],wine["Ash"],"bo");plt.xlabel("Dilution");plt.ylabel("Ash")
plt.plot(wine["Proline"],wine["Ash"],"bo");plt.xlabel("Proline");plt.ylabel("Ash")
plt.plot(wine["Magnesium"],wine["Alcalinity"],"bo");plt.xlabel("Magnesium");plt.ylabel("Alcalinity")
plt.plot(wine["Phenols"],wine["Alcalinity"],"bo");plt.xlabel("Phenols");plt.ylabel("Alcalinity")
plt.plot(wine["Flavanoids"],wine["Alcalinity"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Alcalinity")
plt.plot(wine["Nonflavanoids"],wine["Alcalinity"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Alcalinity")
plt.plot(wine["Proanthocyanins"],wine["Alcalinity"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Alcalinity")
plt.plot(wine["Color"],wine["Alcalinity"],"bo");plt.xlabel("Color");plt.ylabel("Alcalinity")
plt.plot(wine["Hue"],wine["Alcalinity"],"bo");plt.xlabel("Hue");plt.ylabel("Alcalinity")
plt.plot(wine["Dilution"],wine["Alcalinity"],"bo");plt.xlabel("Dilution");plt.ylabel("Alcalinity")
plt.plot(wine["Proline"],wine["Alcalinity"],"bo");plt.xlabel("Proline");plt.ylabel("Alcalinity")
plt.plot(wine["Phenols"],wine["Magnesium"],"bo");plt.xlabel("Phenols");plt.ylabel("Magnesium")
plt.plot(wine["Flavanoids"],wine["Magnesium"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Magnesium")
plt.plot(wine["Nonflavanoids"],wine["Magnesium"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Magnesium")
plt.plot(wine["Proanthocyanins"],wine["Magnesium"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Magnesium")
plt.plot(wine["Color"],wine["Magnesium"],"bo");plt.xlabel("Color");plt.ylabel("Magnesium")
plt.plot(wine["Hue"],wine["Magnesium"],"bo");plt.xlabel("Hue");plt.ylabel("Magnesium")
plt.plot(wine["Dilution"],wine["Magnesium"],"bo");plt.xlabel("Dilution");plt.ylabel("Magnesium")
plt.plot(wine["Proline"],wine["Magnesium"],"bo");plt.xlabel("Proline");plt.ylabel("Magnesium")
plt.plot(wine["Flavanoids"],wine["Phenols"],"bo");plt.xlabel("Flavanoids");plt.ylabel("Phenols")
plt.plot(wine["Nonflavanoids"],wine["Phenols"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Phenols")
plt.plot(wine["Proanthocyanins"],wine["Phenols"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Phenols")
plt.plot(wine["Color"],wine["Phenols"],"bo");plt.xlabel("Color");plt.ylabel("Phenols")
plt.plot(wine["Hue"],wine["Phenols"],"bo");plt.xlabel("Hue");plt.ylabel("Phenols")
plt.plot(wine["Dilution"],wine["Phenols"],"bo");plt.xlabel("Dilution");plt.ylabel("Phenols")
plt.plot(wine["Proline"],wine["Phenols"],"bo");plt.xlabel("Proline");plt.ylabel("Phenols")
plt.plot(wine["Nonflavanoids"],wine["Flavanoids"],"bo");plt.xlabel("Nonflavanoids");plt.ylabel("Flavanoids")
plt.plot(wine["Proanthocyanins"],wine["Flavanoids"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Flavanoids")
plt.plot(wine["Color"],wine["Flavanoids"],"bo");plt.xlabel("Color");plt.ylabel("Flavanoids")
plt.plot(wine["Hue"],wine["Flavanoids"],"bo");plt.xlabel("Hue");plt.ylabel("Flavanoids")
plt.plot(wine["Dilution"],wine["Flavanoids"],"bo");plt.xlabel("Dilution");plt.ylabel("Flavanoids")
plt.plot(wine["Proline"],wine["Flavanoids"],"bo");plt.xlabel("Proline");plt.ylabel("Flavanoids")
plt.plot(wine["Proanthocyanins"],wine["Nonflavanoids"],"bo");plt.xlabel("Proanthocyanins");plt.ylabel("Nonflavanoids")
plt.plot(wine["Color"],wine["Nonflavanoids"],"bo");plt.xlabel("Color");plt.ylabel("Nonflavanoids")
plt.plot(wine["Hue"],wine["Nonflavanoids"],"bo");plt.xlabel("Hue");plt.ylabel("Nonflavanoids")
plt.plot(wine["Dilution"],wine["Nonflavanoids"],"bo");plt.xlabel("Dilution");plt.ylabel("Nonflavanoids")
plt.plot(wine["Proline"],wine["Nonflavanoids"],"bo");plt.xlabel("Proline");plt.ylabel("Nonflavanoids")
plt.plot(wine["Color"],wine["Proanthocyanins"],"bo");plt.xlabel("Color");plt.ylabel("Proanthocyanins")
plt.plot(wine["Hue"],wine["Proanthocyanins"],"bo");plt.xlabel("Hue");plt.ylabel("Proanthocyanins")
plt.plot(wine["Dilution"],wine["Proanthocyanins"],"bo");plt.xlabel("Dilution");plt.ylabel("Proanthocyanins")
plt.plot(wine["Proline"],wine["Proanthocyanins"],"bo");plt.xlabel("Proline");plt.ylabel("Proanthocyanins")
plt.plot(wine["Hue"],wine["Color"],"bo");plt.xlabel("Hue");plt.ylabel("Color")
plt.plot(wine["Dilution"],wine["Color"],"bo");plt.xlabel("Dilution");plt.ylabel("Color")
plt.plot(wine["Proline"],wine["Color"],"bo");plt.xlabel("Proline");plt.ylabel("Color")
plt.plot(wine["Dilution"],wine["Hue"],"bo");plt.xlabel("Dilution");plt.ylabel("Hue")
plt.plot(wine["Proline"],wine["Hue"],"bo");plt.xlabel("Proline");plt.ylabel("Hue")
plt.plot(wine["Proline"],wine["Dilution"],"bo");plt.xlabel("Proline");plt.ylabel("Dilution")
# table 
pd.crosstab(wine["Alcohol"],wine["Type"])
pd.crosstab(wine["Malic"],wine["Type"])
pd.crosstab(wine["Ash"],wine["Type"])
pd.crosstab(wine["Alcalinity"],wine["Type"])
pd.crosstab(wine["Magnesium"],wine["Type"])
pd.crosstab(wine["Phenols"],wine["Type"])
pd.crosstab(wine["Flavanoids"],wine["Type"])
pd.crosstab(wine["Nonflavanoids"],wine["Type"])
pd.crosstab(wine["Proanthocyanins"],wine["Type"])
pd.crosstab(wine["Color"],wine["Type"])
pd.crosstab(wine["Hue"],wine["Type"])
pd.crosstab(wine["Dilution"],wine["Type"])
pd.crosstab(wine["Proline"],wine["Type"])
pd.crosstab(wine["Malic"],wine["Alcohol"])
pd.crosstab(wine["Ash"],wine["Alcohol"])
pd.crosstab(wine["Alcalinity"],wine["Alcohol"])
pd.crosstab(wine["Magnesium"],wine["Alcohol"])
pd.crosstab(wine["Phenols"],wine["Alcohol"])
pd.crosstab(wine["Flavanoids"],wine["Alcohol"])
pd.crosstab(wine["Nonflavanoids"],wine["Alcohol"])
pd.crosstab(wine["Proanthocyanins"],wine["Alcohol"])
pd.crosstab(wine["Color"],wine["Alcohol"])
pd.crosstab(wine["Hue"],wine["Alcohol"])
pd.crosstab(wine["Dilution"],wine["Alcohol"])
pd.crosstab(wine["Proline"],wine["Alcohol"])
pd.crosstab(wine["Ash"],wine["Malic"])
pd.crosstab(wine["Alcalinity"],wine["Malic"])
pd.crosstab(wine["Magnesium"],wine["Malic"])
pd.crosstab(wine["Phenols"],wine["Malic"])
pd.crosstab(wine["Flavanoids"],wine["Malic"])
pd.crosstab(wine["Nonflavanoids"],wine["Malic"])
pd.crosstab(wine["Proanthocyanins"],wine["Malic"])
pd.crosstab(wine["Color"],wine["Malic"])
pd.crosstab(wine["Hue"],wine["Malic"])
pd.crosstab(wine["Dilution"],wine["Malic"])
pd.crosstab(wine["Proline"],wine["Malic"])
pd.crosstab(wine["Alcalinity"],wine["Ash"])
pd.crosstab(wine["Magnesium"],wine["Ash"])
pd.crosstab(wine["Phenols"],wine["Ash"])
pd.crosstab(wine["Flavanoids"],wine["Ash"])
pd.crosstab(wine["Nonflavanoids"],wine["Ash"])
pd.crosstab(wine["Proanthocyanins"],wine["Ash"])
pd.crosstab(wine["Color"],wine["Ash"])
pd.crosstab(wine["Hue"],wine["Ash"])
pd.crosstab(wine["Dilution"],wine["Ash"])
pd.crosstab(wine["Proline"],wine["Ash"])
pd.crosstab(wine["Magnesium"],wine["Alcalinity"])
pd.crosstab(wine["Phenols"],wine["Alcalinity"])
pd.crosstab(wine["Flavanoids"],wine["Alcalinity"])
pd.crosstab(wine["Nonflavanoids"],wine["Alcalinity"])
pd.crosstab(wine["Proanthocyanins"],wine["Alcalinity"])
pd.crosstab(wine["Color"],wine["Alcalinity"])
pd.crosstab(wine["Hue"],wine["Alcalinity"])
pd.crosstab(wine["Dilution"],wine["Alcalinity"])
pd.crosstab(wine["Proline"],wine["Alcalinity"])
pd.crosstab(wine["Phenols"],wine["Magnesium"])
pd.crosstab(wine["Flavanoids"],wine["Magnesium"])
pd.crosstab(wine["Nonflavanoids"],wine["Magnesium"])
pd.crosstab(wine["Proanthocyanins"],wine["Magnesium"])
pd.crosstab(wine["Color"],wine["Magnesium"])
pd.crosstab(wine["Hue"],wine["Magnesium"])
pd.crosstab(wine["Dilution"],wine["Magnesium"])
pd.crosstab(wine["Proline"],wine["Magnesium"])
pd.crosstab(wine["Flavanoids"],wine["Phenols"])
pd.crosstab(wine["Nonflavanoids"],wine["Phenols"])
pd.crosstab(wine["Proanthocyanins"],wine["Phenols"])
pd.crosstab(wine["Color"],wine["Phenols"])
pd.crosstab(wine["Hue"],wine["Phenols"])
pd.crosstab(wine["Dilution"],wine["Phenols"])
pd.crosstab(wine["Proline"],wine["Phenols"])
pd.crosstab(wine["Nonflavanoids"],wine["Flavanoids"])
pd.crosstab(wine["Proanthocyanins"],wine["Flavanoids"])
pd.crosstab(wine["Color"],wine["Flavanoids"])
pd.crosstab(wine["Hue"],wine["Flavanoids"])
pd.crosstab(wine["Dilution"],wine["Flavanoids"])
pd.crosstab(wine["Proline"],wine["Flavanoids"])
pd.crosstab(wine["Proanthocyanins"],wine["Nonflavanoids"])
pd.crosstab(wine["Color"],wine["Nonflavanoids"])
pd.crosstab(wine["Hue"],wine["Nonflavanoids"])
pd.crosstab(wine["Dilution"],wine["Nonflavanoids"])
pd.crosstab(wine["Proline"],wine["Nonflavanoids"])
pd.crosstab(wine["Color"],wine["Proanthocyanins"])
pd.crosstab(wine["Hue"],wine["Proanthocyanins"])
pd.crosstab(wine["Dilution"],wine["Proanthocyanins"])
pd.crosstab(wine["Proline"],wine["Proanthocyanins"])
pd.crosstab(wine["Hue"],wine["Color"])
pd.crosstab(wine["Dilution"],wine["Color"])
pd.crosstab(wine["Proline"],wine["Color"])
pd.crosstab(wine["Dilution"],wine["Hue"])
pd.crosstab(wine["Proline"],wine["Hue"])
pd.crosstab(wine["Proline"],wine["Dilution"])
## Barplot
pd.crosstab(wine["Alcohol"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Malic"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Ash"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Alcalinity"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Magnesium"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Phenols"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Type"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Malic"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Ash"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Alcalinity"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Magnesium"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Phenols"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Alcohol"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Ash"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Alcalinity"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Magnesium"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Phenols"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Malic"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Alcalinity"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Magnesium"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Phenols"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Ash"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Magnesium"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Phenols"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Alcalinity"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Phenols"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Magnesium"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Flavanoids"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Phenols"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Nonflavanoids"],wine["Flavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Flavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Flavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Flavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Flavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Flavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proanthocyanins"],wine["Nonflavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Nonflavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Nonflavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Nonflavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Nonflavanoids"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Color"],wine["Proanthocyanins"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Proanthocyanins"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Proanthocyanins"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Proanthocyanins"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Hue"],wine["Color"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Color"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Color"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Dilution"],wine["Hue"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Hue"]).plot(kind = "bar", width = 1.85)
pd.crosstab(wine["Proline"],wine["Dilution"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Type",data=wine,palette="hls")
sns.countplot(x="Alcohol",data=wine,palette="hls")
sns.countplot(x="Malic",data=wine,palette="hls")
sns.countplot(x="Ash",data=wine,palette="hls")
sns.countplot(x="Alcalinity",data=wine,palette="hls")
sns.countplot(x="Magnesium",data=wine,palette="hls")
sns.countplot(x="Phenols",data=wine,palette="hls")
sns.countplot(x="Flavanoids",data=wine,palette="hls")
sns.countplot(x="Nonflavanoids",data=wine,palette="hls")
sns.countplot(x="Proanthocyanins",data=wine,palette="hls")
sns.countplot(x="Color",data=wine,palette="hls")
sns.countplot(x="Hue",data=wine,palette="hls")
sns.countplot(x="Dilution",data=wine,palette="hls")
sns.countplot(x="Proline",data=wine,palette="hls")
sns.boxplot(x="Alcohol",y="Type",data=wine,palette="hls")
sns.boxplot(x="Malic",y="Type",data=wine,palette="hls")
sns.boxplot(x="Ash",y="Type",data=wine,palette="hls")
sns.boxplot(x="Alcalinity",y="Type",data=wine,palette="hls")
sns.boxplot(x="Magnesium",y="Type",data=wine,palette="hls")
sns.boxplot(x="Phenols",y="Type",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Type",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Type",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Type",data=wine,palette="hls")
sns.boxplot(x="Color",y="Type",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Type",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Type",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Type",data=wine,palette="hls")
sns.boxplot(x="Malic",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Ash",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Alcalinity",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Magnesium",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Phenols",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Color",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Alcohol",data=wine,palette="hls")
sns.boxplot(x="Ash",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Alcalinity",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Magnesium",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Phenols",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Color",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Malic",data=wine,palette="hls")
sns.boxplot(x="Alcalinity",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Magnesium",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Phenols",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Color",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Ash",data=wine,palette="hls")
sns.boxplot(x="Magnesium",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Phenols",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Color",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Alcalinity",data=wine,palette="hls")
sns.boxplot(x="Phenols",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Color",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Magnesium",data=wine,palette="hls")
sns.boxplot(x="Flavanoids",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Color",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Phenols",data=wine,palette="hls")
sns.boxplot(x="Nonflavanoids",y="Flavanoids",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Flavanoids",data=wine,palette="hls")
sns.boxplot(x="Color",y="Flavanoids",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Flavanoids",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Flavanoids",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Flavanoids",data=wine,palette="hls")
sns.boxplot(x="Proanthocyanins",y="Nonflavanoids",data=wine,palette="hls")
sns.boxplot(x="Color",y="Nonflavanoids",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Nonflavanoids",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Nonflavanoids",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Nonflavanoids",data=wine,palette="hls")
sns.boxplot(x="Color",y="Proanthocyanins",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Proanthocyanins",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Proanthocyanins",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Proanthocyanins",data=wine,palette="hls")
sns.boxplot(x="Hue",y="Color",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Color",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Color",data=wine,palette="hls")
sns.boxplot(x="Dilution",y="Hue",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Hue",data=wine,palette="hls")
sns.boxplot(x="Proline",y="Dilution",data=wine,palette="hls")
sns.pairplot(wine.iloc[:,0:14]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(wine,hue="wine",size=2)
wine["Type"].value_counts()
wine["Alcohol"].value_counts()
wine["Malic"].value_counts()
wine["Ash"].value_counts()
wine["Alcalinity"].value_counts()
wine["Magnesium"].value_counts()
wine["Phenols"].value_counts()
wine["Flavanoids"].value_counts()
wine["Nonflavanoids"].value_counts()
wine["Proanthocyanins"].value_counts()
wine["Color"].value_counts()
wine["Hue"].value_counts()
wine["Dilution"].value_counts()
wine["Proline"].value_counts()
wine["Type"].value_counts().plot(kind = "pie")
wine["Alcohol"].value_counts().plot(kind = "pie")
wine["Malic"].value_counts().plot(kind = "pie")
wine["Ash"].value_counts().plot(kind = "pie")
wine["Alcalinity"].value_counts().plot(kind = "pie")
wine["Magnesium"].value_counts().plot(kind = "pie")
wine["Phenols"].value_counts().plot(kind = "pie")
wine["Flavanoids"].value_counts().plot(kind = "pie")
wine["Nonflavanoids"].value_counts().plot(kind = "pie")
wine["Proanthocyanins"].value_counts().plot(kind = "pie")
wine["Color"].value_counts().plot(kind = "pie")
wine["Hue"].value_counts().plot(kind = "pie")
wine["Dilution"].value_counts().plot(kind = "pie")
wine["Proline"].value_counts().plot(kind = "pie")
sns.pairplot(wine,hue="Alcohol",size=4,diag_kind = "kde")
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Alcohol","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Malic","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Ash","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Alcalinity","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Magnesium","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Phenols","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Flavanoids","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Nonflavanoids","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Proanthocyanins","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Color","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Hue","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Dilution","Type").add_legend()
sns.FacetGrid(wine,hue="Type").map(plt.scatter,"Proline","Type").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Malic","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Ash","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Alcalinity","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Magnesium","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Phenols","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Flavanoids","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Nonflavanoids","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Proanthocyanins","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Color","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Hue","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Dilution","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(plt.scatter,"Proline","Alcohol").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Ash","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Alcalinity","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Magnesium","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Phenols","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Flavanoids","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Nonflavanoids","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Proanthocyanins","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Color","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Hue","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Dilution","Malic").add_legend()
sns.FacetGrid(wine,hue="Malic").map(plt.scatter,"Proline","Malic").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Alcalinity","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Magnesium","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Phenols","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Flavanoids","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Nonflavanoids","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Proanthocyanins","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Color","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Hue","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Dilution","Ash").add_legend()
sns.FacetGrid(wine,hue="Ash").map(plt.scatter,"Proline","Ash").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Magnesium","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Phenols","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Flavanoids","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Nonflavanoids","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Proanthocyanins","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Color","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Hue","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Dilution","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(plt.scatter,"Proline","Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Phenols","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Flavanoids","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Nonflavanoids","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Proanthocyanins","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Color","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Hue","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Dilution","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(plt.scatter,"Proline","Magnesium").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Flavanoids","Phenols").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Nonflavanoids","Phenols").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Proanthocyanins","Phenols").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Color","Phenols").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Hue","Phenols").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Dilution","Phenols").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(plt.scatter,"Proline","Phenols").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(plt.scatter,"Nonflavanoids","Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(plt.scatter,"Proanthocyanins","Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(plt.scatter,"Color","Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(plt.scatter,"Hue","Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(plt.scatter,"Dilution","Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(plt.scatter,"Proline","Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(plt.scatter,"Proanthocyanins","Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(plt.scatter,"Color","Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(plt.scatter,"Hue","Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(plt.scatter,"Dilution","Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(plt.scatter,"Proline","Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(plt.scatter,"Color","Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(plt.scatter,"Hue","Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(plt.scatter,"Dilution","Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(plt.scatter,"Proline","Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Color").map(plt.scatter,"Hue","Color").add_legend()
sns.FacetGrid(wine,hue="Color").map(plt.scatter,"Dilution","Color").add_legend()
sns.FacetGrid(wine,hue="Color").map(plt.scatter,"Proline","Color").add_legend()
sns.FacetGrid(wine,hue="Hue").map(plt.scatter,"Dilution","Hue").add_legend()
sns.FacetGrid(wine,hue="Hue").map(plt.scatter,"Proline","Hue").add_legend()
sns.FacetGrid(wine,hue="Dilution").map(plt.scatter,"Dilution","Hue").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Alcohol").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Malic").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Ash").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Magnesium").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Phenols").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Type").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Malic").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Ash").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Magnesium").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Phenols").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Alcohol").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Ash").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Magnesium").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Phenols").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Malic").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Alcalinity").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Magnesium").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Phenols").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Ash").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Magnesium").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Phenols").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Alcalinity").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Phenols").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Magnesium").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Flavanoids").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Phenols").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(sns.kdeplot,"Nonflavanoids").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Flavanoids").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(sns.kdeplot,"Proanthocyanins").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Nonflavanoids").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(sns.kdeplot,"Color").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Proanthocyanins").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Color").map(sns.kdeplot,"Hue").add_legend()
sns.FacetGrid(wine,hue="Color").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Color").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Hue").map(sns.kdeplot,"Dilution").add_legend()
sns.FacetGrid(wine,hue="Hue").map(sns.kdeplot,"Proline").add_legend()
sns.FacetGrid(wine,hue="Dilution").map(sns.kdeplot,"Proline").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(wine['Type'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Type']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Type']),dist="norm",plot=pylab)
stats.probplot((wine['Type'] * wine['Type']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Type']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Type'])*np.exp(wine['Type']),dist="norm",plot=pylab)
reci_1=1/wine['Type']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((wine['Type'] * wine['Type'])+wine['Type']),dist="norm",plot=pylab)
stats.probplot(wine['Alcohol'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Alcohol']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Alcohol']),dist="norm",plot=pylab)
stats.probplot((wine['Alcohol'] * wine['Alcohol']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Alcohol']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Alcohol'])*np.exp(wine['Alcohol']),dist="norm",plot=pylab)
reci_2=1/wine['Alcohol']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((wine['Alcohol'] * wine['Alcohol'])+wine['Alcohol']),dist="norm",plot=pylab)
stats.probplot(wine['Malic'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Malic']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Malic']),dist="norm",plot=pylab)
stats.probplot((wine['Malic'] * wine['Malic']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Malic']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Malic'])*np.exp(wine['Malic']),dist="norm",plot=pylab)
reci_3=1/wine['Malic']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((wine['Malic'] * wine['Malic'])+wine['Malic']),dist="norm",plot=pylab)
stats.probplot(wine['Ash'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Ash']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Ash']),dist="norm",plot=pylab)
stats.probplot((wine['Ash'] * wine['Ash']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Ash']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Ash'])*np.exp(wine['Ash']),dist="norm",plot=pylab)
reci_4=1/wine['Ash']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((wine['Ash'] * wine['Ash'])+wine['Ash']),dist="norm",plot=pylab)
stats.probplot(wine['Alcalinity'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Alcalinity']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Alcalinity']),dist="norm",plot=pylab)
stats.probplot((wine['Alcalinity'] * wine['Alcalinity']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Alcalinity']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Alcalinity'])*np.exp(wine['Alcalinity']),dist="norm",plot=pylab)
reci_5=1/wine['Alcalinity']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((wine['Alcalinity'] * wine['Alcalinity'])+wine['Alcalinity']),dist="norm",plot=pylab)
stats.probplot(wine['Magnesium'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Magnesium']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Magnesium']),dist="norm",plot=pylab)
stats.probplot((wine['Magnesium'] * wine['Magnesium']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Magnesium']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Magnesium'])*np.exp(wine['Magnesium']),dist="norm",plot=pylab)
reci_6=1/wine['Magnesium']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((wine['Magnesium'] * wine['Magnesium'])+wine['Magnesium']),dist="norm",plot=pylab)
stats.probplot(wine['Phenols'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Phenols']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Phenols']),dist="norm",plot=pylab)
stats.probplot((wine['Phenols'] * wine['Phenols']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Phenols']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Phenols'])*np.exp(wine['Phenols']),dist="norm",plot=pylab)
reci_7=1/wine['Phenols']
reci_7_2=reci_7 * reci_7
reci_7_4=reci_7_2 * reci_7_2
stats.probplot(reci_7*reci_7,dist="norm",plot=pylab)
stats.probplot(reci_7_2,dist="norm",plot=pylab)
stats.probplot(reci_7_4,dist="norm",plot=pylab)
stats.probplot(reci_7_4*reci_7_4,dist="norm",plot=pylab)
stats.probplot((reci_7_4*reci_7_4)*(reci_7_4*reci_7_4),dist="norm",plot=pylab)
stats.probplot(((wine['Phenols'] * wine['Phenols'])+wine['Phenols']),dist="norm",plot=pylab)
stats.probplot(wine['Flavanoids'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Flavanoids']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Flavanoids']),dist="norm",plot=pylab)
stats.probplot((wine['Flavanoids'] * wine['Flavanoids']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Flavanoids']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Flavanoids'])*np.exp(wine['Flavanoids']),dist="norm",plot=pylab)
reci_8=1/wine['Flavanoids']
reci_8_2=reci_8 * reci_8
reci_8_4=reci_8_2 * reci_8_2
stats.probplot(reci_8*reci_8,dist="norm",plot=pylab)
stats.probplot(reci_8_2,dist="norm",plot=pylab)
stats.probplot(reci_8_4,dist="norm",plot=pylab)
stats.probplot(reci_8_4*reci_8_4,dist="norm",plot=pylab)
stats.probplot((reci_8_4*reci_8_4)*(reci_8_4*reci_8_4),dist="norm",plot=pylab)
stats.probplot(((wine['Flavanoids'] * wine['Flavanoids'])+wine['Flavanoids']),dist="norm",plot=pylab)
stats.probplot(wine['Nonflavanoids'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Nonflavanoids']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Nonflavanoids']),dist="norm",plot=pylab)
stats.probplot((wine['Nonflavanoids'] * wine['Nonflavanoids']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Nonflavanoids']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Nonflavanoids'])*np.exp(wine['Nonflavanoids']),dist="norm",plot=pylab)
reci_9=1/wine['Nonflavanoids']
reci_9_2=reci_9 * reci_9
reci_9_4=reci_9_2 * reci_9_2
stats.probplot(reci_9*reci_9,dist="norm",plot=pylab)
stats.probplot(reci_9_2,dist="norm",plot=pylab)
stats.probplot(reci_9_4,dist="norm",plot=pylab)
stats.probplot(reci_9_4*reci_9_4,dist="norm",plot=pylab)
stats.probplot((reci_9_4*reci_9_4)*(reci_9_4*reci_9_4),dist="norm",plot=pylab)
stats.probplot(((wine['Nonflavanoids'] * wine['Nonflavanoids'])+wine['Nonflavanoids']),dist="norm",plot=pylab)
stats.probplot(wine['Proanthocyanins'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Proanthocyanins']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Proanthocyanins']),dist="norm",plot=pylab)
stats.probplot((wine['Proanthocyanins'] * wine['Proanthocyanins']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Proanthocyanins']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Proanthocyanins'])*np.exp(wine['Proanthocyanins']),dist="norm",plot=pylab)
reci_10=1/wine['Proanthocyanins']
reci_10_2=reci_10 * reci_10
reci_10_4=reci_10_2 * reci_10_2
stats.probplot(reci_10*reci_10,dist="norm",plot=pylab)
stats.probplot(reci_10_2,dist="norm",plot=pylab)
stats.probplot(reci_10_4,dist="norm",plot=pylab)
stats.probplot(reci_10_4*reci_10_4,dist="norm",plot=pylab)
stats.probplot((reci_10_4*reci_10_4)*(reci_10_4*reci_10_4),dist="norm",plot=pylab)
stats.probplot(((wine['Proanthocyanins'] * wine['Proanthocyanins'])+wine['Proanthocyanins']),dist="norm",plot=pylab)
stats.probplot(wine['Color'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Color']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Color']),dist="norm",plot=pylab)
stats.probplot((wine['Color'] * wine['Color']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Color']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Color'])*np.exp(wine['Color']),dist="norm",plot=pylab)
reci_11=1/wine['Color']
reci_11_2=reci_11 * reci_11
reci_11_4=reci_11_2 * reci_11_2
stats.probplot(reci_11*reci_11,dist="norm",plot=pylab)
stats.probplot(reci_11_2,dist="norm",plot=pylab)
stats.probplot(reci_11_4,dist="norm",plot=pylab)
stats.probplot(reci_11_4*reci_11_4,dist="norm",plot=pylab)
stats.probplot((reci_11_4*reci_11_4)*(reci_11_4*reci_11_4),dist="norm",plot=pylab)
stats.probplot(((wine['Color'] * wine['Color'])+wine['Color']),dist="norm",plot=pylab)
stats.probplot(wine['Hue'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Hue']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Hue']),dist="norm",plot=pylab)
stats.probplot((wine['Hue'] * wine['Hue']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Hue']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Hue'])*np.exp(wine['Hue']),dist="norm",plot=pylab)
reci_12=1/wine['Hue']
reci_12_2=reci_12 * reci_12
reci_12_4=reci_12_2 * reci_12_2
stats.probplot(reci_12*reci_12,dist="norm",plot=pylab)
stats.probplot(reci_12_2,dist="norm",plot=pylab)
stats.probplot(reci_12_4,dist="norm",plot=pylab)
stats.probplot(reci_12_4*reci_12_4,dist="norm",plot=pylab)
stats.probplot((reci_12_4*reci_12_4)*(reci_12_4*reci_12_4),dist="norm",plot=pylab)
stats.probplot(((wine['Hue'] * wine['Hue'])+wine['Hue']),dist="norm",plot=pylab)
stats.probplot(wine['Dilution'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Dilution']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Dilution']),dist="norm",plot=pylab)
stats.probplot((wine['Dilution'] * wine['Dilution']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Dilution']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Dilution'])*np.exp(wine['Dilution']),dist="norm",plot=pylab)
reci_13=1/wine['Dilution']
reci_13_2=reci_13 * reci_13
reci_13_4=reci_13_2 * reci_13_2
stats.probplot(reci_13*reci_13,dist="norm",plot=pylab)
stats.probplot(reci_13_2,dist="norm",plot=pylab)
stats.probplot(reci_13_4,dist="norm",plot=pylab)
stats.probplot(reci_13_4*reci_13_4,dist="norm",plot=pylab)
stats.probplot((reci_13_4*reci_13_4)*(reci_13_4*reci_13_4),dist="norm",plot=pylab)
stats.probplot(((wine['Dilution'] * wine['Dilution'])+wine['Dilution']),dist="norm",plot=pylab)
stats.probplot(wine['Proline'],dist="norm",plot=pylab)
stats.probplot(np.log(wine['Proline']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(wine['Proline']),dist="norm",plot=pylab)
stats.probplot((wine['Proline'] * wine['Proline']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Proline']),dist="norm",plot=pylab)
stats.probplot(np.exp(wine['Proline'])*np.exp(wine['Proline']),dist="norm",plot=pylab)
reci_14=1/wine['Proline']
reci_14_2=reci_14 * reci_14
reci_14_4=reci_14_2 * reci_14_2
stats.probplot(reci_14*reci_14,dist="norm",plot=pylab)
stats.probplot(reci_14_2,dist="norm",plot=pylab)
stats.probplot(reci_14_4,dist="norm",plot=pylab)
stats.probplot(reci_14_4*reci_14_4,dist="norm",plot=pylab)
stats.probplot((reci_14_4*reci_14_4)*(reci_14_4*reci_14_4),dist="norm",plot=pylab)
stats.probplot(((wine['Proline'] * wine['Proline'])+wine['Proline']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Type
stats.norm.ppf(0.975,1.938202,0.775035)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Type"],1.938202,0.775035) # similar to pnorm in R 
#### Alcohol
stats.norm.ppf(0.975,13.000618,0.811827)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Alcohol"],13.000618,0.811827) # similar to pnorm in R 
#### Malic
stats.norm.ppf(0.975,2.336348,1.117146)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Malic"],2.336348,1.117146) # similar to pnorm in R 
#### Ash
stats.norm.ppf(0.975,2.366517,0.274344)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Ash"],2.366517,0.274344) # similar to pnorm in R 
#### Alcalinity
stats.norm.ppf(0.975,19.494944,3.339564)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Alcalinity"],19.494944,3.339564) # similar to pnorm in R 
#### Magnesium
stats.norm.ppf(0.975,99.741573,14.282484)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Magnesium"],99.741573,14.282484) # similar to pnorm in R 
#### Phenols
stats.norm.ppf(0.975,2.295112,0.625851)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Phenols"],2.295112,0.625851) # similar to pnorm in R
#### Flavanoids
stats.norm.ppf(0.975,2.02927,0.998859)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Flavanoids"],2.02927,0.998859) # similar to pnorm in R 
#### Nonflavanoids
stats.norm.ppf(0.975,0.361854,0.124453)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Nonflavanoids"],0.361854,0.124453) # similar to pnorm in R 
#### Proanthocyanins
stats.norm.ppf(0.975,1.590899,0.572359)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Proanthocyanins"],1.590899,0.572359) # similar to pnorm in R 
#### Color
stats.norm.ppf(0.975,5.058090,2.318286)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Color"],5.058090,2.318286) # similar to pnorm in R 
#### Hue
stats.norm.ppf(0.975,0.957449,0.228572)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Hue"],0.957449,0.228572) # similar to pnorm in R 
#### Dilution
stats.norm.ppf(0.975,2.611685,0.709990)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Dilution"],2.611685,0.709990) # similar to pnorm in R 
#### Proline
stats.norm.ppf(0.975,746.893258,314.907474)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(wine["Proline"],746.893258,314.907474) # similar to pnorm in R 
##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
wine.corr(method = "pearson")
wine.corr(method = "kendall")
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(wine['Type'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(wine['Alcohol'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(wine['Malic'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(wine['Ash'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(wine['Alcalinity'])
normalized_E = preprocessing.normalize([e_array])
f_array = np.array(wine['Magnesium'])
normalized_F = preprocessing.normalize([f_array])
g_array = np.array(wine['Phenols'])
normalized_G = preprocessing.normalize([g_array])
h_array = np.array(wine['Flavanoids'])
normalized_H = preprocessing.normalize([h_array])
i_array = np.array(wine['Nonflavanoids'])
normalized_I = preprocessing.normalize([i_array])
j_array = np.array(wine['Proanthocyanins'])
normalized_J = preprocessing.normalize([j_array])
k_array = np.array(wine['Color'])
normalized_K = preprocessing.normalize([k_array])
l_array = np.array(wine['Hue'])
normalized_L = preprocessing.normalize([l_array])
m_array = np.array(wine['Dilution'])
normalized_M = preprocessing.normalize([m_array])
n_array = np.array(wine['Proline'])
normalized_N = preprocessing.normalize([n_array])
# to get top 6 rows
wine.head(40) # to get top n rows use cars.head(10)
wine.tail(10)
# Correlation matrix 
wine.corr()
# Scatter plot between the variables along with histograms
sns.pairplot(wine)
pd.tools.plotting.scatter_matrix(wine) ##-> also used for plotting all in one graph
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
wine = wine[wine.columns[wine.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
wine = wine.loc[wine.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
wine = wine.fillna(wine.median())
##wine['gender'].fillna(wine['gender'].value_counts().idxmax(), inplace=True)
##wine['children'].fillna(wine['children'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##wine['column_name'].fillna(wine['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim1 = wine['Type'].mean () + wine['Type'].std () * factor   ### 4.26
lower_lim1 = wine['Type'].mean () - wine['Type'].std () * factor   ### -0.38
wine1 = wine[(wine['Type'] < upper_lim1) & (wine['Type'] > lower_lim1)]
upper_lim2 = wine['Alcohol'].mean () + wine['Alcohol'].std () * factor  #### 15.44
lower_lim2 = wine['Alcohol'].mean () - wine['Alcohol'].std () * factor  ## 10.56
wine2 = wine[(wine['Alcohol'] < upper_lim2) & (wine['Alcohol'] > lower_lim2)]
upper_lim3 = wine['Malic'].mean () + wine['Malic'].std () * factor  ### 5.68
lower_lim3 = wine['Malic'].mean () - wine['Malic'].std () * factor  ### -1.01
wine3 = wine[(wine['Malic'] < upper_lim3) & (wine['Malic'] > lower_lim3)]
upper_lim4 = wine['Ash'].mean () + wine['Ash'].std () * factor ###6.62
lower_lim4 = wine['Ash'].mean () - wine['Ash'].std () * factor  #####  -0.39
wine4 = wine[(wine['Ash'] < upper_lim4) & (wine['Ash'] > lower_lim4)]
upper_lim5 = wine['Alcalinity'].mean () + wine['Alcalinity'].std () * factor   #### 29.51
lower_lim5 = wine['Alcalinity'].mean () - wine['Alcalinity'].std () * factor  ### 9.47
wine5 = wine[(wine['Alcalinity'] < upper_lim5) & (wine['Alcalinity'] > lower_lim5)]
upper_lim6 = wine['Magnesium'].mean () + wine['Magnesium'].std () * factor  #### 142.59
lower_lim6 = wine['Magnesium'].mean () - wine['Magnesium'].std () * factor ### 56.89
wine6 = wine[(wine['Magnesium'] < upper_lim6) & (wine['Magnesium'] > lower_lim6)]
upper_lim7 = wine['Phenols'].mean () + wine['Phenols'].std () * factor ### 4.17
lower_lim7 = wine['Phenols'].mean () - wine['Phenols'].std () * factor  ### 0.42
wine = wine[(wine['Phenols'] < upper_lim7) & (wine['Phenols'] > lower_lim7)]
upper_lim8 = wine['Flavanoids'].mean () + wine['Flavanoids'].std () * factor   ### 5.03
lower_lim8 = wine['Flavanoids'].mean () - wine['Flavanoids'].std () * factor   ### -0.97
wine8 = wine[(wine['Flavanoids'] < upper_lim8) & (wine['Flavanoids'] > lower_lim8)]
upper_lim9 = wine['Nonflavanoids'].mean () + wine['Nonflavanoids'].std () * factor   ### 0.73
lower_lim9 = wine['Nonflavanoids'].mean () - wine['Nonflavanoids'].std () * factor   ### -0.01
wine9 = wine[(wine['Nonflavanoids'] < upper_lim9) & (wine['Nonflavanoids'] > lower_lim9)]
upper_lim10 = wine['Proanthocyanins'].mean () + wine['Proanthocyanins'].std () * factor  ### 3.31
lower_lim10 = wine['Proanthocyanins'].mean () - wine['Proanthocyanins'].std () * factor  ### -0.13
wine10 = wine[(wine['Proanthocyanins'] < upper_lim10) & (wine['Proanthocyanins'] > lower_lim10)]
upper_lim11 = wine['Color'].mean () + wine['Color'].std () * factor ### 12.01
lower_lim11 = wine['Color'].mean () - wine['Color'].std () * factor  #####  -1.89
wine11 = wine[(wine['Color'] < upper_lim11) & (wine['Color'] > lower_lim11)]
upper_lim12 = wine['Hue'].mean () + wine['Hue'].std () * factor   #### 1.64
lower_lim12 = wine['Hue'].mean () - wine['Hue'].std () * factor  ### 0.27
wine12 = wine[(wine['Hue'] < upper_lim12) & (wine['Hue'] > lower_lim12)]
upper_lim13 = wine['Dilution'].mean () + wine['Dilution'].std () * factor  #### 142.59
lower_lim13 = wine['Dilution'].mean () - wine['Dilution'].std () * factor ### 513.89
wine13 = wine[(wine['Dilution'] < upper_lim13) & (wine['Dilution'] > lower_lim13)]
upper_lim14 = wine['Proline'].mean () + wine['Proline'].std () * factor 
lower_lim14 = wine['Proline'].mean () - wine['Proline'].std () * factor  
wine = wine[(wine['Proline'] < upper_lim14) & (wine['Proline'] > lower_lim14)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim15 = wine['Type'].quantile(.95)
lower_lim15 = wine['Type'].quantile(.05)
wine15 = wine[(wine['Type'] < upper_lim15) & (wine['Type'] > lower_lim15)]
upper_lim16 = wine['Alcohol'].quantile(.95)
lower_lim16 = wine['Alcohol'].quantile(.05)
wine16 = wine[(wine['Alcohol'] < upper_lim16) & (wine['Alcohol'] > lower_lim16)]
upper_lim17 = wine['Malic'].quantile(.95)
lower_lim17 = wine['Malic'].quantile(.05)
wine17 = wine[(wine['Malic'] < upper_lim17) & (wine['Malic'] > lower_lim17)]
upper_lim18 = wine['Ash'].quantile(.95)
lower_lim18 = wine['Ash'].quantile(.05)
wine18 = wine[(wine['Ash'] < upper_lim18) & (wine['Ash'] > lower_lim18)]
upper_lim19 = wine['Alcalinity'].quantile(.95)
lower_lim19 = wine['Alcalinity'].quantile(.05)
wine19 = wine[(wine['Alcalinity'] < upper_lim19) & (wine['Alcalinity'] > lower_lim19)]
upper_lim20 = wine['Magnesium'].quantile(.95)
lower_lim20 = wine['Magnesium'].quantile(.05)
wine20 = wine[(wine['Magnesium'] < upper_lim20) & (wine['Magnesium'] > lower_lim20)]
upper_lim21 = wine['Phenols'].quantile(.95)
lower_lim21 = wine['Phenols'].quantile(.05)
wine21 = wine[(wine['Phenols'] < upper_lim21) & (wine['Phenols'] > lower_lim21)]
upper_lim22 = wine['Flavanoids'].quantile(.95)
lower_lim22 = wine['Flavanoids'].quantile(.05)
wine22 = wine[(wine['Flavanoids'] < upper_lim22) & (wine['Flavanoids'] > lower_lim22)]
upper_lim23 = wine['Nonflavanoids'].quantile(.95)
lower_lim23 = wine['Nonflavanoids'].quantile(.05)
wine23 = wine[(wine['Nonflavanoids'] < upper_lim23) & (wine['Nonflavanoids'] > lower_lim23)]
upper_lim24 = wine['Proanthocyanins'].quantile(.95)
lower_lim24 = wine['Proanthocyanins'].quantile(.05)
wine24 = wine[(wine['Proanthocyanins'] < upper_lim24) & (wine['Proanthocyanins'] > lower_lim24)]
upper_lim25 = wine['Color'].quantile(.95)
lower_lim25 = wine['Color'].quantile(.05)
wine25 = wine[(wine['Color'] < upper_lim25) & (wine['Color'] > lower_lim25)]
upper_lim26 = wine['Hue'].quantile(.95)
lower_lim26 = wine['Hue'].quantile(.05)
wine26 = wine[(wine['Hue'] < upper_lim26) & (wine['Hue'] > lower_lim26)]
upper_lim27 = wine['Dilution'].quantile(.95)
lower_lim27 = wine['Dilution'].quantile(.05)
wine27 = wine[(wine['Dilution'] < upper_lim27) & (wine['Dilution'] > lower_lim27)]
upper_lim28 = wine['Proline'].quantile(.95)
lower_lim28 = wine['Proline'].quantile(.05)
wine28 = wine[(wine['Proline'] < upper_lim28) & (wine['Proline'] > lower_lim28)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
wine.loc[(wine['Type'] > upper_lim15)] = upper_lim15
wine.loc[(wine['Type'] < lower_lim15)] = lower_lim15
wine.loc[(wine['Alcohol'] > upper_lim16)] = upper_lim16
wine.loc[(wine['Alcohol'] < lower_lim16)] = lower_lim16
wine.loc[(wine['Malic'] > upper_lim17)] = upper_lim17
wine.loc[(wine['Malic'] < lower_lim17)] = lower_lim17
wine.loc[(wine['Ash'] > upper_lim18)] = upper_lim18
wine.loc[(wine['Ash'] < lower_lim18)] = lower_lim18
wine.loc[(wine['Alcalinity'] > upper_lim19)] = upper_lim19
wine.loc[(wine['Alcalinity'] < lower_lim19)] = lower_lim19
wine.loc[(wine['Magnesium'] > upper_lim20)] = upper_lim20
wine.loc[(wine['Magnesium'] < lower_lim20)] = lower_lim20
wine.loc[(wine['Phenols'] > upper_lim21)] = upper_lim21
wine.loc[(wine['Phenols'] < lower_lim21)] = lower_lim21
wine.loc[(wine['Flavanoids'] > upper_lim22)] = upper_lim22
wine.loc[(wine['Flavanoids'] < lower_lim22)] = lower_lim22
wine.loc[(wine['Nonflavanoids'] > upper_lim23)] = upper_lim23
wine.loc[(wine['Nonflavanoids'] < lower_lim23)] = lower_lim23
wine.loc[(wine['Proanthocyanins'] > upper_lim24)] = upper_lim24
wine.loc[(wine['Proanthocyanins'] < lower_lim24)] = lower_lim24
wine.loc[(wine['Color'] > upper_lim25)] = upper_lim25
wine.loc[(wine['Color'] < lower_lim25)] = lower_lim25
wine.loc[(wine['Hue'] > upper_lim26)] = upper_lim26
wine.loc[(wine['Hue'] < lower_lim26)] = lower_lim26
wine.loc[(wine['Dilution'] > upper_lim27)] = upper_lim27
wine.loc[(wine['Dilution'] < lower_lim27)] = lower_lim27
wine.loc[(wine['Proline'] > upper_lim28)] = upper_lim28
wine.loc[(wine['Proline'] < lower_lim28)] = lower_lim28
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
wine['bin1'] = pd.cut(wine['Type'], bins=[1,2,3], labels=["Low","High"])
wine['bin2'] = pd.cut(wine['Alcohol'], bins=[11.03,12.50,14.83], labels=["Low","High"])
wine['bin3'] = pd.cut(wine['Malic'], bins=[0.74,3,5.8], labels=["Low","High"])
wine['bin4'] = pd.cut(wine['Ash'],bins=[1.36,2.4,3.23],labels=["Low", "High"])
wine['bin5'] = pd.cut(wine['Alcalinity'],bins=[10.6,20,30],labels=["Low", "High"])
wine['bin6'] = pd.cut(wine['Magnesium'],bins=[70,110,162],labels=["Low","High"])
wine['bin7'] = pd.cut(wine['Phenols'],bins=[0.98,2.5,3.88],labels=["Low","High"])
wine['bin8'] = pd.cut(wine['Flavanoids'], bins=[0.34,2.5,5.08], labels=["Low","High"])
wine['bin9'] = pd.cut(wine['Nonflavanoids'], bins=[0.13,0.4,0.66], labels=["Low","High"])
wine['bin10'] = pd.cut(wine['Proanthocyanins'], bins=[0.41,2,3.58], labels=["Low","High"])
wine['bin11'] = pd.cut(wine['Color'],bins=[1.28,7,13],labels=["Low", "High"])
wine['bin12'] = pd.cut(wine['Hue'],bins=[0.48,1,1.71],labels=["Low", "High"])
wine['bin13'] = pd.cut(wine['Dilution'],bins=[1.27,2.7,4],labels=["Low","High"])
wine['bin14'] = pd.cut(wine['Proline'],bins=[278,800,1680],labels=["Low","High"])
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
wine = pd.DataFrame({'Type':wine.iloc[:,0]})
wine['log+1'] = (wine['Type']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Type']-wine['Type'].min()+1).transform(np.log)
wine = pd.DataFrame({'Alcohol':wine.iloc[:,1]})
wine['log+1'] = (wine['Alcohol']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Alcohol']-wine['Alcohol'].min()+1).transform(np.log)
wine = pd.DataFrame({'Malic':wine.iloc[:,2]})
wine['log+1'] = (wine['Malic']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Malic']-wine['Malic'].min()+1).transform(np.log)
wine = pd.DataFrame({'Ash':wine.iloc[:,3]})
wine['log+1'] = (wine['Ash']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Ash']-wine['Ash'].min()+1).transform(np.log)
wine = pd.DataFrame({'Alcalinity':wine.iloc[:,4]})
wine['log+1'] = (wine['Alcalinity']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Alcalinity']-wine['Alcalinity'].min()+1).transform(np.log)
wine = pd.DataFrame({'Magnesium':wine.iloc[:,5]})
wine['log+1'] = (wine['Magnesium']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Magnesium']-wine['Magnesium'].min()+1).transform(np.log)
wine = pd.DataFrame({'Phenols':wine.iloc[:,6]})
wine['log+1'] = (wine['Phenols']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Phenols']-wine['Phenols'].min()+1).transform(np.log)
wine = pd.DataFrame({'Flavanoids':wine.iloc[:,7]})
wine['log+1'] = (wine['Flavanoids']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Flavanoids']-wine['Flavanoids'].min()+1).transform(np.log)
wine = pd.DataFrame({'Nonflavanoids':wine.iloc[:,8]})
wine['log+1'] = (wine['Nonflavanoids']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Nonflavanoids']-wine['Nonflavanoids'].min()+1).transform(np.log)
wine = pd.DataFrame({'Proanthocyanins':wine.iloc[:,9]})
wine['log+1'] = (wine['Proanthocyanins']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Proanthocyanins']-wine['Proanthocyanins'].min()+1).transform(np.log)
wine = pd.DataFrame({'Color':wine.iloc[:,10]})
wine['log+1'] = (wine['Color']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Color']-wine['Color'].min()+1).transform(np.log)
wine = pd.DataFrame({'Hue':wine.iloc[:,11]})
wine['log+1'] = (wine['Hue']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Hue']-wine['Hue'].min()+1).transform(np.log)
wine = pd.DataFrame({'Dilution':wine.iloc[:,12]})
wine['log+1'] = (wine['Dilution']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Dilution']-wine['Dilution'].min()+1).transform(np.log)
wine = pd.DataFrame({'Proline':wine.iloc[:,13]})
wine['log+1'] = (wine['Proline']+1).transform(np.log)
#Negative Values Handling
wine['log'] = (wine['Proline']-wine['Proline'].min()+1).transform(np.log)                                  
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = wine.groupby('Type')
sums = grouped['Type'].sum().add_suffix('_sum')
avgs = grouped['Type'].mean().add_suffix('_avg')
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(wine.iloc[:,0:14])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
df_std = stan(wine.iloc[:,0:14])
##### Feature Extraction
X = wine.drop("Alcohol", axis=1)
Y=wine['Alcohol']
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
##X = pd.get_dummies(X, prefix_sep='_')
Y = LabelEncoder().fit_transform(Y)
X = StandardScaler().fit_transform(X)
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 178)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=178).fit(X_Train,Y_Train)
    print(time.process_time() - start) ### 1.234375
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest))
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, wine['Alcohol']], axis = 1)
PCA_df['Alcohol'] = LabelEncoder().fit_transform(PCA_df['Alcohol'])
PCA_df.head()    
Alcohol = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for Alcohol, color in zip(Alcohol, colors):
    plt.scatter(PCA_df.loc[PCA_df['Alcohol'] == Alcohol, 'PC1'], 
                PCA_df.loc[PCA_df['Alcohol'] == Alcohol, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 20)
plt.legend(['Less', 'More'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
####Independent Component Analysis
from sklearn.decomposition import FastICA
ica = FastICA(n_components=3)
X_ica = ica.fit_transform(X)
forest_test(X_ica, Y)
####Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1)
# run an LDA and use it to transform the features
labels=np.unique(Y)
X_lda = lda.fit(X,Y).transform(X)
print('Original number of features:', X.shape[1])  ## 13
print('Reduced number of features:', X_lda.shape[1]) ##1
forest_test(X_lda, Y)
#####LDA can also be used as a classifier. Therefore, we can now test how an LDA Classifier can perform in this situation.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.30,random_state = 178) 
start = time.process_time()
lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)
print(time.process_time() - start)  ## 0.84375
predictionlda = lda.predict(X_Test_Reduced)
print(confusion_matrix(Y_Test_Reduced,predictionlda))
print(classification_report(Y_Test_Reduced,predictionlda))
#####Locally Linear Embedding is a dimensionality reduction technique based on Manifold 
##Learning. A Manifold is an object of D dimensions which is embedded in an higher-dimensional space.
## Manifold Learning aims then to make this object representable in its original D dimensions instead of being represented in an unnecessary greater space.
from sklearn.manifold import LocallyLinearEmbedding
embedding = LocallyLinearEmbedding(n_components=3)
X_lle = embedding.fit_transform(X) 
forest_test(X_lle, Y)
####t-SNE is non-linear dimensionality reduction technique which is typically used to visualize high dimensional datasets.
#####t-SNE works by minimizing the divergence between a distribution constituted by the pairwise probability similarities
### of the input features in the original high dimensional space and its equivalent in the reduced low dimensional space. 
##t-SNE makes then use of the Kullback-Leiber (KL) divergence in order to measure the dissimilarity of the two different 
####distributions. The KL divergence is then minimized using gradient descent.
from sklearn.manifold import TSNE
start = time.process_time() ###792.96875
tsne = TSNE(n_components=3, verbose=1, perplexity=400, n_iter=3000)
X_tsne = tsne.fit_transform(X) ### KL divergence after 3000 iterations: 0.991811
print(time.process_time() - start)  ### 28.875
forest_test(X_tsne, Y)
#####Autoencoders are a family of Machine Learning algorithms which can be used as a 
###dimensionality reduction technique. The main difference between Autoencoders and 
##other dimensionality reduction techniques is that Autoencoders use non-linear 
###transformations to project data from a high dimension to a lower one.
from keras.layers import Input, Dense
from keras.models import Model
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(3, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='softmax')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=178)
autoencoder.fit(X1, Y1,epochs=178,batch_size=178,shuffle=True,verbose = 178,validation_data=(X2, Y2))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)
###########################################################################
wine.describe()
wine.head()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
# Considering only numerical data 
wine.data = wine.iloc[:,0:]
wine.data.head(4)
# Normalizing the numerical data 
wine_normal = scale(wine.data)
pca = PCA(n_components = 6)
pca_values = pca.fit_transform(wine_normal)
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:6]
plt.scatter(x,y,color=["red","blue"])
from mpl_toolkits.mplot3d.Axes3D import scatter
scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])
################### Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:6])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_
##########################################################
new_df = pd.DataFrame(pca_values[:,0:6])
from sklearn.cluster import AgglomerativeClustering
agg1 = AgglomerativeClustering(n_clusters = 3,affinity='euclidean',linkage='ward')
agg1.fit(new_df)
agg1.labels_
agg2 = AgglomerativeClustering(n_clusters = 4,affinity='euclidean',linkage='average')
agg2.fit(new_df)
agg2.labels_
agg3 = AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='single')
agg3.fit(new_df)
agg3.labels_
agg4 = AgglomerativeClustering(n_clusters = 5,affinity='euclidean',linkage='complete')
agg4.fit(new_df)
agg4.labels_
agg5 = AgglomerativeClustering(n_clusters = 6,affinity='manhattan',linkage='average')
agg5.fit(new_df)
agg5.labels_
agg6 = AgglomerativeClustering(n_clusters = 3,affinity='manhattan',linkage='single')
agg6.fit(new_df)
agg6.labels_
agg7 = AgglomerativeClustering(n_clusters = 5,affinity='manhattan',linkage='complete')
agg7.fit(new_df)
agg7.labels_
agg8 = AgglomerativeClustering(n_clusters = 4,affinity='canberra',linkage='average')
agg8.fit(new_df)
agg8.labels_
agg9 = AgglomerativeClustering(n_clusters = 3,affinity='canberra',linkage='single')
agg9.fit(new_df)
agg9.labels_
agg10 = AgglomerativeClustering(n_clusters = 5,affinity='canberra',linkage='complete')
agg10.fit(new_df)
agg10.labels_
agg11 = AgglomerativeClustering(n_clusters = 4,affinity='minkowski',linkage='average')
agg11.fit(new_df)
agg11.labels_
agg12 = AgglomerativeClustering(n_clusters = 3,affinity='minkowski',linkage='single')
agg12.fit(new_df)
agg12.labels_
agg13 = AgglomerativeClustering(n_clusters = 5,affinity='minkowski',linkage='complete')
agg13.fit(new_df)
agg13.labels_
agg14 = AgglomerativeClustering(n_clusters = 4,affinity='cosine',linkage='average')
agg14.fit(new_df)
agg14.labels_
agg15 = AgglomerativeClustering(n_clusters = 1,affinity='cosine',linkage='single')
agg15.fit(new_df)
agg15.labels_
agg16 = AgglomerativeClustering(n_clusters = 3,affinity='cosine',linkage='complete')
agg16.fit(new_df)
agg16.labels_
agg17 = AgglomerativeClustering(n_clusters = 3,affinity='l1',linkage='average')
agg17.fit(new_df)
agg17.labels_
agg18 = AgglomerativeClustering(n_clusters = 3,affinity='l1',linkage='single')
agg18.fit(new_df)
agg18.labels_
agg19 = AgglomerativeClustering(n_clusters = 3,affinity='l1',linkage='complete')
agg19.fit(new_df)
agg19.labels_
agg20 = AgglomerativeClustering(n_clusters = 3,affinity='l2',linkage='average')
agg20.fit(new_df)
agg20.labels_
agg21 = AgglomerativeClustering(n_clusters = 3,affinity='l2',linkage='single')
agg21.fit(new_df)
agg21.labels_
agg22 = AgglomerativeClustering(n_clusters = 3,affinity='l2',linkage='complete')
agg22.fit(new_df)
agg22.labels_
agg23 = AgglomerativeClustering(n_clusters = 3,affinity='precomputed',linkage='average')
agg23.fit(new_df)
agg23.labels_
agg24 = AgglomerativeClustering(n_clusters = 3,affinity='precomputed',linkage='single')
agg24.fit(new_df)
agg24.labels_
agg25 = AgglomerativeClustering(n_clusters = 3,affinity='precomputed',linkage='complete')
agg25.fit(new_df)
agg25.labels_