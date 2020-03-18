import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# reading a csv file using pandas library
forestfire = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\forestfires.csv")
forestfire.drop(["month"],axis=1,inplace=True)
forestfire.drop(["day"],axis=1,inplace=True)
#####Exploratory Data Analysis#########################################################
forestfire.mean() #### FFMC- 90.644681,DMC- 110.872340,DC- 547.940039,ISI-9.021663,
#### temp-18.889168,RH- 44.288201,wind- 4.017602,rain- 0.021663,area-12.847292,dayfri-0.164410,
#### daymon-0.143133,daysat-0.162476,daysun-0.183752,daythu-0.117988,daytue-0.123791,daywed-0.104449,
#### monthapr-0.017408,monthaug-0.355899,monthdec-0.017408,monthfeb-0.038685,monthjan-0.003868,
#### monthjul-0.061896,monthjun-0.032882,monthmar-0.104449,monthmay-0.003868,monthnov-0.001934,
#### monthoct-0.029014,monthsep-0.332689
forestfire.median()
forestfire.std() #### FFMC- 5.520111,DMC- 64.046482,DC- 248.066192,ISI-4.559477,
#### temp-5.806625,RH- 16.317469,wind- 1.791653,rain- 0.295959,area-63.655818,dayfri-0.371006,
#### daymon- 0.350548,daysat-0.369244,daysun-0.387657,daythu-0.322907,daytue-0.329662,daywed-0.306138,
#### monthapr-0.130913,monthaug-0.479249,monthdec-0.130913,monthfeb-0.193029,monthjan- 0.062137,
#### monthjul-0.241199,monthjun-0.178500,monthmar-0.306138,monthmay-0.062137,monthnov-0.043980,
#### monthoct-0.168007,monthsep-0.471632
#### Calculate the range value
range1 = max(forestfire['FFMC'])-min(forestfire['FFMC'])### 77.5
range2 = max(forestfire['DMC'])-min(forestfire['DMC']) ### 290.2
range3 = max(forestfire['DC'])-min(forestfire['DC']) ### 852.7
range4 = max(forestfire['ISI'])-min(forestfire['ISI']) ### 56.1
range5 = max(forestfire['temp'])-min(forestfire['temp']) ## 31.09
range6 = max(forestfire['RH'])-min(forestfire['RH'])### 85
range7 = max(forestfire['wind'])-min(forestfire['wind']) ### 9.0
range8 = max(forestfire['rain'])-min(forestfire['rain']) ### 6.4
range9 = max(forestfire['area'])-min(forestfire['area']) ### 1090.84
range10 = max(forestfire['dayfri'])-min(forestfire['dayfri']) ## 1
range11 = max(forestfire['daymon'])-min(forestfire['daymon']) ## 1
range12 = max(forestfire['monthapr'])-min(forestfire['monthapr']) ## 1
range12 = max(forestfire['monthaug'])-min(forestfire['monthaug']) ## 1
### Calculate skewness and Kurtosis
forestfire.skew() 
forestfire.kurt() 
####Graphidelivery_time Representation 
plt.hist(forestfire["FFMC"])
plt.hist(forestfire["DMC"])
plt.hist(forestfire["DC"])
plt.hist(forestfire["ISI"])
plt.hist(forestfire["temp"])
plt.hist(forestfire["RH"])
plt.hist(forestfire["wind"])
plt.hist(forestfire["rain"])
plt.hist(forestfire["area"])
plt.hist(forestfire["dayfri"])
plt.hist(forestfire["daymon"])
plt.hist(forestfire["daysat"])
plt.hist(forestfire["daysun"])
plt.hist(forestfire["daythu"])
plt.hist(forestfire["daytue"])
plt.hist(forestfire["daywed"])
plt.hist(forestfire["monthapr"])
plt.hist(forestfire["monthaug"])
plt.hist(forestfire["monthdec"])
plt.hist(forestfire["monthfeb"])
plt.hist(forestfire["monthjan"])
plt.hist(forestfire["monthjul"])
plt.hist(forestfire["monthjun"])
plt.hist(forestfire["monthmar"])
plt.hist(forestfire["monthmay"])
plt.hist(forestfire["monthnov"])
plt.hist(forestfire["monthoct"])
plt.hist(forestfire["monthsep"])
plt.hist(forestfire["size_category"])
plt.boxplot(forestfire["FFMC"],0,"rs",0)
plt.boxplot(forestfire["DMC"],0,"rs",0)
plt.boxplot(forestfire["DC"],0,"rs",0)
plt.boxplot(forestfire["ISI"],0,"rs",0)
plt.boxplot(forestfire["temp"],0,"rs",0)
plt.boxplot(forestfire["RH"],0,"rs",0)
plt.boxplot(forestfire["wind"],0,"rs",0)
plt.boxplot(forestfire["rain"],0,"rs",0)
plt.boxplot(forestfire["area"],0,"rs",0)
plt.boxplot(forestfire["dayfri"],0,"rs",0)
plt.boxplot(forestfire["daymon"],0,"rs",0)
plt.boxplot(forestfire["daysat"],0,"rs",0)
plt.boxplot(forestfire["daysun"],0,"rs",0)
plt.boxplot(forestfire["daythu"],0,"rs",0)
plt.boxplot(forestfire["daytue"],0,"rs",0)
plt.boxplot(forestfire["daywed"],0,"rs",0)
plt.boxplot(forestfire["monthapr"],0,"rs",0)
plt.boxplot(forestfire["monthaug"],0,"rs",0)
plt.boxplot(forestfire["monthdec"],0,"rs",0)
plt.boxplot(forestfire["monthfeb"],0,"rs",0)
plt.boxplot(forestfire["monthjan"],0,"rs",0)
plt.boxplot(forestfire["monthjul"],0,"rs",0)
plt.boxplot(forestfire["monthjun"],0,"rs",0)
plt.boxplot(forestfire["monthmar"],0,"rs",0)
plt.boxplot(forestfire["monthmay"],0,"rs",0)
plt.boxplot(forestfire["monthnov"],0,"rs",0)
plt.boxplot(forestfire["monthoct"],0,"rs",0)
plt.boxplot(forestfire["monthsep"],0,"rs",0)
plt.plot(forestfire["FFMC"],forestfire["size_category"],"bo");plt.xlabel("FFMC");plt.ylabel("size_category")
plt.plot(forestfire["DMC"],forestfire["size_category"],"bo");plt.xlabel("DMC");plt.ylabel("size_category")
plt.plot(forestfire["DC"],forestfire["size_category"],"bo");plt.xlabel("DC");plt.ylabel("size_category")
plt.plot(forestfire["ISI"],forestfire["size_category"],"bo");plt.xlabel("ISI");plt.ylabel("size_category")
plt.plot(forestfire["temp"],forestfire["size_category"],"bo");plt.xlabel("temp");plt.ylabel("size_category")
plt.plot(forestfire["RH"],forestfire["size_category"],"bo");plt.xlabel("RH");plt.ylabel("size_category")
plt.plot(forestfire["wind"],forestfire["size_category"],"bo");plt.xlabel("wind");plt.ylabel("size_category")
plt.plot(forestfire["rain"],forestfire["size_category"],"bo");plt.xlabel("rain");plt.ylabel("size_category")
plt.plot(forestfire["area"],forestfire["size_category"],"bo");plt.xlabel("area");plt.ylabel("size_category")
plt.plot(forestfire["dayfri"],forestfire["size_category"],"bo");plt.xlabel("dayfri");plt.ylabel("size_category")
plt.plot(forestfire["daymon"],forestfire["size_category"],"bo");plt.xlabel("daymon");plt.ylabel("size_category")
plt.plot(forestfire["daysat"],forestfire["size_category"],"bo");plt.xlabel("daysat");plt.ylabel("size_category")
plt.plot(forestfire["daysun"],forestfire["size_category"],"bo");plt.xlabel("daysun");plt.ylabel("size_category")
plt.plot(forestfire["daythu"],forestfire["size_category"],"bo");plt.xlabel("daythu");plt.ylabel("size_category")
plt.plot(forestfire["daytue"],forestfire["size_category"],"bo");plt.xlabel("daytue");plt.ylabel("size_category")
plt.plot(forestfire["daywed"],forestfire["size_category"],"bo");plt.xlabel("daywed");plt.ylabel("size_category")
plt.plot(forestfire["monthapr"],forestfire["size_category"],"bo");plt.xlabel("monthapr");plt.ylabel("size_category")
plt.plot(forestfire["monthaug"],forestfire["size_category"],"bo");plt.xlabel("monthaug");plt.ylabel("size_category")
plt.plot(forestfire["monthdec"],forestfire["size_category"],"bo");plt.xlabel("monthdec");plt.ylabel("size_category")
plt.plot(forestfire["monthfeb"],forestfire["size_category"],"bo");plt.xlabel("monthfeb");plt.ylabel("size_category")
plt.plot(forestfire["monthjan"],forestfire["size_category"],"bo");plt.xlabel("monthjan");plt.ylabel("size_category")
plt.plot(forestfire["monthjul"],forestfire["size_category"],"bo");plt.xlabel("monthjul");plt.ylabel("size_category")
plt.plot(forestfire["monthjun"],forestfire["size_category"],"bo");plt.xlabel("monthjun");plt.ylabel("size_category")
plt.plot(forestfire["monthmar"],forestfire["size_category"],"bo");plt.xlabel("monthmar");plt.ylabel("size_category")
plt.plot(forestfire["monthmay"],forestfire["size_category"],"bo");plt.xlabel("monthmay");plt.ylabel("size_category")
plt.plot(forestfire["monthnov"],forestfire["size_category"],"bo");plt.xlabel("monthnov");plt.ylabel("size_category")
plt.plot(forestfire["monthoct"],forestfire["size_category"],"bo");plt.xlabel("monthoct");plt.ylabel("size_category")
plt.plot(forestfire["monthsep"],forestfire["size_category"],"bo");plt.xlabel("monthsep");plt.ylabel("size_category")## Barplot
plt.plot(forestfire["DMC"],forestfire["FFMC"],"bo");plt.xlabel("DMC");plt.ylabel("FFMC")
plt.plot(forestfire["DC"],forestfire["FFMC"],"bo");plt.xlabel("DC");plt.ylabel("FFMC")
plt.plot(forestfire["ISI"],forestfire["FFMC"],"bo");plt.xlabel("ISI");plt.ylabel("FFMC")
plt.plot(forestfire["temp"],forestfire["FFMC"],"bo");plt.xlabel("temp");plt.ylabel("FFMC")
plt.plot(forestfire["RH"],forestfire["FFMC"],"bo");plt.xlabel("RH");plt.ylabel("FFMC")
plt.plot(forestfire["wind"],forestfire["FFMC"],"bo");plt.xlabel("wind");plt.ylabel("FFMC")
plt.plot(forestfire["rain"],forestfire["FFMC"],"bo");plt.xlabel("rain");plt.ylabel("FFMC")
plt.plot(forestfire["area"],forestfire["FFMC"],"bo");plt.xlabel("area");plt.ylabel("FFMC")
plt.plot(forestfire["dayfri"],forestfire["FFMC"],"bo");plt.xlabel("dayfri");plt.ylabel("FFMC")
plt.plot(forestfire["daymon"],forestfire["FFMC"],"bo");plt.xlabel("daymon");plt.ylabel("FFMC")
plt.plot(forestfire["daysat"],forestfire["FFMC"],"bo");plt.xlabel("daysat");plt.ylabel("FFMC")
plt.plot(forestfire["daysun"],forestfire["FFMC"],"bo");plt.xlabel("daysun");plt.ylabel("FFMC")
plt.plot(forestfire["daythu"],forestfire["FFMC"],"bo");plt.xlabel("daythu");plt.ylabel("FFMC")
plt.plot(forestfire["daytue"],forestfire["FFMC"],"bo");plt.xlabel("daytue");plt.ylabel("FFMC")
plt.plot(forestfire["daywed"],forestfire["FFMC"],"bo");plt.xlabel("daywed");plt.ylabel("FFMC")
plt.plot(forestfire["monthapr"],forestfire["FFMC"],"bo");plt.xlabel("monthapr");plt.ylabel("FFMC")
plt.plot(forestfire["monthaug"],forestfire["FFMC"],"bo");plt.xlabel("monthaug");plt.ylabel("FFMC")
plt.plot(forestfire["monthdec"],forestfire["FFMC"],"bo");plt.xlabel("monthdec");plt.ylabel("FFMC")
plt.plot(forestfire["monthfeb"],forestfire["FFMC"],"bo");plt.xlabel("monthfeb");plt.ylabel("FFMC")
plt.plot(forestfire["monthjan"],forestfire["FFMC"],"bo");plt.xlabel("monthjan");plt.ylabel("FFMC")
plt.plot(forestfire["monthjul"],forestfire["FFMC"],"bo");plt.xlabel("monthjul");plt.ylabel("FFMC")
plt.plot(forestfire["monthjun"],forestfire["FFMC"],"bo");plt.xlabel("monthjun");plt.ylabel("FFMC")
plt.plot(forestfire["monthmar"],forestfire["FFMC"],"bo");plt.xlabel("monthmar");plt.ylabel("FFMC")
plt.plot(forestfire["monthmay"],forestfire["FFMC"],"bo");plt.xlabel("monthmay");plt.ylabel("FFMC")
plt.plot(forestfire["monthnov"],forestfire["FFMC"],"bo");plt.xlabel("monthnov");plt.ylabel("FFMC")
plt.plot(forestfire["monthoct"],forestfire["FFMC"],"bo");plt.xlabel("monthoct");plt.ylabel("FFMC")
plt.plot(forestfire["monthsep"],forestfire["FFMC"],"bo");plt.xlabel("monthsep");plt.ylabel("FFMC")
plt.plot(forestfire["DC"],forestfire["DMC"],"bo");plt.xlabel("DC");plt.ylabel("DMC")
plt.plot(forestfire["ISI"],forestfire["DMC"],"bo");plt.xlabel("ISI");plt.ylabel("DMC")
plt.plot(forestfire["temp"],forestfire["DMC"],"bo");plt.xlabel("temp");plt.ylabel("DMC")
plt.plot(forestfire["RH"],forestfire["DMC"],"bo");plt.xlabel("RH");plt.ylabel("DMC")
plt.plot(forestfire["wind"],forestfire["DMC"],"bo");plt.xlabel("wind");plt.ylabel("DMC")
plt.plot(forestfire["rain"],forestfire["DMC"],"bo");plt.xlabel("rain");plt.ylabel("DMC")
plt.plot(forestfire["area"],forestfire["DMC"],"bo");plt.xlabel("area");plt.ylabel("DMC")
plt.plot(forestfire["dayfri"],forestfire["DMC"],"bo");plt.xlabel("dayfri");plt.ylabel("DMC")
plt.plot(forestfire["daymon"],forestfire["DMC"],"bo");plt.xlabel("daymon");plt.ylabel("DMC")
plt.plot(forestfire["daysat"],forestfire["DMC"],"bo");plt.xlabel("daysat");plt.ylabel("DMC")
plt.plot(forestfire["daysun"],forestfire["DMC"],"bo");plt.xlabel("daysun");plt.ylabel("DMC")
plt.plot(forestfire["daythu"],forestfire["DMC"],"bo");plt.xlabel("daythu");plt.ylabel("DMC")
plt.plot(forestfire["daytue"],forestfire["DMC"],"bo");plt.xlabel("daytue");plt.ylabel("DMC")
plt.plot(forestfire["daywed"],forestfire["DMC"],"bo");plt.xlabel("daywed");plt.ylabel("DMC")
plt.plot(forestfire["monthapr"],forestfire["DMC"],"bo");plt.xlabel("monthapr");plt.ylabel("DMC")
plt.plot(forestfire["monthaug"],forestfire["DMC"],"bo");plt.xlabel("monthaug");plt.ylabel("DMC")
plt.plot(forestfire["monthdec"],forestfire["DMC"],"bo");plt.xlabel("monthdec");plt.ylabel("DMC")
plt.plot(forestfire["monthfeb"],forestfire["DMC"],"bo");plt.xlabel("monthfeb");plt.ylabel("DMC")
plt.plot(forestfire["monthjan"],forestfire["DMC"],"bo");plt.xlabel("monthjan");plt.ylabel("DMC")
plt.plot(forestfire["monthjul"],forestfire["DMC"],"bo");plt.xlabel("monthjul");plt.ylabel("DMC")
plt.plot(forestfire["monthjun"],forestfire["DMC"],"bo");plt.xlabel("monthjun");plt.ylabel("DMC")
plt.plot(forestfire["monthmar"],forestfire["DMC"],"bo");plt.xlabel("monthmar");plt.ylabel("DMC")
plt.plot(forestfire["monthmay"],forestfire["DMC"],"bo");plt.xlabel("monthmay");plt.ylabel("DMC")
plt.plot(forestfire["monthnov"],forestfire["DMC"],"bo");plt.xlabel("monthnov");plt.ylabel("DMC")
plt.plot(forestfire["monthoct"],forestfire["DMC"],"bo");plt.xlabel("monthoct");plt.ylabel("DMC")
plt.plot(forestfire["monthsep"],forestfire["DMC"],"bo");plt.xlabel("monthsep");plt.ylabel("DMC")
plt.plot(forestfire["ISI"],forestfire["DC"],"bo");plt.xlabel("ISI");plt.ylabel("DC")
plt.plot(forestfire["temp"],forestfire["DC"],"bo");plt.xlabel("temp");plt.ylabel("DC")
plt.plot(forestfire["RH"],forestfire["DC"],"bo");plt.xlabel("RH");plt.ylabel("DC")
plt.plot(forestfire["wind"],forestfire["DC"],"bo");plt.xlabel("wind");plt.ylabel("DC")
plt.plot(forestfire["rain"],forestfire["DC"],"bo");plt.xlabel("rain");plt.ylabel("DC")
plt.plot(forestfire["area"],forestfire["DC"],"bo");plt.xlabel("area");plt.ylabel("DC")
plt.plot(forestfire["dayfri"],forestfire["DC"],"bo");plt.xlabel("dayfri");plt.ylabel("DC")
plt.plot(forestfire["daymon"],forestfire["DC"],"bo");plt.xlabel("daymon");plt.ylabel("DC")
plt.plot(forestfire["daysat"],forestfire["DC"],"bo");plt.xlabel("daysat");plt.ylabel("DC")
plt.plot(forestfire["daysun"],forestfire["DC"],"bo");plt.xlabel("daysun");plt.ylabel("DC")
plt.plot(forestfire["daythu"],forestfire["DC"],"bo");plt.xlabel("daythu");plt.ylabel("DC")
plt.plot(forestfire["daytue"],forestfire["DC"],"bo");plt.xlabel("daytue");plt.ylabel("DC")
plt.plot(forestfire["daywed"],forestfire["DC"],"bo");plt.xlabel("daywed");plt.ylabel("DC")
plt.plot(forestfire["monthapr"],forestfire["DC"],"bo");plt.xlabel("monthapr");plt.ylabel("DC")
plt.plot(forestfire["monthaug"],forestfire["DC"],"bo");plt.xlabel("monthaug");plt.ylabel("DC")
plt.plot(forestfire["monthdec"],forestfire["DC"],"bo");plt.xlabel("monthdec");plt.ylabel("DC")
plt.plot(forestfire["monthfeb"],forestfire["DC"],"bo");plt.xlabel("monthfeb");plt.ylabel("DC")
plt.plot(forestfire["monthjan"],forestfire["DC"],"bo");plt.xlabel("monthjan");plt.ylabel("DC")
plt.plot(forestfire["monthjul"],forestfire["DC"],"bo");plt.xlabel("monthjul");plt.ylabel("DC")
plt.plot(forestfire["monthjun"],forestfire["DC"],"bo");plt.xlabel("monthjun");plt.ylabel("DC")
plt.plot(forestfire["monthmar"],forestfire["DC"],"bo");plt.xlabel("monthmar");plt.ylabel("DC")
plt.plot(forestfire["monthmay"],forestfire["DC"],"bo");plt.xlabel("monthmay");plt.ylabel("DC")
plt.plot(forestfire["monthnov"],forestfire["DC"],"bo");plt.xlabel("monthnov");plt.ylabel("DC")
plt.plot(forestfire["monthoct"],forestfire["DC"],"bo");plt.xlabel("monthoct");plt.ylabel("DC")
plt.plot(forestfire["monthsep"],forestfire["DC"],"bo");plt.xlabel("monthsep");plt.ylabel("DC")
plt.plot(forestfire["temp"],forestfire["ISI"],"bo");plt.xlabel("temp");plt.ylabel("ISI")
plt.plot(forestfire["RH"],forestfire["ISI"],"bo");plt.xlabel("RH");plt.ylabel("ISI")
plt.plot(forestfire["wind"],forestfire["ISI"],"bo");plt.xlabel("wind");plt.ylabel("ISI")
plt.plot(forestfire["rain"],forestfire["ISI"],"bo");plt.xlabel("rain");plt.ylabel("ISI")
plt.plot(forestfire["area"],forestfire["ISI"],"bo");plt.xlabel("area");plt.ylabel("ISI")
plt.plot(forestfire["dayfri"],forestfire["ISI"],"bo");plt.xlabel("dayfri");plt.ylabel("ISI")
plt.plot(forestfire["daymon"],forestfire["ISI"],"bo");plt.xlabel("daymon");plt.ylabel("ISI")
plt.plot(forestfire["daysat"],forestfire["ISI"],"bo");plt.xlabel("daysat");plt.ylabel("ISI")
plt.plot(forestfire["daysun"],forestfire["ISI"],"bo");plt.xlabel("daysun");plt.ylabel("ISI")
plt.plot(forestfire["daythu"],forestfire["ISI"],"bo");plt.xlabel("daythu");plt.ylabel("ISI")
plt.plot(forestfire["daytue"],forestfire["ISI"],"bo");plt.xlabel("daytue");plt.ylabel("ISI")
plt.plot(forestfire["daywed"],forestfire["ISI"],"bo");plt.xlabel("daywed");plt.ylabel("ISI")
plt.plot(forestfire["monthapr"],forestfire["ISI"],"bo");plt.xlabel("monthapr");plt.ylabel("ISI")
plt.plot(forestfire["monthaug"],forestfire["ISI"],"bo");plt.xlabel("monthaug");plt.ylabel("ISI")
plt.plot(forestfire["monthdec"],forestfire["ISI"],"bo");plt.xlabel("monthdec");plt.ylabel("ISI")
plt.plot(forestfire["monthfeb"],forestfire["ISI"],"bo");plt.xlabel("monthfeb");plt.ylabel("ISI")
plt.plot(forestfire["monthjan"],forestfire["ISI"],"bo");plt.xlabel("monthjan");plt.ylabel("ISI")
plt.plot(forestfire["monthjul"],forestfire["ISI"],"bo");plt.xlabel("monthjul");plt.ylabel("ISI")
plt.plot(forestfire["monthjun"],forestfire["ISI"],"bo");plt.xlabel("monthjun");plt.ylabel("ISI")
plt.plot(forestfire["monthmar"],forestfire["ISI"],"bo");plt.xlabel("monthmar");plt.ylabel("ISI")
plt.plot(forestfire["monthmay"],forestfire["ISI"],"bo");plt.xlabel("monthmay");plt.ylabel("ISI")
plt.plot(forestfire["monthnov"],forestfire["ISI"],"bo");plt.xlabel("monthnov");plt.ylabel("ISI")
plt.plot(forestfire["monthoct"],forestfire["ISI"],"bo");plt.xlabel("monthoct");plt.ylabel("ISI")
plt.plot(forestfire["monthsep"],forestfire["ISI"],"bo");plt.xlabel("monthsep");plt.ylabel("ISI")
plt.plot(forestfire["RH"],forestfire["temp"],"bo");plt.xlabel("RH");plt.ylabel("temp")
plt.plot(forestfire["wind"],forestfire["temp"],"bo");plt.xlabel("wind");plt.ylabel("temp")
plt.plot(forestfire["rain"],forestfire["temp"],"bo");plt.xlabel("rain");plt.ylabel("temp")
plt.plot(forestfire["area"],forestfire["temp"],"bo");plt.xlabel("area");plt.ylabel("temp")
plt.plot(forestfire["dayfri"],forestfire["temp"],"bo");plt.xlabel("dayfri");plt.ylabel("temp")
plt.plot(forestfire["daymon"],forestfire["temp"],"bo");plt.xlabel("daymon");plt.ylabel("temp")
plt.plot(forestfire["daysat"],forestfire["temp"],"bo");plt.xlabel("daysat");plt.ylabel("temp")
plt.plot(forestfire["daysun"],forestfire["temp"],"bo");plt.xlabel("daysun");plt.ylabel("temp")
plt.plot(forestfire["daythu"],forestfire["temp"],"bo");plt.xlabel("daythu");plt.ylabel("temp")
plt.plot(forestfire["daytue"],forestfire["temp"],"bo");plt.xlabel("daytue");plt.ylabel("temp")
plt.plot(forestfire["daywed"],forestfire["temp"],"bo");plt.xlabel("daywed");plt.ylabel("temp")
plt.plot(forestfire["monthapr"],forestfire["temp"],"bo");plt.xlabel("monthapr");plt.ylabel("temp")
plt.plot(forestfire["monthaug"],forestfire["temp"],"bo");plt.xlabel("monthaug");plt.ylabel("temp")
plt.plot(forestfire["monthdec"],forestfire["temp"],"bo");plt.xlabel("monthdec");plt.ylabel("temp")
plt.plot(forestfire["monthfeb"],forestfire["temp"],"bo");plt.xlabel("monthfeb");plt.ylabel("temp")
plt.plot(forestfire["monthjan"],forestfire["temp"],"bo");plt.xlabel("monthjan");plt.ylabel("temp")
plt.plot(forestfire["monthjul"],forestfire["temp"],"bo");plt.xlabel("monthjul");plt.ylabel("temp")
plt.plot(forestfire["monthjun"],forestfire["temp"],"bo");plt.xlabel("monthjun");plt.ylabel("temp")
plt.plot(forestfire["monthmar"],forestfire["temp"],"bo");plt.xlabel("monthmar");plt.ylabel("temp")
plt.plot(forestfire["monthmay"],forestfire["temp"],"bo");plt.xlabel("monthmay");plt.ylabel("temp")
plt.plot(forestfire["monthnov"],forestfire["temp"],"bo");plt.xlabel("monthnov");plt.ylabel("temp")
plt.plot(forestfire["monthoct"],forestfire["temp"],"bo");plt.xlabel("monthoct");plt.ylabel("temp")
plt.plot(forestfire["monthsep"],forestfire["temp"],"bo");plt.xlabel("monthsep");plt.ylabel("temp")
plt.plot(forestfire["wind"],forestfire["RH"],"bo");plt.xlabel("wind");plt.ylabel("RH")
plt.plot(forestfire["rain"],forestfire["RH"],"bo");plt.xlabel("rain");plt.ylabel("RH")
plt.plot(forestfire["area"],forestfire["RH"],"bo");plt.xlabel("area");plt.ylabel("RH")
plt.plot(forestfire["dayfri"],forestfire["RH"],"bo");plt.xlabel("dayfri");plt.ylabel("RH")
plt.plot(forestfire["daymon"],forestfire["RH"],"bo");plt.xlabel("daymon");plt.ylabel("RH")
plt.plot(forestfire["daysat"],forestfire["RH"],"bo");plt.xlabel("daysat");plt.ylabel("RH")
plt.plot(forestfire["daysun"],forestfire["RH"],"bo");plt.xlabel("daysun");plt.ylabel("RH")
plt.plot(forestfire["daythu"],forestfire["RH"],"bo");plt.xlabel("daythu");plt.ylabel("RH")
plt.plot(forestfire["daytue"],forestfire["RH"],"bo");plt.xlabel("daytue");plt.ylabel("RH")
plt.plot(forestfire["daywed"],forestfire["RH"],"bo");plt.xlabel("daywed");plt.ylabel("RH")
plt.plot(forestfire["monthapr"],forestfire["RH"],"bo");plt.xlabel("monthapr");plt.ylabel("RH")
plt.plot(forestfire["monthaug"],forestfire["RH"],"bo");plt.xlabel("monthaug");plt.ylabel("RH")
plt.plot(forestfire["monthdec"],forestfire["RH"],"bo");plt.xlabel("monthdec");plt.ylabel("RH")
plt.plot(forestfire["monthfeb"],forestfire["RH"],"bo");plt.xlabel("monthfeb");plt.ylabel("RH")
plt.plot(forestfire["monthjan"],forestfire["RH"],"bo");plt.xlabel("monthjan");plt.ylabel("RH")
plt.plot(forestfire["monthjul"],forestfire["RH"],"bo");plt.xlabel("monthjul");plt.ylabel("RH")
plt.plot(forestfire["monthjun"],forestfire["RH"],"bo");plt.xlabel("monthjun");plt.ylabel("RH")
plt.plot(forestfire["monthmar"],forestfire["RH"],"bo");plt.xlabel("monthmar");plt.ylabel("RH")
plt.plot(forestfire["monthmay"],forestfire["RH"],"bo");plt.xlabel("monthmay");plt.ylabel("RH")
plt.plot(forestfire["monthnov"],forestfire["RH"],"bo");plt.xlabel("monthnov");plt.ylabel("RH")
plt.plot(forestfire["monthoct"],forestfire["RH"],"bo");plt.xlabel("monthoct");plt.ylabel("RH")
plt.plot(forestfire["monthsep"],forestfire["RH"],"bo");plt.xlabel("monthsep");plt.ylabel("RH")
plt.plot(forestfire["rain"],forestfire["wind"],"bo");plt.xlabel("rain");plt.ylabel("wind")
plt.plot(forestfire["area"],forestfire["wind"],"bo");plt.xlabel("area");plt.ylabel("wind")
plt.plot(forestfire["dayfri"],forestfire["wind"],"bo");plt.xlabel("dayfri");plt.ylabel("wind")
plt.plot(forestfire["daymon"],forestfire["wind"],"bo");plt.xlabel("daymon");plt.ylabel("wind")
plt.plot(forestfire["daysat"],forestfire["wind"],"bo");plt.xlabel("daysat");plt.ylabel("wind")
plt.plot(forestfire["daysun"],forestfire["wind"],"bo");plt.xlabel("daysun");plt.ylabel("wind")
plt.plot(forestfire["daythu"],forestfire["wind"],"bo");plt.xlabel("daythu");plt.ylabel("wind")
plt.plot(forestfire["daytue"],forestfire["wind"],"bo");plt.xlabel("daytue");plt.ylabel("wind")
plt.plot(forestfire["daywed"],forestfire["wind"],"bo");plt.xlabel("daywed");plt.ylabel("wind")
plt.plot(forestfire["monthapr"],forestfire["wind"],"bo");plt.xlabel("monthapr");plt.ylabel("wind")
plt.plot(forestfire["monthaug"],forestfire["wind"],"bo");plt.xlabel("monthaug");plt.ylabel("wind")
plt.plot(forestfire["monthdec"],forestfire["wind"],"bo");plt.xlabel("monthdec");plt.ylabel("wind")
plt.plot(forestfire["monthfeb"],forestfire["wind"],"bo");plt.xlabel("monthfeb");plt.ylabel("wind")
plt.plot(forestfire["monthjan"],forestfire["wind"],"bo");plt.xlabel("monthjan");plt.ylabel("wind")
plt.plot(forestfire["monthjul"],forestfire["wind"],"bo");plt.xlabel("monthjul");plt.ylabel("wind")
plt.plot(forestfire["monthjun"],forestfire["wind"],"bo");plt.xlabel("monthjun");plt.ylabel("wind")
plt.plot(forestfire["monthmar"],forestfire["wind"],"bo");plt.xlabel("monthmar");plt.ylabel("wind")
plt.plot(forestfire["monthmay"],forestfire["wind"],"bo");plt.xlabel("monthmay");plt.ylabel("wind")
plt.plot(forestfire["monthnov"],forestfire["wind"],"bo");plt.xlabel("monthnov");plt.ylabel("wind")
plt.plot(forestfire["monthoct"],forestfire["wind"],"bo");plt.xlabel("monthoct");plt.ylabel("wind")
plt.plot(forestfire["monthsep"],forestfire["wind"],"bo");plt.xlabel("monthsep");plt.ylabel("wind")
plt.plot(forestfire["area"],forestfire["rain"],"bo");plt.xlabel("area");plt.ylabel("rain")
plt.plot(forestfire["dayfri"],forestfire["rain"],"bo");plt.xlabel("dayfri");plt.ylabel("rain")
plt.plot(forestfire["daymon"],forestfire["rain"],"bo");plt.xlabel("daymon");plt.ylabel("rain")
plt.plot(forestfire["daysat"],forestfire["rain"],"bo");plt.xlabel("daysat");plt.ylabel("rain")
plt.plot(forestfire["daysun"],forestfire["rain"],"bo");plt.xlabel("daysun");plt.ylabel("rain")
plt.plot(forestfire["daythu"],forestfire["rain"],"bo");plt.xlabel("daythu");plt.ylabel("rain")
plt.plot(forestfire["daytue"],forestfire["rain"],"bo");plt.xlabel("daytue");plt.ylabel("rain")
plt.plot(forestfire["daywed"],forestfire["rain"],"bo");plt.xlabel("daywed");plt.ylabel("rain")
plt.plot(forestfire["monthapr"],forestfire["rain"],"bo");plt.xlabel("monthapr");plt.ylabel("rain")
plt.plot(forestfire["monthaug"],forestfire["rain"],"bo");plt.xlabel("monthaug");plt.ylabel("rain")
plt.plot(forestfire["monthdec"],forestfire["rain"],"bo");plt.xlabel("monthdec");plt.ylabel("rain")
plt.plot(forestfire["monthfeb"],forestfire["rain"],"bo");plt.xlabel("monthfeb");plt.ylabel("rain")
plt.plot(forestfire["monthjan"],forestfire["rain"],"bo");plt.xlabel("monthjan");plt.ylabel("rain")
plt.plot(forestfire["monthjul"],forestfire["rain"],"bo");plt.xlabel("monthjul");plt.ylabel("rain")
plt.plot(forestfire["monthjun"],forestfire["rain"],"bo");plt.xlabel("monthjun");plt.ylabel("rain")
plt.plot(forestfire["monthmar"],forestfire["rain"],"bo");plt.xlabel("monthmar");plt.ylabel("rain")
plt.plot(forestfire["monthmay"],forestfire["rain"],"bo");plt.xlabel("monthmay");plt.ylabel("rain")
plt.plot(forestfire["monthnov"],forestfire["rain"],"bo");plt.xlabel("monthnov");plt.ylabel("rain")
plt.plot(forestfire["monthoct"],forestfire["rain"],"bo");plt.xlabel("monthoct");plt.ylabel("rain")
plt.plot(forestfire["monthsep"],forestfire["rain"],"bo");plt.xlabel("monthsep");plt.ylabel("rain")
plt.plot(forestfire["dayfri"],forestfire["area"],"bo");plt.xlabel("dayfri");plt.ylabel("area")
plt.plot(forestfire["daymon"],forestfire["area"],"bo");plt.xlabel("daymon");plt.ylabel("area")
plt.plot(forestfire["daysat"],forestfire["area"],"bo");plt.xlabel("daysat");plt.ylabel("area")
plt.plot(forestfire["daysun"],forestfire["area"],"bo");plt.xlabel("daysun");plt.ylabel("area")
plt.plot(forestfire["daythu"],forestfire["area"],"bo");plt.xlabel("daythu");plt.ylabel("area")
plt.plot(forestfire["daytue"],forestfire["area"],"bo");plt.xlabel("daytue");plt.ylabel("area")
plt.plot(forestfire["daywed"],forestfire["area"],"bo");plt.xlabel("daywed");plt.ylabel("area")
plt.plot(forestfire["monthapr"],forestfire["area"],"bo");plt.xlabel("monthapr");plt.ylabel("area")
plt.plot(forestfire["monthaug"],forestfire["area"],"bo");plt.xlabel("monthaug");plt.ylabel("area")
plt.plot(forestfire["monthdec"],forestfire["area"],"bo");plt.xlabel("monthdec");plt.ylabel("area")
plt.plot(forestfire["monthfeb"],forestfire["area"],"bo");plt.xlabel("monthfeb");plt.ylabel("area")
plt.plot(forestfire["monthjan"],forestfire["area"],"bo");plt.xlabel("monthjan");plt.ylabel("area")
plt.plot(forestfire["monthjul"],forestfire["area"],"bo");plt.xlabel("monthjul");plt.ylabel("area")
plt.plot(forestfire["monthjun"],forestfire["area"],"bo");plt.xlabel("monthjun");plt.ylabel("area")
plt.plot(forestfire["monthmar"],forestfire["area"],"bo");plt.xlabel("monthmar");plt.ylabel("area")
plt.plot(forestfire["monthmay"],forestfire["area"],"bo");plt.xlabel("monthmay");plt.ylabel("area")
plt.plot(forestfire["monthnov"],forestfire["area"],"bo");plt.xlabel("monthnov");plt.ylabel("area")
plt.plot(forestfire["monthoct"],forestfire["area"],"bo");plt.xlabel("monthoct");plt.ylabel("area")
plt.plot(forestfire["monthsep"],forestfire["area"],"bo");plt.xlabel("monthsep");plt.ylabel("area")
plt.plot(forestfire["daymon"],forestfire["dayfri"],"bo");plt.xlabel("daymon");plt.ylabel("dayfri")
plt.plot(forestfire["daysat"],forestfire["dayfri"],"bo");plt.xlabel("daysat");plt.ylabel("dayfri")
plt.plot(forestfire["daysun"],forestfire["dayfri"],"bo");plt.xlabel("daysun");plt.ylabel("dayfri")
plt.plot(forestfire["daythu"],forestfire["dayfri"],"bo");plt.xlabel("daythu");plt.ylabel("dayfri")
plt.plot(forestfire["daytue"],forestfire["dayfri"],"bo");plt.xlabel("daytue");plt.ylabel("dayfri")
plt.plot(forestfire["daywed"],forestfire["dayfri"],"bo");plt.xlabel("daywed");plt.ylabel("dayfri")
plt.plot(forestfire["monthapr"],forestfire["dayfri"],"bo");plt.xlabel("monthapr");plt.ylabel("dayfri")
plt.plot(forestfire["monthaug"],forestfire["dayfri"],"bo");plt.xlabel("monthaug");plt.ylabel("dayfri")
plt.plot(forestfire["monthdec"],forestfire["dayfri"],"bo");plt.xlabel("monthdec");plt.ylabel("dayfri")
plt.plot(forestfire["monthfeb"],forestfire["dayfri"],"bo");plt.xlabel("monthfeb");plt.ylabel("dayfri")
plt.plot(forestfire["monthjan"],forestfire["dayfri"],"bo");plt.xlabel("monthjan");plt.ylabel("dayfri")
plt.plot(forestfire["monthjul"],forestfire["dayfri"],"bo");plt.xlabel("monthjul");plt.ylabel("dayfri")
plt.plot(forestfire["monthjun"],forestfire["dayfri"],"bo");plt.xlabel("monthjun");plt.ylabel("dayfri")
plt.plot(forestfire["monthmar"],forestfire["dayfri"],"bo");plt.xlabel("monthmar");plt.ylabel("dayfri")
plt.plot(forestfire["monthmay"],forestfire["dayfri"],"bo");plt.xlabel("monthmay");plt.ylabel("dayfri")
plt.plot(forestfire["monthnov"],forestfire["dayfri"],"bo");plt.xlabel("monthnov");plt.ylabel("dayfri")
plt.plot(forestfire["monthoct"],forestfire["dayfri"],"bo");plt.xlabel("monthoct");plt.ylabel("dayfri")
plt.plot(forestfire["monthsep"],forestfire["dayfri"],"bo");plt.xlabel("monthsep");plt.ylabel("dayfri")
plt.plot(forestfire["daysat"],forestfire["daymon"],"bo");plt.xlabel("daysat");plt.ylabel("daymon")
plt.plot(forestfire["daysun"],forestfire["daymon"],"bo");plt.xlabel("daysun");plt.ylabel("daymon")
plt.plot(forestfire["daythu"],forestfire["daymon"],"bo");plt.xlabel("daythu");plt.ylabel("daymon")
plt.plot(forestfire["daytue"],forestfire["daymon"],"bo");plt.xlabel("daytue");plt.ylabel("daymon")
plt.plot(forestfire["daywed"],forestfire["daymon"],"bo");plt.xlabel("daywed");plt.ylabel("daymon")
plt.plot(forestfire["monthapr"],forestfire["daymon"],"bo");plt.xlabel("monthapr");plt.ylabel("daymon")
plt.plot(forestfire["monthaug"],forestfire["daymon"],"bo");plt.xlabel("monthaug");plt.ylabel("daymon")
plt.plot(forestfire["monthdec"],forestfire["daymon"],"bo");plt.xlabel("monthdec");plt.ylabel("daymon")
plt.plot(forestfire["monthfeb"],forestfire["daymon"],"bo");plt.xlabel("monthfeb");plt.ylabel("daymon")
plt.plot(forestfire["monthjan"],forestfire["daymon"],"bo");plt.xlabel("monthjan");plt.ylabel("daymon")
plt.plot(forestfire["monthjul"],forestfire["daymon"],"bo");plt.xlabel("monthjul");plt.ylabel("daymon")
plt.plot(forestfire["monthjun"],forestfire["daymon"],"bo");plt.xlabel("monthjun");plt.ylabel("daymon")
plt.plot(forestfire["monthmar"],forestfire["daymon"],"bo");plt.xlabel("monthmar");plt.ylabel("daymon")
plt.plot(forestfire["monthmay"],forestfire["daymon"],"bo");plt.xlabel("monthmay");plt.ylabel("daymon")
plt.plot(forestfire["monthnov"],forestfire["daymon"],"bo");plt.xlabel("monthnov");plt.ylabel("daymon")
plt.plot(forestfire["monthoct"],forestfire["daymon"],"bo");plt.xlabel("monthoct");plt.ylabel("daymon")
plt.plot(forestfire["monthsep"],forestfire["daymon"],"bo");plt.xlabel("monthsep");plt.ylabel("daymon")
plt.plot(forestfire["daysun"],forestfire["daysat"],"bo");plt.xlabel("daysun");plt.ylabel("daysat")
plt.plot(forestfire["daythu"],forestfire["daysat"],"bo");plt.xlabel("daythu");plt.ylabel("daysat")
plt.plot(forestfire["daytue"],forestfire["daysat"],"bo");plt.xlabel("daytue");plt.ylabel("daysat")
plt.plot(forestfire["daywed"],forestfire["daysat"],"bo");plt.xlabel("daywed");plt.ylabel("daysat")
plt.plot(forestfire["monthapr"],forestfire["daysat"],"bo");plt.xlabel("monthapr");plt.ylabel("daysat")
plt.plot(forestfire["monthaug"],forestfire["daysat"],"bo");plt.xlabel("monthaug");plt.ylabel("daysat")
plt.plot(forestfire["monthdec"],forestfire["daysat"],"bo");plt.xlabel("monthdec");plt.ylabel("daysat")
plt.plot(forestfire["monthfeb"],forestfire["daysat"],"bo");plt.xlabel("monthfeb");plt.ylabel("daysat")
plt.plot(forestfire["monthjan"],forestfire["daysat"],"bo");plt.xlabel("monthjan");plt.ylabel("daysat")
plt.plot(forestfire["monthjul"],forestfire["daysat"],"bo");plt.xlabel("monthjul");plt.ylabel("daysat")
plt.plot(forestfire["monthjun"],forestfire["daysat"],"bo");plt.xlabel("monthjun");plt.ylabel("daysat")
plt.plot(forestfire["monthmar"],forestfire["daysat"],"bo");plt.xlabel("monthmar");plt.ylabel("daysat")
plt.plot(forestfire["monthmay"],forestfire["daysat"],"bo");plt.xlabel("monthmay");plt.ylabel("daysat")
plt.plot(forestfire["monthnov"],forestfire["daysat"],"bo");plt.xlabel("monthnov");plt.ylabel("daysat")
plt.plot(forestfire["monthoct"],forestfire["daysat"],"bo");plt.xlabel("monthoct");plt.ylabel("daysat")
plt.plot(forestfire["monthsep"],forestfire["daysat"],"bo");plt.xlabel("monthsep");plt.ylabel("daysat")
plt.plot(forestfire["daythu"],forestfire["daysun"],"bo");plt.xlabel("daythu");plt.ylabel("daysun")
plt.plot(forestfire["daytue"],forestfire["daysun"],"bo");plt.xlabel("daytue");plt.ylabel("daysun")
plt.plot(forestfire["daywed"],forestfire["daysun"],"bo");plt.xlabel("daywed");plt.ylabel("daysun")
plt.plot(forestfire["monthapr"],forestfire["daysun"],"bo");plt.xlabel("monthapr");plt.ylabel("daysun")
plt.plot(forestfire["monthaug"],forestfire["daysun"],"bo");plt.xlabel("monthaug");plt.ylabel("daysun")
plt.plot(forestfire["monthdec"],forestfire["daysun"],"bo");plt.xlabel("monthdec");plt.ylabel("daysun")
plt.plot(forestfire["monthfeb"],forestfire["daysun"],"bo");plt.xlabel("monthfeb");plt.ylabel("daysun")
plt.plot(forestfire["monthjan"],forestfire["daysun"],"bo");plt.xlabel("monthjan");plt.ylabel("daysun")
plt.plot(forestfire["monthjul"],forestfire["daysun"],"bo");plt.xlabel("monthjul");plt.ylabel("daysun")
plt.plot(forestfire["monthjun"],forestfire["daysun"],"bo");plt.xlabel("monthjun");plt.ylabel("daysun")
plt.plot(forestfire["monthmar"],forestfire["daysun"],"bo");plt.xlabel("monthmar");plt.ylabel("daysun")
plt.plot(forestfire["monthmay"],forestfire["daysun"],"bo");plt.xlabel("monthmay");plt.ylabel("daysun")
plt.plot(forestfire["monthnov"],forestfire["daysun"],"bo");plt.xlabel("monthnov");plt.ylabel("daysun")
plt.plot(forestfire["monthoct"],forestfire["daysun"],"bo");plt.xlabel("monthoct");plt.ylabel("daysun")
plt.plot(forestfire["monthsep"],forestfire["daysun"],"bo");plt.xlabel("monthsep");plt.ylabel("daysun")
plt.plot(forestfire["daytue"],forestfire["daythu"],"bo");plt.xlabel("daytue");plt.ylabel("daythu")
plt.plot(forestfire["daywed"],forestfire["daythu"],"bo");plt.xlabel("daywed");plt.ylabel("daythu")
plt.plot(forestfire["monthapr"],forestfire["daythu"],"bo");plt.xlabel("monthapr");plt.ylabel("daythu")
plt.plot(forestfire["monthaug"],forestfire["daythu"],"bo");plt.xlabel("monthaug");plt.ylabel("daythu")
plt.plot(forestfire["monthdec"],forestfire["daythu"],"bo");plt.xlabel("monthdec");plt.ylabel("daythu")
plt.plot(forestfire["monthfeb"],forestfire["daythu"],"bo");plt.xlabel("monthfeb");plt.ylabel("daythu")
plt.plot(forestfire["monthjan"],forestfire["daythu"],"bo");plt.xlabel("monthjan");plt.ylabel("daythu")
plt.plot(forestfire["monthjul"],forestfire["daythu"],"bo");plt.xlabel("monthjul");plt.ylabel("daythu")
plt.plot(forestfire["monthjun"],forestfire["daythu"],"bo");plt.xlabel("monthjun");plt.ylabel("daythu")
plt.plot(forestfire["monthmar"],forestfire["daythu"],"bo");plt.xlabel("monthmar");plt.ylabel("daythu")
plt.plot(forestfire["monthmay"],forestfire["daythu"],"bo");plt.xlabel("monthmay");plt.ylabel("daythu")
plt.plot(forestfire["monthnov"],forestfire["daythu"],"bo");plt.xlabel("monthnov");plt.ylabel("daythu")
plt.plot(forestfire["monthoct"],forestfire["daythu"],"bo");plt.xlabel("monthoct");plt.ylabel("daythu")
plt.plot(forestfire["monthsep"],forestfire["daythu"],"bo");plt.xlabel("monthsep");plt.ylabel("daythu")
plt.plot(forestfire["daywed"],forestfire["daytue"],"bo");plt.xlabel("daywed");plt.ylabel("daytue")
plt.plot(forestfire["monthapr"],forestfire["daytue"],"bo");plt.xlabel("monthapr");plt.ylabel("daytue")
plt.plot(forestfire["monthaug"],forestfire["daytue"],"bo");plt.xlabel("monthaug");plt.ylabel("daytue")
plt.plot(forestfire["monthdec"],forestfire["daytue"],"bo");plt.xlabel("monthdec");plt.ylabel("daytue")
plt.plot(forestfire["monthfeb"],forestfire["daytue"],"bo");plt.xlabel("monthfeb");plt.ylabel("daytue")
plt.plot(forestfire["monthjan"],forestfire["daytue"],"bo");plt.xlabel("monthjan");plt.ylabel("daytue")
plt.plot(forestfire["monthjul"],forestfire["daytue"],"bo");plt.xlabel("monthjul");plt.ylabel("daytue")
plt.plot(forestfire["monthjun"],forestfire["daytue"],"bo");plt.xlabel("monthjun");plt.ylabel("daytue")
plt.plot(forestfire["monthmar"],forestfire["daytue"],"bo");plt.xlabel("monthmar");plt.ylabel("daytue")
plt.plot(forestfire["monthmay"],forestfire["daytue"],"bo");plt.xlabel("monthmay");plt.ylabel("daytue")
plt.plot(forestfire["monthnov"],forestfire["daytue"],"bo");plt.xlabel("monthnov");plt.ylabel("daytue")
plt.plot(forestfire["monthoct"],forestfire["daytue"],"bo");plt.xlabel("monthoct");plt.ylabel("daytue")
plt.plot(forestfire["monthsep"],forestfire["daytue"],"bo");plt.xlabel("monthsep");plt.ylabel("daytue")
plt.plot(forestfire["monthapr"],forestfire["daywed"],"bo");plt.xlabel("monthapr");plt.ylabel("daywed")
plt.plot(forestfire["monthaug"],forestfire["daywed"],"bo");plt.xlabel("monthaug");plt.ylabel("daywed")
plt.plot(forestfire["monthdec"],forestfire["daywed"],"bo");plt.xlabel("monthdec");plt.ylabel("daywed")
plt.plot(forestfire["monthfeb"],forestfire["daywed"],"bo");plt.xlabel("monthfeb");plt.ylabel("daywed")
plt.plot(forestfire["monthjan"],forestfire["daywed"],"bo");plt.xlabel("monthjan");plt.ylabel("daywed")
plt.plot(forestfire["monthjul"],forestfire["daywed"],"bo");plt.xlabel("monthjul");plt.ylabel("daywed")
plt.plot(forestfire["monthjun"],forestfire["daywed"],"bo");plt.xlabel("monthjun");plt.ylabel("daywed")
plt.plot(forestfire["monthmar"],forestfire["daywed"],"bo");plt.xlabel("monthmar");plt.ylabel("daywed")
plt.plot(forestfire["monthmay"],forestfire["daywed"],"bo");plt.xlabel("monthmay");plt.ylabel("daywed")
plt.plot(forestfire["monthnov"],forestfire["daywed"],"bo");plt.xlabel("monthnov");plt.ylabel("daywed")
plt.plot(forestfire["monthoct"],forestfire["daywed"],"bo");plt.xlabel("monthoct");plt.ylabel("daywed")
plt.plot(forestfire["monthsep"],forestfire["daywed"],"bo");plt.xlabel("monthsep");plt.ylabel("daywed")
plt.plot(forestfire["monthaug"],forestfire["monthapr"],"bo");plt.xlabel("monthaug");plt.ylabel("monthapr")
plt.plot(forestfire["monthdec"],forestfire["monthapr"],"bo");plt.xlabel("monthdec");plt.ylabel("monthapr")
plt.plot(forestfire["monthfeb"],forestfire["monthapr"],"bo");plt.xlabel("monthfeb");plt.ylabel("monthapr")
plt.plot(forestfire["monthjan"],forestfire["monthapr"],"bo");plt.xlabel("monthjan");plt.ylabel("monthapr")
plt.plot(forestfire["monthjul"],forestfire["monthapr"],"bo");plt.xlabel("monthjul");plt.ylabel("monthapr")
plt.plot(forestfire["monthjun"],forestfire["monthapr"],"bo");plt.xlabel("monthjun");plt.ylabel("monthapr")
plt.plot(forestfire["monthmar"],forestfire["monthapr"],"bo");plt.xlabel("monthmar");plt.ylabel("monthapr")
plt.plot(forestfire["monthmay"],forestfire["monthapr"],"bo");plt.xlabel("monthmay");plt.ylabel("monthapr")
plt.plot(forestfire["monthnov"],forestfire["monthapr"],"bo");plt.xlabel("monthnov");plt.ylabel("monthapr")
plt.plot(forestfire["monthoct"],forestfire["monthapr"],"bo");plt.xlabel("monthoct");plt.ylabel("monthapr")
plt.plot(forestfire["monthsep"],forestfire["monthapr"],"bo");plt.xlabel("monthsep");plt.ylabel("monthapr")
plt.plot(forestfire["monthdec"],forestfire["monthaug"],"bo");plt.xlabel("monthdec");plt.ylabel("monthaug")
plt.plot(forestfire["monthfeb"],forestfire["monthaug"],"bo");plt.xlabel("monthfeb");plt.ylabel("monthaug")
plt.plot(forestfire["monthjan"],forestfire["monthaug"],"bo");plt.xlabel("monthjan");plt.ylabel("monthaug")
plt.plot(forestfire["monthjul"],forestfire["monthaug"],"bo");plt.xlabel("monthjul");plt.ylabel("monthaug")
plt.plot(forestfire["monthjun"],forestfire["monthaug"],"bo");plt.xlabel("monthjun");plt.ylabel("monthaug")
plt.plot(forestfire["monthmar"],forestfire["monthaug"],"bo");plt.xlabel("monthmar");plt.ylabel("monthaug")
plt.plot(forestfire["monthmay"],forestfire["monthaug"],"bo");plt.xlabel("monthmay");plt.ylabel("monthaug")
plt.plot(forestfire["monthnov"],forestfire["monthaug"],"bo");plt.xlabel("monthnov");plt.ylabel("monthaug")
plt.plot(forestfire["monthoct"],forestfire["monthaug"],"bo");plt.xlabel("monthoct");plt.ylabel("monthaug")
plt.plot(forestfire["monthsep"],forestfire["monthaug"],"bo");plt.xlabel("monthsep");plt.ylabel("monthaug")
plt.plot(forestfire["monthfeb"],forestfire["monthdec"],"bo");plt.xlabel("monthfeb");plt.ylabel("monthdec")
plt.plot(forestfire["monthjan"],forestfire["monthdec"],"bo");plt.xlabel("monthjan");plt.ylabel("monthdec")
plt.plot(forestfire["monthjul"],forestfire["monthdec"],"bo");plt.xlabel("monthjul");plt.ylabel("monthdec")
plt.plot(forestfire["monthjun"],forestfire["monthdec"],"bo");plt.xlabel("monthjun");plt.ylabel("monthdec")
plt.plot(forestfire["monthmar"],forestfire["monthdec"],"bo");plt.xlabel("monthmar");plt.ylabel("monthdec")
plt.plot(forestfire["monthmay"],forestfire["monthdec"],"bo");plt.xlabel("monthmay");plt.ylabel("monthdec")
plt.plot(forestfire["monthnov"],forestfire["monthdec"],"bo");plt.xlabel("monthnov");plt.ylabel("monthdec")
plt.plot(forestfire["monthoct"],forestfire["monthdec"],"bo");plt.xlabel("monthoct");plt.ylabel("monthdec")
plt.plot(forestfire["monthsep"],forestfire["monthdec"],"bo");plt.xlabel("monthsep");plt.ylabel("monthdec")
plt.plot(forestfire["monthjan"],forestfire["monthfeb"],"bo");plt.xlabel("monthjan");plt.ylabel("monthfeb")
plt.plot(forestfire["monthjul"],forestfire["monthfeb"],"bo");plt.xlabel("monthjul");plt.ylabel("monthfeb")
plt.plot(forestfire["monthjun"],forestfire["monthfeb"],"bo");plt.xlabel("monthjun");plt.ylabel("monthfeb")
plt.plot(forestfire["monthmar"],forestfire["monthfeb"],"bo");plt.xlabel("monthmar");plt.ylabel("monthfeb")
plt.plot(forestfire["monthmay"],forestfire["monthfeb"],"bo");plt.xlabel("monthmay");plt.ylabel("monthfeb")
plt.plot(forestfire["monthnov"],forestfire["monthfeb"],"bo");plt.xlabel("monthnov");plt.ylabel("monthfeb")
plt.plot(forestfire["monthoct"],forestfire["monthfeb"],"bo");plt.xlabel("monthoct");plt.ylabel("monthfeb")
plt.plot(forestfire["monthsep"],forestfire["monthfeb"],"bo");plt.xlabel("monthsep");plt.ylabel("monthfeb")
plt.plot(forestfire["monthjul"],forestfire["monthjan"],"bo");plt.xlabel("monthjul");plt.ylabel("monthjan")
plt.plot(forestfire["monthjun"],forestfire["monthjan"],"bo");plt.xlabel("monthjun");plt.ylabel("monthjan")
plt.plot(forestfire["monthmar"],forestfire["monthjan"],"bo");plt.xlabel("monthmar");plt.ylabel("monthjan")
plt.plot(forestfire["monthmay"],forestfire["monthjan"],"bo");plt.xlabel("monthmay");plt.ylabel("monthjan")
plt.plot(forestfire["monthnov"],forestfire["monthjan"],"bo");plt.xlabel("monthnov");plt.ylabel("monthjan")
plt.plot(forestfire["monthoct"],forestfire["monthjan"],"bo");plt.xlabel("monthoct");plt.ylabel("monthjan")
plt.plot(forestfire["monthsep"],forestfire["monthjan"],"bo");plt.xlabel("monthsep");plt.ylabel("monthjan")
plt.plot(forestfire["monthjun"],forestfire["monthjul"],"bo");plt.xlabel("monthjun");plt.ylabel("monthjul")
plt.plot(forestfire["monthmar"],forestfire["monthjul"],"bo");plt.xlabel("monthmar");plt.ylabel("monthjul")
plt.plot(forestfire["monthmay"],forestfire["monthjul"],"bo");plt.xlabel("monthmay");plt.ylabel("monthjul")
plt.plot(forestfire["monthnov"],forestfire["monthjul"],"bo");plt.xlabel("monthnov");plt.ylabel("monthjul")
plt.plot(forestfire["monthoct"],forestfire["monthjul"],"bo");plt.xlabel("monthoct");plt.ylabel("monthjul")
plt.plot(forestfire["monthsep"],forestfire["monthjul"],"bo");plt.xlabel("monthsep");plt.ylabel("monthjul")
plt.plot(forestfire["monthmar"],forestfire["monthjun"],"bo");plt.xlabel("monthmar");plt.ylabel("monthjun")
plt.plot(forestfire["monthmay"],forestfire["monthjun"],"bo");plt.xlabel("monthmay");plt.ylabel("monthjun")
plt.plot(forestfire["monthnov"],forestfire["monthjun"],"bo");plt.xlabel("monthnov");plt.ylabel("monthjun")
plt.plot(forestfire["monthoct"],forestfire["monthjun"],"bo");plt.xlabel("monthoct");plt.ylabel("monthjun")
plt.plot(forestfire["monthsep"],forestfire["monthjun"],"bo");plt.xlabel("monthsep");plt.ylabel("monthjun")
plt.plot(forestfire["monthmay"],forestfire["monthmar"],"bo");plt.xlabel("monthmay");plt.ylabel("monthmar")
plt.plot(forestfire["monthnov"],forestfire["monthmar"],"bo");plt.xlabel("monthnov");plt.ylabel("monthmar")
plt.plot(forestfire["monthoct"],forestfire["monthmar"],"bo");plt.xlabel("monthoct");plt.ylabel("monthmar")
plt.plot(forestfire["monthsep"],forestfire["monthmar"],"bo");plt.xlabel("monthsep");plt.ylabel("monthmar")
plt.plot(forestfire["monthnov"],forestfire["monthmay"],"bo");plt.xlabel("monthnov");plt.ylabel("monthmay")
plt.plot(forestfire["monthoct"],forestfire["monthmay"],"bo");plt.xlabel("monthoct");plt.ylabel("monthmay")
plt.plot(forestfire["monthsep"],forestfire["monthmay"],"bo");plt.xlabel("monthsep");plt.ylabel("monthmay")
plt.plot(forestfire["monthoct"],forestfire["monthnov"],"bo");plt.xlabel("monthoct");plt.ylabel("monthnov")
plt.plot(forestfire["monthsep"],forestfire["monthnov"],"bo");plt.xlabel("monthsep");plt.ylabel("monthnov")
plt.plot(forestfire["monthsep"],forestfire["monthoct"],"bo");plt.xlabel("monthsep");plt.ylabel("monthoct")
pd.crosstab(forestfire["FFMC"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["DMC"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["DC"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["ISI"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["temp"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["RH"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["size_category"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["DMC"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["DC"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["ISI"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["temp"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["RH"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["FFMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["DC"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["ISI"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["temp"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["RH"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["DMC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["ISI"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["temp"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["RH"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["DC"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["temp"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["RH"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["ISI"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["RH"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["temp"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["wind"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["RH"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["rain"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["wind"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["area"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["rain"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["dayfri"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["area"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daymon"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["dayfri"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysat"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["daymon"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daysun"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["daysat"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daythu"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["daysun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daytue"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["daythu"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["daywed"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["daytue"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthapr"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["daywed"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthaug"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthapr"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthdec"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthaug"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthfeb"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthdec"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjan"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthfeb"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjul"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthjan"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthjun"],forestfire["monthjul"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthjul"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthjul"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthjul"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthjul"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthjul"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmar"],forestfire["monthjun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthjun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthjun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthjun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthjun"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthmay"],forestfire["monthmar"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthmar"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthmar"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthmar"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthnov"],forestfire["monthmay"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthmay"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthmay"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthoct"],forestfire["monthnov"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthnov"]).plot(kind = "bar",width=1.85)
pd.crosstab(forestfire["monthsep"],forestfire["monthoct"]).plot(kind = "bar",width=1.85)
import seaborn as sns 
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="FFMC",y="size_category",data=forestfire)
sns.boxplot(x="DMC",y="size_category",data=forestfire)
sns.boxplot(x="DC",y="size_category",data=forestfire)
sns.boxplot(x="ISI",y="size_category",data=forestfire)
sns.boxplot(x="temp",y="size_category",data=forestfire)
sns.boxplot(x="RH",y="size_category",data=forestfire)
sns.boxplot(x="wind",y="size_category",data=forestfire)
sns.boxplot(x="rain",y="size_category",data=forestfire)
sns.boxplot(x="area",y="size_category",data=forestfire)
sns.pairplot(forestfire.iloc[:,0:10]) # histogram of each column and scatter plot of each variable with respect to other columns
sns.pairplot(forestfire,hue="size_category",size=5)
forestfire["month"].value_counts()
forestfire["day"].value_counts()
forestfire["FFMC"].value_counts()
forestfire["DMC"].value_counts()
forestfire["DC"].value_counts()
forestfire["ISI"].value_counts()
forestfire["temp"].value_counts()
forestfire["RH"].value_counts()
forestfire["wind"].value_counts()
forestfire["rain"].value_counts()
forestfire["area"].value_counts()
forestfire["size_category"].value_counts()
forestfire["month"].value_counts().plot(kind="pie")
forestfire["day"].value_counts().plot(kind="pie")
forestfire["FFMC"].value_counts().plot(kind="pie")
forestfire["DMC"].value_counts().plot(kind="pie")
forestfire["DC"].value_counts().plot(kind="pie")
forestfire["ISI"].value_counts().plot(kind="pie")
forestfire["temp"].value_counts().plot(kind="pie")
forestfire["RH"].value_counts().plot(kind="pie")
forestfire["wind"].value_counts().plot(kind="pie")
forestfire["rain"].value_counts().plot(kind="pie")
forestfire["area"].value_counts().plot(kind="pie")
forestfire["size_category"].value_counts().plot(kind="pie")
#sns.pairplot(forestfire,hue="area",size=4,diag_kind = "kde")
help(plt.plot) # explore different visualizations among the scatter plot
import scipy.stats as stats
import pylab   
# Checking Whether data is normally distributed
stats.probplot(forestfire['FFMC'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['FFMC']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['FFMC']),dist="norm",plot=pylab)
stats.probplot((forestfire['FFMC'] * forestfire['FFMC']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['FFMC']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['FFMC'])*np.exp(forestfire['FFMC']),dist="norm",plot=pylab)
reci_1=1/forestfire['FFMC']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['FFMC'] * forestfire['FFMC'])+forestfire['FFMC']),dist="norm",plot=pylab)
stats.probplot(forestfire['DMC'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['DMC']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['DMC']),dist="norm",plot=pylab)
stats.probplot((forestfire['DMC'] * forestfire['DMC']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['DMC']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['DMC'])*np.exp(forestfire['DMC']),dist="norm",plot=pylab)
reci_2=1/forestfire['DMC']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['DMC'] * forestfire['DMC'])+forestfire['DMC']),dist="norm",plot=pylab)
stats.probplot(forestfire['DC'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['DC']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['DC']),dist="norm",plot=pylab)
stats.probplot((forestfire['DC'] * forestfire['DC']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['DC']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['DC'])*np.exp(forestfire['DC']),dist="norm",plot=pylab)
reci_3=1/forestfire['DC']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['DC'] * forestfire['DC'])+forestfire['DC']),dist="norm",plot=pylab)
stats.probplot(forestfire['ISI'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['ISI']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['ISI']),dist="norm",plot=pylab)
stats.probplot((forestfire['ISI'] * forestfire['ISI']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['ISI']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['ISI'])*np.exp(forestfire['ISI']),dist="norm",plot=pylab)
reci_4=1/forestfire['ISI']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['ISI'] * forestfire['ISI'])+forestfire['ISI']),dist="norm",plot=pylab)
stats.probplot(forestfire['temp'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['temp']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['temp']),dist="norm",plot=pylab)
stats.probplot((forestfire['temp'] * forestfire['temp']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['temp']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['temp'])*np.exp(forestfire['temp']),dist="norm",plot=pylab)
reci_5=1/forestfire['temp']
reci_5_2=reci_5 * reci_5
reci_5_4=reci_5_2 * reci_5_2
stats.probplot(reci_5*reci_5,dist="norm",plot=pylab)
stats.probplot(reci_5_2,dist="norm",plot=pylab)
stats.probplot(reci_5_4,dist="norm",plot=pylab)
stats.probplot(reci_5_4*reci_5_4,dist="norm",plot=pylab)
stats.probplot((reci_5_4*reci_5_4)*(reci_5_4*reci_5_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['temp'] * forestfire['temp'])+forestfire['temp']),dist="norm",plot=pylab)
stats.probplot(forestfire['RH'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['RH']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['RH']),dist="norm",plot=pylab)
stats.probplot((forestfire['RH'] * forestfire['RH']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['RH']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['RH'])*np.exp(forestfire['RH']),dist="norm",plot=pylab)
reci_6=1/forestfire['RH']
reci_6_2=reci_6 * reci_6
reci_6_4=reci_6_2 * reci_6_2
stats.probplot(reci_6*reci_6,dist="norm",plot=pylab)
stats.probplot(reci_6_2,dist="norm",plot=pylab)
stats.probplot(reci_6_4,dist="norm",plot=pylab)
stats.probplot(reci_6_4*reci_6_4,dist="norm",plot=pylab)
stats.probplot((reci_6_4*reci_6_4)*(reci_6_4*reci_6_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['RH'] * forestfire['RH'])+forestfire['RH']),dist="norm",plot=pylab)
stats.probplot(forestfire['wind'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['wind']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['wind']),dist="norm",plot=pylab)
stats.probplot((forestfire['wind'] * forestfire['wind']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['wind']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['wind'])*np.exp(forestfire['wind']),dist="norm",plot=pylab)
reci_7=1/forestfire['wind']
reci_7_2=reci_7 * reci_7
reci_7_4=reci_7_2 * reci_7_2
stats.probplot(reci_7*reci_7,dist="norm",plot=pylab)
stats.probplot(reci_7_2,dist="norm",plot=pylab)
stats.probplot(reci_7_4,dist="norm",plot=pylab)
stats.probplot(reci_7_4*reci_7_4,dist="norm",plot=pylab)
stats.probplot((reci_7_4*reci_7_4)*(reci_7_4*reci_7_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['wind'] * forestfire['wind'])+forestfire['wind']),dist="norm",plot=pylab)
stats.probplot(forestfire['rain'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['rain']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['rain']),dist="norm",plot=pylab)
stats.probplot((forestfire['rain'] * forestfire['rain']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['rain']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['rain'])*np.exp(forestfire['rain']),dist="norm",plot=pylab)
reci_8=1/forestfire['rain']
reci_8_2=reci_8 * reci_8
reci_8_4=reci_8_2 * reci_8_2
stats.probplot(reci_8*reci_8,dist="norm",plot=pylab)
stats.probplot(reci_8_2,dist="norm",plot=pylab)
stats.probplot(reci_8_4,dist="norm",plot=pylab)
stats.probplot(reci_8_4*reci_8_4,dist="norm",plot=pylab)
stats.probplot((reci_8_4*reci_8_4)*(reci_8_4*reci_8_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['rain'] * forestfire['rain'])+forestfire['rain']),dist="norm",plot=pylab)
stats.probplot(forestfire['area'],dist="norm",plot=pylab)
stats.probplot(np.log(forestfire['area']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(forestfire['area']),dist="norm",plot=pylab)
stats.probplot((forestfire['area'] * forestfire['area']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['area']),dist="norm",plot=pylab)
stats.probplot(np.exp(forestfire['area'])*np.exp(forestfire['area']),dist="norm",plot=pylab)
reci_9=1/forestfire['area']
reci_9_2=reci_9 * reci_9
reci_9_4=reci_9_2 * reci_9_2
stats.probplot(reci_9*reci_9,dist="norm",plot=pylab)
stats.probplot(reci_9_2,dist="norm",plot=pylab)
stats.probplot(reci_9_4,dist="norm",plot=pylab)
stats.probplot(reci_9_4*reci_9_4,dist="norm",plot=pylab)
stats.probplot((reci_9_4*reci_9_4)*(reci_9_4*reci_9_4),dist="norm",plot=pylab)
stats.probplot(((forestfire['area'] * forestfire['area'])+forestfire['area']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### FFMC
stats.norm.ppf(0.975,90.644681,5.520111)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["FFMC"],90.644681,5.520111) # similar to pnorm in R 
#### DMC
stats.norm.ppf(0.975,110.872340,64.046482)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["DMC"],110.872340,64.046482) # similar to pnorm in R 
#### DC
stats.norm.ppf(0.975,547.940039,248.066192)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["DC"],547.940039,248.066192) # similar to pnorm in R 
#### ISI
stats.norm.ppf(0.975, 9.021663,4.559477)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["ISI"], 9.021663,4.559477) # similar to pnorm in R 
#### temp
stats.norm.ppf(0.975,18.889168, 5.806625)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["temp"],18.889168,5.806625) # similar to pnorm in R 
#### RH
stats.norm.ppf(0.975,44.288201,16.317469)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["RH"],44.288201,16.317469) # similar to pnorm in R 
#### wind
stats.norm.ppf(0.975,4.017602,1.791653)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["wind"],4.017602,1.791653) # similar to pnorm in R 
#### rain
stats.norm.ppf(0.975,0.021663,0.295959)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["rain"],0.021663,0.295959) # similar to pnorm in R 
#### area
stats.norm.ppf(0.975,12.847292,63.655818)# similar to qnorm in R 
# cdf => cumulative distributive function 
stats.norm.cdf(forestfire["area"],12.847292,63.655818) # similar to pnorm in R##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
forestfire.corr(method = "pearson")
forestfire.corr(method = "kendall")
###### Lets do normalization
from sklearn import preprocessing
a_array = np.array(forestfire['FFMC'])
normalized_A = preprocessing.normalize([a_array])
b_array = np.array(forestfire['DMC'])
normalized_B = preprocessing.normalize([b_array])
c_array = np.array(forestfire['DC'])
normalized_C = preprocessing.normalize([c_array])
d_array = np.array(forestfire['ISI'])
normalized_D = preprocessing.normalize([d_array])
e_array = np.array(forestfire['temp'])
normalized_E = preprocessing.normalize([e_array])
f_array = np.array(forestfire['RH'])
normalized_F = preprocessing.normalize([f_array])
g_array = np.array(forestfire['wind'])
normalized_G = preprocessing.normalize([g_array])
h_array = np.array(forestfire['rain'])
normalized_H = preprocessing.normalize([h_array])
i_array = np.array(forestfire['area'])
normalized_I = preprocessing.normalize([i_array])
#### to get top 6 rows
forestfire.head(10) # to get top n rows use cars.head(10)
forestfire.tail(10)
# Correlation matrix 
forestfire.corr()
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(forestfire)
##sns.tools.plotting.scatter_matrix(forestfire) ##-> also used for plotting all in one graph
#####Feature Engineering####################################################
####Imputation
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
forestfire = forestfire[forestfire.columns[forestfire.isnull().mean() < threshold]]
#Dropping rows with missing value rate higher than threshold
forestfire = forestfire.loc[forestfire.isnull().mean(axis=1) < threshold]
####Numerical Imputation#######
#####Imputation is a more preferable option rather than dropping because it preserves the data size
#Filling missing values with medians of the columns
forestfire = forestfire.fillna(forestfire.median())
forestfire['FFMC'].fillna(forestfire['FFMC'].value_counts().idxmax(), inplace=True)
forestfire['DMC'].fillna(forestfire['DMC'].value_counts().idxmax(), inplace=True)
forestfire['DC'].fillna(forestfire['DC'].value_counts().idxmax(), inplace=True)
forestfire['ISI'].fillna(forestfire['ISI'].value_counts().idxmax(), inplace=True)
forestfire['temp'].fillna(forestfire['temp'].value_counts().idxmax(), inplace=True)
forestfire['RH'].fillna(forestfire['RH'].value_counts().idxmax(), inplace=True)
forestfire['wind'].fillna(forestfire['wind'].value_counts().idxmax(), inplace=True)
forestfire['rain'].fillna(forestfire['rain'].value_counts().idxmax(), inplace=True)
forestfire['area'].fillna(forestfire['area'].value_counts().idxmax(), inplace=True)
#Max fill function for categorical columns
##forestfire['column_name'].fillna(forestfire['column_name'].value_counts().idxmax(), inplace=True)
#### Lets's handle outliers
#####The best way to detect the outliers is to demonstrate the data visually. All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.
####Two different ways of handling outliers. These will detect them using standard deviation, and percentiles.
####If a value has a distance to the average higher than x * standard deviation, it can be assumed as an outlier. 
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim6 = forestfire['FFMC'].mean () + forestfire['FFMC'].std () * factor   
lower_lim6= forestfire['FFMC'].mean () - forestfire['FFMC'].std () * factor 
forestfire6 = forestfire[(forestfire['FFMC'] < upper_lim6) & (forestfire['FFMC'] > lower_lim6)]
upper_lim7 = forestfire['DMC'].mean () + forestfire['DMC'].std () * factor   
lower_lim7= forestfire['DMC'].mean () - forestfire['DMC'].std () * factor 
forestfire7 = forestfire[(forestfire['DMC'] < upper_lim7) & (forestfire['DMC'] > lower_lim7)]
upper_lim8 = forestfire['DC'].mean () + forestfire['DC'].std () * factor  
lower_lim8 = forestfire['DC'].mean () - forestfire['DC'].std () * factor 
forestfire8 = forestfire[(forestfire['DC'] < upper_lim8) & (forestfire['DC'] > lower_lim8)]
upper_lim9 = forestfire['ISI'].mean () + forestfire['ISI'].std () * factor  
lower_lim9 = forestfire['ISI'].mean () - forestfire['ISI'].std () * factor 
forestfire9 = forestfire[(forestfire['ISI'] < upper_lim9) & (forestfire['ISI'] > lower_lim9)]
upper_lim5 = forestfire['temp'].mean () + forestfire['temp'].std () * factor  
lower_lim5 = forestfire['temp'].mean () - forestfire['temp'].std () * factor 
forestfire5 = forestfire[(forestfire['temp'] < upper_lim5) & (forestfire['temp'] > lower_lim5)]
upper_lim6 = forestfire['RH'].mean () + forestfire['RH'].std () * factor   
lower_lim6= forestfire['RH'].mean () - forestfire['RH'].std () * factor 
forestfire6 = forestfire[(forestfire['RH'] < upper_lim6) & (forestfire['RH'] > lower_lim6)]
upper_lim7 = forestfire['wind'].mean () + forestfire['wind'].std () * factor   
lower_lim7= forestfire['wind'].mean () - forestfire['wind'].std () * factor 
forestfire7 = forestfire[(forestfire['wind'] < upper_lim7) & (forestfire['wind'] > lower_lim7)]
upper_lim8 = forestfire['rain'].mean () + forestfire['rain'].std () * factor  
lower_lim8 = forestfire['rain'].mean () - forestfire['rain'].std () * factor 
forestfire8 = forestfire[(forestfire['rain'] < upper_lim8) & (forestfire['rain'] > lower_lim8)]
upper_lim9 = forestfire['area'].mean () + forestfire['area'].std () * factor  
lower_lim9 = forestfire['area'].mean () - forestfire['area'].std () * factor 
forestfire9 = forestfire[(forestfire['area'] < upper_lim9) & (forestfire['area'] > lower_lim9)]
#############Outlier Detection with Percentiles
#Dropping the outlier rows with Percentiles
upper_lim10 = forestfire['FFMC'].quantile(.95)
lower_lim10 = forestfire['FFMC'].quantile(.05)
forestfire10 = forestfire[(forestfire['FFMC'] < upper_lim10) & (forestfire['FFMC'] > lower_lim10)]
upper_lim11 = forestfire['DMC'].quantile(.95)
lower_lim11 = forestfire['DMC'].quantile(.05)
forestfire11 = forestfire[(forestfire['DMC'] < upper_lim11) & (forestfire['DMC'] > lower_lim11)]
upper_lim12 = forestfire['DC'].quantile(.95)
lower_lim12 = forestfire['DC'].quantile(.05)
forestfire12 = forestfire[(forestfire['DC'] < upper_lim12) & (forestfire['DC'] > lower_lim12)]
upper_lim13 = forestfire['ISI'].quantile(.95)
lower_lim13 = forestfire['ISI'].quantile(.05)
forestfire13 = forestfire[(forestfire['ISI'] < upper_lim13) & (forestfire['ISI'] > lower_lim13)]
upper_lim14 = forestfire['temp'].quantile(.95)
lower_lim14 = forestfire['temp'].quantile(.05)
forestfire14 = forestfire[(forestfire['temp'] < upper_lim14) & (forestfire['temp'] > lower_lim14)]
upper_lim15 = forestfire['RH'].quantile(.95)
lower_lim15 = forestfire['RH'].quantile(.05)
forestfire15 = forestfire[(forestfire['RH'] < upper_lim15) & (forestfire['RH'] > lower_lim15)]
upper_lim16 = forestfire['wind'].quantile(.95)
lower_lim16 = forestfire['wind'].quantile(.05)
forestfire16 = forestfire[(forestfire['wind'] < upper_lim16) & (forestfire['wind'] > lower_lim16)]
upper_lim17 = forestfire['rain'].quantile(.95)
lower_lim17 = forestfire['rain'].quantile(.05)
forestfire17 = forestfire[(forestfire['rain'] < upper_lim17) & (forestfire['rain'] > lower_lim17)]
upper_lim18 = forestfire['area'].quantile(.95)
lower_lim18 = forestfire['area'].quantile(.05)
forestfire18 = forestfire[(forestfire['area'] < upper_lim18) & (forestfire['area'] > lower_lim18)]
### Another option for handling outliers is to cap them instead of dropping. So you can keep your data size and at the end of the day, it might be better for the final model performance.
#### On the other hand, capping can affect the distribution of the data, thus it better not to exaggerate it.
#Capping the outlier rows with Percentiles
forestfire.loc[(forestfire['FFMC'] > upper_lim10)] = upper_lim10
forestfire.loc[(forestfire['FFMC'] < lower_lim10)] = lower_lim10
forestfire.loc[(forestfire['DMC'] > upper_lim11)] = upper_lim11
forestfire.loc[(forestfire['DMC'] < lower_lim11)] = lower_lim11
forestfire.loc[(forestfire['DC'] > upper_lim12)] = upper_lim12
forestfire.loc[(forestfire['DC'] < lower_lim12)] = lower_lim12
forestfire.loc[(forestfire['ISI'] > upper_lim13)] = upper_lim13
forestfire.loc[(forestfire['ISI'] < lower_lim13)] = lower_lim13
forestfire.loc[(forestfire['temp'] > upper_lim14)] = upper_lim14
forestfire.loc[(forestfire['temp'] < lower_lim14)] = lower_lim14
forestfire.loc[(forestfire['RH'] > upper_lim15)] = upper_lim15
forestfire.loc[(forestfire['RH'] < lower_lim15)] = lower_lim15
forestfire.loc[(forestfire['wind'] > upper_lim16)] = upper_lim16
forestfire.loc[(forestfire['wind'] < lower_lim16)] = lower_lim16
forestfire.loc[(forestfire['rain'] > upper_lim17)] = upper_lim17
forestfire.loc[(forestfire['rain'] < lower_lim17)] = lower_lim17
forestfire.loc[(forestfire['area'] > upper_lim18)] = upper_lim18
forestfire.loc[(forestfire['area'] < lower_lim18)] = lower_lim18
###### Let's check Binning:It can be applied on both categorical and numerical data:
#####The main aim of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance
####Numerical Binning
forestfire['bin1'] = pd.cut(forestfire['FFMC'], bins=[18.7,50,96.2], labels=["Less","More"])
forestfire['bin2'] = pd.cut(forestfire['DMC'], bins=[1.1,150,291.3], labels=["Low","Good"])
forestfire['bin3'] = pd.cut(forestfire['DC'], bins=[7.9,430,860.6], labels=["Good","Superb"])
forestfire['bin4'] = pd.cut(forestfire['ISI'], bins=[0,30,56.1], labels=["Less","More"])
forestfire['bin5'] = pd.cut(forestfire['temp'], bins=[2.2,17,33.3], labels=["Good","High"])
forestfire['bin6'] = pd.cut(forestfire['RH'], bins=[15,60,100], labels=["Low","Good"])
forestfire['bin7'] = pd.cut(forestfire['wind'], bins=[0.4,5,9.4], labels=["Good","Superb"])
forestfire['bin8'] = pd.cut(forestfire['rain'], bins=[0,1.5,6.4], labels=["Less","More"])
forestfire['bin9'] = pd.cut(forestfire['area'], bins=[0,500,1090.84], labels=["Good","High"])
conditions = [
    forestfire['month'].str.contains('jan'),
    forestfire['month'].str.contains('feb'),
    forestfire['month'].str.contains('mar'),
    forestfire['month'].str.contains('apr'),
    forestfire['month'].str.contains('may'),
    forestfire['month'].str.contains('jun'),
    forestfire['month'].str.contains('jul'),
    forestfire['month'].str.contains('aug'),
    forestfire['month'].str.contains('sep'),
    forestfire['month'].str.contains('oct'),
    forestfire['month'].str.contains('nov'),
    forestfire['month'].str.contains('dec')]
choices=['1','2','3','4','5','6','7','8','9','10','11','12']
forestfire['choices']=np.select(conditions,choices,default='Other')
conditions1 = [
    forestfire['day'].str.contains('mon'),
    forestfire['day'].str.contains('tue'),    
    forestfire['day'].str.contains('wed'),
    forestfire['day'].str.contains('thu'),
    forestfire['day'].str.contains('fri'),
    forestfire['day'].str.contains('sat'),
    forestfire['day'].str.contains('sun')]
choices1= ['1','2','3','4','5','6','7']
forestfire['choices1']=np.select(conditions1,choices1,default='Other')
###Log Transform- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
###It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
forestfire = pd.DataFrame({'FFMC':forestfire.iloc[:,2]})
forestfire['log+1'] = (forestfire['FFMC']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['FFMC']-forestfire['FFMC'].min()+1).transform(np.log)
###############################################################
forestfire = pd.DataFrame({'DMC':forestfire.iloc[:,3]})
forestfire['log+1'] = (forestfire['DMC']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['DMC']-forestfire['DMC'].min()+1).transform(np.log)
######################################################################
forestfire = pd.DataFrame({'DC':forestfire.iloc[:,4]})
forestfire['log+1'] = (forestfire['DC']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['DC']-forestfire['DC'].min()+1).transform(np.log)
###########################################################
forestfire = pd.DataFrame({'ISI':forestfire.iloc[:,5]})
forestfire['log+1'] = (forestfire['ISI']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['ISI']-forestfire['ISI'].min()+1).transform(np.log)
###############################################################################
forestfire = pd.DataFrame({'temp':forestfire.iloc[:,6]})
forestfire['log+1'] = (forestfire['temp']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['temp']-forestfire['temp'].min()+1).transform(np.log)
#####################################################################################
forestfire = pd.DataFrame({'RH':forestfire.iloc[:,7]})
forestfire['log+1'] = (forestfire['RH']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['RH']-forestfire['RH'].min()+1).transform(np.log)
#####################################################################################
forestfire = pd.DataFrame({'wind':forestfire.iloc[:,8]})
forestfire['log+1'] = (forestfire['wind']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['wind']-forestfire['wind'].min()+1).transform(np.log)
######################################################################################
forestfire = pd.DataFrame({'rain':forestfire.iloc[:,9]})
forestfire['log+1'] = (forestfire['rain']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['rain']-forestfire['rain'].min()+1).transform(np.log)
#######################################################################################
forestfire = pd.DataFrame({'area':forestfire.iloc[:,10]})
forestfire['log+1'] = (forestfire['area']+1).transform(np.log)
#Negative Values Handling
forestfire['log'] = (forestfire['area']-forestfire['area'].min()+1).transform(np.log)
#####One-hot encoding
encoded_columns = pd.get_dummies(forestfire['size_category'])
forestfire = forestfire.join(encoded_columns.add_suffix('_size_category')).drop('size_category', axis=1) 
####Numerical Column Grouping
#sum_cols: List of columns to sum
#mean_cols: List of columns to average
grouped = forestfire.groupby('Salary')
sums = grouped['size_category'].sum().add_suffix('_sum')
avgs = grouped['size_category'].mean().add_suffix('_avg')
####Categorical Column grouping
forestfire.groupby('Salary').agg(lambda x: x.value_counts().index[0])
#####Scaling
#####Normalization (or min-max normalization) scale all values in a fixed range between 0 and 1. 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
##df_norm = norm_func(forestfire.iloc[:,0:15])
#####Standardization (or z-score normalization) scales the values while taking into account standard deviation. If the standard deviation of features
## is different, their range also would differ from each other. This reduces the effect of the outliers in the features.
def stan(i):
    x = (i-i.mean())/(i.std())
    return (x)
##df_std = stan(forestfire.iloc[:,0:9])
##### Feature Extraction
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
##from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
X = forestfire.drop('size_category', axis=1)
Y=forestfire['size_category']
##Y=pd.concat([y,X],axis=1)
##X = pd.get_dummies(X, prefix_sep='_')
Y=LabelEncoder().fit_transform(Y)
Y.astype(float)
##X = pd.get_dummies(X, prefix_sep='_')
X = StandardScaler().fit_transform(X)
##forestfire.drop(fores['month'], inplace = True) 
def forest_test(X, Y):
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y,test_size = 0.30, random_state = 517)
    start = time.process_time()
    trainedforest = RandomForestClassifier(n_estimators=517).fit(X_Train,Y_Train)
    print(time.process_time() - start)   ### 2.109375
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(Y_Test,predictionforest))
    print(classification_report(Y_Test,predictionforest)) ### accuracy - 1.00
    forest_test(X, Y)
#######Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
##labels=['Normal','Excellent']
##bins=['<=50K','>50K']
##Y=pd.cut(Y,bins=bins,labels=labels)
PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
PCA_df = pd.concat([PCA_df, forestfire['size_category']], axis = 1)
PCA_df['size_category'] = LabelEncoder().fit_transform(PCA_df['size_category'])
PCA_df.head()    
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
size_category = LabelEncoder().fit_transform(Y)
colors = ['r', 'b']
for size_category, color in zip(size_category, colors):
    plt.scatter(PCA_df.loc[PCA_df['size_category'] == size_category, 'PC1'], 
                PCA_df.loc[PCA_df['size_category'] == size_category, 'PC2'], 
                c = color)
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('2D PCA', fontsize = 20)
plt.legend(['Normal', 'Excellent'])
plt.grid()
##Running again a Random Forest Classifier using the set of 3 features constructed by PCA (instead of the whole dataset) led to 98% classification accuracy while using just 2 features 95% accuracy.
pca = PCA(n_components=3,svd_solver='full')
X_pca = pca.fit_transform(X)
print(pca.explained_variance_)
forest_test(X_pca, Y)
###Additionally, using our two-dimensional dataset, we can now also visualize the decision boundary used by our Random Forest in order to classify each of the different data points.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y,test_size = 0.30, random_state = 517)
trainedforest = RandomForestClassifier(n_estimators=517).fit(X_Reduced,Y_Reduced)
x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1
y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)
plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')
plt.xlabel('Principal Component 1', fontsize = 12)
plt.ylabel('Principal Component 2', fontsize = 12)
plt.title('Random Forest', fontsize = 20)
plt.show()
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
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_lda.shape[1])
forest_test(X_lda, Y)
#####LDA can also be used as a classifier. Therefore, we can now test how an LDA Classifier can perform in this situation.
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y,test_size = 0.30,random_state = 517) 
start = time.process_time()
lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)
print(time.process_time() - start)
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
start = time.process_time()
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=3000)
X_tsne = tsne.fit_transform(X)
print(time.process_time() - start)
forest_test(X_tsne, Y)
#####Autoencoders are a family of Machine Learning algorithms which can be used as a 
###dimensionality reduction technique. The main difference between Autoencoders and 
##other dimensionality reduction techniques is that Autoencoders use non-linear 
###transformations to project data from a high dimension to a lower one.
from tensorflow import keras
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(3, activation='relu')(input_layer)
decoded = Dense(X.shape[1], activation='softmax')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
##X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=517)
forestfire = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\forestfires.csv")
##forestfire = pd.read_csv("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\SVM\\Assignments\\forestfires.csv")
##forestfire.drop(['workclass','education','maritalstatus','occupation','relationship','race','sex','native'],axis=1,inplace=True)
########################################################################
autoencoder.fit(X_Train, Y_Train,epochs=517,batch_size=517,shuffle=True,verbose = 517,validation_data=(X_Test, Y_Test))            
encoder = Model(input_layer, encoded)
X_ae = encoder.predict(X)
forest_test(X_ae, Y)
##################################################################################
sns.pairplot(data=forestfire)
sns.pairplot(data=forestfire)
##forestfire1.drop(['size_category'])
from sklearn.svm import SVC
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(X_Train,Y_Train)
pred_test_linear = model_linear.predict(X_Test)
np.mean(pred_test_linear==Y_Test)  #### 0.85
# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(X_Train,Y_Train)
pred_test_poly = model_poly.predict(X_Test)
np.mean(pred_test_poly==Y_Test) ### 0.71
# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(X_Train,Y_Train)
pred_test_rbf = model_rbf.predict(X_Test)
np.mean(pred_test_rbf==Y_Test) 
# kernel = tanh
model_tanh = SVC(kernel = "tanh")
model_tanh.fit(X_Train,Y_Train)
pred_test_tanh = model_tanh.predict(X_Test)
np.mean(pred_test_tanh==Y_Test) 
# kernel = laplace
model_laplace = SVC(kernel = "laplace")
model_laplace.fit(X_Train,Y_Train)
pred_test_laplace = model_laplace.predict(X_Test)
np.mean(pred_test_laplace==Y_Test) 
# kernel = bessel
model_bessel = SVC(kernel = "bessel")
model_bessel.fit(X_Train,Y_Train)
pred_test_bessel = model_bessel.predict(X_Test)
np.mean(pred_test_bessel==Y_Test) 
# kernel = anova
model_anova = SVC(kernel = "anova")
model_anova.fit(X_Train,Y_Train)
pred_test_anova = model_anova.predict(X_Test)
np.mean(pred_test_anova==Y_Test) 
# kernel = anova
model_spline = SVC(kernel = "spline")
model_spline.fit(X_Train,Y_Train)
pred_test_spline = model_spline.predict(X_Test)
np.mean(pred_test_spline==Y_Test) 
# kernel = spline
model_spline = SVC(kernel = "spline")
model_spline.fit(X_Train,Y_Train)
pred_test_spline = model_spline.predict(X_Test)
np.mean(pred_test_spline==Y_Test) 
###kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(X_Train,Y_Train)
pred_test_sigmoid = model_sigmoid.predict(X_Test)
np.mean(pred_test_sigmoid==Y_Test) 