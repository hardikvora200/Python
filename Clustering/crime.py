import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
# reading a csv file using pandas library
crime=pd.read_csv("C:\\Users\\sanu\\Downloads\\Desktop\\Documents\\Excelr_1\\Python\\Clustering\\Assignments\\crime_data.csv")
crime.rename({"Unnamed: 0":"a"}, axis="columns",inplace=True)
crime.columns
crime.drop(["a"],axis=1,inplace=True)
crime.columns
# To get the count of null values in the data 
crime.isnull().sum()
crime.shape # 50 4 => Before dropping null values
# To drop null values ( dropping rows)
crime.dropna().shape # 50 4 => After dropping null values
#####Exploratory Data Analysis#########################################################
crime.mean() ## Murder - 7.788, Assault-170.760, UrbanPop- 65.540,Rape - 21.232
crime.median() #### Murder - 7.25, Assault-159.00, UrbanPop- 66.00,Rape - 20.10
crime.mode() 
####Measures of Dispersion
crime.var() 
crime.std() ## Murder - 4.355510, Assault-83.337661, UrbanPop- 14.474763,Rape - 9.366385
#### Calculate the range value
range1 = max(crime['Murder'])-min(crime['Murder'])  ### 16.59
range2 = max(crime['Assault'])-min(crime['Assault']) ### 292
range3 = max(crime['UrbanPop'])-min(crime['UrbanPop']) ### 59
range4 = max(crime['Rape'])-min(crime['Rape']) ### 38.7
### Calculate skewness and Kurtosis
crime.skew() ## Murder - 0.393956, Assault-0.234410, UrbanPop- -0.226009,Rape - 0.801200
crime.kurt() ## Murder - -0.827488, Assault- -1.053848, UrbanPop-  -0.738360,Rape - 0.353964
#### Various graphs 
plt.hist(crime["Murder"])
plt.hist(crime["Assault"])
plt.hist(crime["UrbanPop"])
plt.hist(crime["Rape"])
plt.boxplot(crime["Murder"],0,"rs",0)
plt.boxplot(crime["Assault"],0,"rs",0)
plt.boxplot(crime["UrbanPop"],0,"rs",0)
plt.boxplot(crime["Rape"],0,"rs",0)
# table 
pd.crosstab(crime["Murder"],crime["Assault"])
pd.crosstab(crime["Murder"],crime["UrbanPop"])
pd.crosstab(crime["Murder"],crime["Rape"])
pd.crosstab(crime["Assault"],crime["UrbanPop"])
pd.crosstab(crime["Assault"],crime["Rape"])
pd.crosstab(crime["UrbanPop"],crime["Rape"])
## Barplot
pd.crosstab(crime["Murder"],crime["Assault"]).plot(kind = "bar", width = 1.85)
pd.crosstab(crime["Murder"],crime["UrbanPop"]).plot(kind = "bar", width = 1.85)
pd.crosstab(crime["Murder"],crime["Rape"]).plot(kind = "bar", width = 1.85)
pd.crosstab(crime["Assault"],crime["UrbanPop"]).plot(kind = "bar", width = 1.85)
pd.crosstab(crime["Assault"],crime["Rape"]).plot(kind = "bar", width = 1.85)
pd.crosstab(crime["UrbanPop"],crime["Rape"]).plot(kind = "bar", width = 1.85)
sns.countplot(x="Murder",data=crime,palette="hls")
sns.countplot(x="Assault",data=crime,palette="hls")
sns.countplot(x="UrbanPop",data=crime,palette="hls")
sns.countplot(x="Rape",data=crime,palette="hls")
# getting boxplot of Delivery Time with respect to each category of Sorting Time 
sns.boxplot(x="Murder",y="Assault",data=crime,palette="hls")
sns.boxplot(x="Murder",y="UrbanPop",data=crime,palette="hls")
sns.boxplot(x="Murder",y="Rape",data=crime,palette="hls")
sns.boxplot(x="Assault",y="UrbanPop",data=crime,palette="hls")
sns.boxplot(x="Assault",y="Rape",data=crime,palette="hls")
sns.boxplot(x="UrbanPop",y="Rape",data=crime,palette="hls")
sns.pairplot(crime.iloc[:,0:4]) # histogram of each column and scatter plot of each variable with respect to other columns
crime["Murder"].value_counts()
crime["Assault"].value_counts()
crime["UrbanPop"].value_counts()
crime["Rape"].value_counts()
crime["Murder"].value_counts().plot(kind = "pie")
crime["Assault"].value_counts().plot(kind = "pie")
crime["UrbanPop"].value_counts().plot(kind = "pie")
crime["Rape"].value_counts().plot(kind = "pie")
sns.pairplot(crime,hue="Murder",height=1.2,diag_kind = "kde")
sns.pairplot(crime,hue="Assault",height=1.2,diag_kind = "kde")
sns.pairplot(crime,hue="UrbanPop",height=1.2,diag_kind = "kde")
sns.pairplot(crime,hue="Rape",height=1.2,diag_kind = "kde")
sns.FacetGrid(crime,hue="Assault").map(plt.scatter,"Murder","Assault").add_legend()
sns.FacetGrid(crime,hue="Assault").map(plt.scatter,"UrbanPop","Assault").add_legend()
sns.FacetGrid(crime,hue="Assault").map(plt.scatter,"Rape","Assault").add_legend()
sns.FacetGrid(crime,hue="Murder").map(plt.scatter,"Assault","Murder").add_legend()
sns.FacetGrid(crime,hue="Murder").map(plt.scatter,"UrbanPop","Murder").add_legend()
sns.FacetGrid(crime,hue="Murder").map(plt.scatter,"Rape","Murder").add_legend()
sns.FacetGrid(crime,hue="UrbanPop").map(plt.scatter,"Assault","UrbanPop").add_legend()
sns.FacetGrid(crime,hue="UrbanPop").map(plt.scatter,"Murder","UrbanPop").add_legend()
sns.FacetGrid(crime,hue="UrbanPop").map(plt.scatter,"Rape","UrbanPop").add_legend()
sns.FacetGrid(crime,hue="Rape").map(plt.scatter,"Assault","Rape").add_legend()
sns.FacetGrid(crime,hue="Rape").map(plt.scatter,"UrbanPop","Rape").add_legend()
sns.FacetGrid(crime,hue="Rape").map(plt.scatter,"Murder","Rape").add_legend()
sns.FacetGrid(crime,hue="Assault").map(sns.kdeplot,"Murder").add_legend()
sns.FacetGrid(crime,hue="Assault").map(sns.kdeplot,"UrbanPop").add_legend()
sns.FacetGrid(crime,hue="Assault").map(sns.kdeplot,"Rape").add_legend()
sns.FacetGrid(crime,hue="Murder").map(sns.kdeplot,"Assault").add_legend()
sns.FacetGrid(crime,hue="Murder").map(sns.kdeplot,"UrbanPop").add_legend()
sns.FacetGrid(crime,hue="Murder").map(sns.kdeplot,"Rape").add_legend()
sns.FacetGrid(crime,hue="UrbanPop").map(sns.kdeplot,"Assault").add_legend()
sns.FacetGrid(crime,hue="UrbanPop").map(sns.kdeplot,"Murder").add_legend()
sns.FacetGrid(crime,hue="UrbanPop").map(sns.kdeplot,"Rape").add_legend()
sns.FacetGrid(crime,hue="Rape").map(sns.kdeplot,"Assault").add_legend()
sns.FacetGrid(crime,hue="Rape").map(sns.kdeplot,"Murder").add_legend()
sns.FacetGrid(crime,hue="Rape").map(sns.kdeplot,"UrbanPop").add_legend()
help(plt.plot) # explore different visualizations among the scatter plot
import pylab   
# Checking Whether data is normally distributed
stats.probplot(crime['Murder'],dist="norm",plot=pylab)
stats.probplot(np.log(crime['Murder']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(crime['Murder']),dist="norm",plot=pylab)
stats.probplot((crime['Murder'] * crime['Murder']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['Murder']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['Murder'])*np.exp(crime['Murder']),dist="norm",plot=pylab)
reci_1=1/crime['Murder']
reci_1_2=reci_1 * reci_1
reci_1_4=reci_1_2 * reci_1_2
stats.probplot(reci_1*reci_1,dist="norm",plot=pylab)
stats.probplot(reci_1_2,dist="norm",plot=pylab)
stats.probplot(reci_1_4,dist="norm",plot=pylab)
stats.probplot(reci_1_4*reci_1_4,dist="norm",plot=pylab)
stats.probplot((reci_1_4*reci_1_4)*(reci_1_4*reci_1_4),dist="norm",plot=pylab)
stats.probplot(((crime['Murder'] * crime['Murder'])+crime['Murder']),dist="norm",plot=pylab)
stats.probplot(crime['Assault'],dist="norm",plot=pylab)
stats.probplot(np.log(crime['Assault']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(crime['Assault']),dist="norm",plot=pylab)
stats.probplot((crime['Assault'] * crime['Assault']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['Assault']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['Assault'])*np.exp(crime['Assault']),dist="norm",plot=pylab)
reci_2=1/crime['Assault']
reci_2_2=reci_2 * reci_2
reci_2_4=reci_2_2 * reci_2_2
stats.probplot(reci_2*reci_2,dist="norm",plot=pylab)
stats.probplot(reci_2_2,dist="norm",plot=pylab)
stats.probplot(reci_2_4,dist="norm",plot=pylab)
stats.probplot(reci_2_4*reci_2_4,dist="norm",plot=pylab)
stats.probplot((reci_2_4*reci_2_4)*(reci_2_4*reci_2_4),dist="norm",plot=pylab)
stats.probplot(((crime['Assault'] * crime['Assault'])+crime['Assault']),dist="norm",plot=pylab)
stats.probplot(crime['Rape'],dist="norm",plot=pylab)
stats.probplot(np.log(crime['Rape']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(crime['Rape']),dist="norm",plot=pylab)
stats.probplot((crime['Rape'] * crime['Rape']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['Rape']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['Rape'])*np.exp(crime['Rape']),dist="norm",plot=pylab)
reci_3=1/crime['Rape']
reci_3_2=reci_3 * reci_3
reci_3_4=reci_3_2 * reci_3_2
stats.probplot(reci_3*reci_3,dist="norm",plot=pylab)
stats.probplot(reci_3_2,dist="norm",plot=pylab)
stats.probplot(reci_3_4,dist="norm",plot=pylab)
stats.probplot(reci_3_4*reci_3_4,dist="norm",plot=pylab)
stats.probplot((reci_3_4*reci_3_4)*(reci_3_4*reci_3_4),dist="norm",plot=pylab)
stats.probplot(((crime['Rape'] * crime['Rape'])+crime['Rape']),dist="norm",plot=pylab)
stats.probplot(crime['UrbanPop'],dist="norm",plot=pylab)
stats.probplot(np.log(crime['UrbanPop']),dist="norm",plot=pylab)
stats.probplot(np.sqrt(crime['UrbanPop']),dist="norm",plot=pylab)
stats.probplot((crime['UrbanPop'] * crime['UrbanPop']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['UrbanPop']),dist="norm",plot=pylab)
stats.probplot(np.exp(crime['UrbanPop'])*np.exp(crime['UrbanPop']),dist="norm",plot=pylab)
reci_4=1/crime['UrbanPop']
reci_4_2=reci_4 * reci_4
reci_4_4=reci_4_2 * reci_4_2
stats.probplot(reci_4*reci_4,dist="norm",plot=pylab)
stats.probplot(reci_4_2,dist="norm",plot=pylab)
stats.probplot(reci_4_4,dist="norm",plot=pylab)
stats.probplot(reci_4_4*reci_4_4,dist="norm",plot=pylab)
stats.probplot((reci_4_4*reci_4_4)*(reci_4_4*reci_4_4),dist="norm",plot=pylab)
stats.probplot(((crime['UrbanPop'] * crime['UrbanPop'])+crime['UrbanPop']),dist="norm",plot=pylab)
# ppf => Percent point function 
#### Murder
stats.norm.ppf(0.975,7.788,4.355510)# similar to qnorm in R ---- 16.32464273430405
# cdf => cumulative distributive function 
stats.norm.cdf(crime["Murder"],7.788,4.355510) # similar to pnorm in R 
#### Assault
stats.norm.ppf(0.975,170.760,83.337661)# similar to qnorm in R ---- 334.09881411580824
# cdf => cumulative distributive function 
stats.norm.cdf(crime["Assault"],170.760,83.337661) # similar to pnorm in R 
#### Rape
stats.norm.ppf(0.975,21.232,9.366385)# similar to qnorm in R ---- 39.589777265336195
# cdf => cumulative distributive function 
stats.norm.cdf(crime["Rape"],21.232,9.366385) # similar to pnorm in R 
#### UrbanPop
stats.norm.ppf(0.975, 65.540, 14.474763)# similar to qnorm in R ---- 93.91001416475295
# cdf => cumulative distributive function 
stats.norm.cdf(crime["UrbanPop"], 65.540, 14.474763) # similar to pnorm in R 

##stats.t.ppf(0.975, 13) # similar to qt in R ------  2.1603686564610127
####Correlation 
crime.corr(method = "pearson")
crime.corr(method = "kendall")
np.corrcoef(crime["Murder"],crime["Assault"])
np.corrcoef(crime["Murder"],crime["UrbanPop"])
np.corrcoef(crime["Murder"],crime["Rape"])
np.corrcoef(crime["Assault"],crime["UrbanPop"])
np.corrcoef(crime["Assault"],crime["Rape"])
np.corrcoef(crime["UrbanPop"],crime["Rape"])
# to get top 6 rows
crime.head(30) # to get top n rows use cars.head(10)
crime.tail(10)
# Correlation matrix 
crime.corr()
# Scatter plot between the variables along with histograms
sns.pairplot(crime)
##### Hierarchical clustering
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime.iloc[:,0:])
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,1:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data1.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,0,3]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data2.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,0,3]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data3.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data4.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data5.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,0,3]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data6.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data7.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data8.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data9.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,0,3]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data10.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data11.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data12.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data13.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,0,3]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data14.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data15.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data16.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data17.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data18.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data19.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data20.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data21.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data22.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,4,3,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data23.csv",encoding="utf-8")
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
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime = crime.iloc[:,[1,2,3,4,0]]
crime.head()
# getting aggregate mean of each cluster
crime.iloc[:,0:].groupby(crime.clust).median()
# creating a csv file 
crime.to_csv("crim_data24.csv",encoding="utf-8")
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
crime.drop(["clust"],axis=1,inplace=True)
model.fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()
crime = crime.iloc[:,[1,0,2,3,4]]
crime.iloc[:,0:4].groupby(crime.clust).mean()
crime.to_csv("Crime_kmeans.csv")