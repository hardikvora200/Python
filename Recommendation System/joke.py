import pandas as pd
import numpy as np
joke = pd.read_excel("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr\\Recommender system\\joke.xlsx")
joke.head()
joke.shape  #### (50691,151)
#Lets consider ratings named dataframe with first 10000 rows and all columns from 1(first column being 0) is dataset
ratings = joke.iloc[:10000, 1:]
ratings
#Change the column indices from 0 to 99
ratings.columns = range(ratings.shape[1])
# In the dataset, the null ratings are given as 99.00, so replace all 99.0s with 0
ratings = ratings.replace(99.0, 0)
ratings
#### Still 99 is visible,so replace that one also with 0
# In the dataset, the null ratings are given as 99.00, so replace all 99.0s with 0
ratings = ratings.replace(99, 0)
ratings##### 99 and 99.0 have been replaced with 0 in 10000 rows and 150 columns 
# Visualize the ratings for joke # 148
import matplotlib.pyplot as plt
plt.figure(figsize=(20,12))
plt.hist(ratings[148],bins=5)
plt.xlabel('Rating')
plt.ylabel('Number of ratings')
plt.suptitle('Joke- Ratings/Num of ratings')
#Lets normalize all these ratings using StandardScaler  and save them in ratings_diff variable
from sklearn.preprocessing import StandardScaler
ratings_diff = StandardScaler().fit_transform(ratings)
ratings_diff
# Using the popularity based recommendation system find the jokes that will be highly recommended
#Find the mean for which column in ratings_diff i.e for each joke
#Here each row represents a joke and the columns are different entities who have rated this joke
mean_ratings = ratings_diff.mean(axis = 0)
mean_ratings
#Consider all the mean ratings and find the jokes with the highest mean value and display the top 10 joke IDs
#First create a dataframe
mean_ratings = pd.DataFrame(mean_ratings)
mean_ratings.iloc[:,0]
mean_ratings.iloc[:,0].argsort()[:-20:-1]
mean_ratings.plot()
x = ratings.iloc[1:4,:-100]
##jokeCorr = joke.corrwith(joke[50])
from sklearn.metrics.pairwise import cosine_similarity
df1 = ratings.iloc[:100]
df2 = ratings.iloc[100:200]
x = df1.iloc[1]
cs1 = cosine_similarity(x.values.reshape(1,-1), df1)
cs2 = cosine_similarity(x.values.reshape(1,-1), df2)
cs1[0][1] = 0
print(cs1)
print(cs2)
print(np.argmax(cs1))
print(np.argmax(cs2))
r = [95,0]
cs = [cs1[0][95], cs2[0][0]]
print(cs)    ###################  [0.6679950680417246, 0.6417384628471862]
np.argmax(cs)  ### 0
from sklearn.metrics.pairwise import euclidean_distances
cs3 = euclidean_distances(x.values.reshape(1,-1), df1)
cs4 = euclidean_distances(x.values.reshape(1,-1), df2)
cs3[0][1] = 0
print(cs3)
print(cs4)
print(np.argmax(cs3))  ## 28
print(np.argmax(cs4))   ### 80
r = [28,80]
cs_1 = [cs3[0][28], cs4[0][80]]
print(cs_1)    ###################  [109.970628180892, 97.8845381687016]
np.argmax(cs_1)  ### 0
from sklearn.metrics.pairwise import _argmin_min_reduce
cs5 = _argmin_min_reduce(x.values.reshape(1,-1),df1)
cs6 = _argmin_min_reduce(x.values.reshape(1,-1),df2)
cs5[0][0] = 0
print(cs5)
print(cs6)
print(np.argmax(cs5))  
print(np.argmax(cs6))   
r = [0,0]
cs_2 = [cs5[0][0], cs6[0][0]]
print(cs_2)    ###################  [0,15]
np.argmax(cs_2)  ### 1
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
cs7 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="euclidean")
cs8 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="euclidean")
cs7[0][0] = 1
print(cs7)
print(cs8)
print(np.argmax(cs7))  
print(np.argmax(cs8))   
r = [0,0]
cs_3 = [cs7[0][0], cs8[0][0]]
print(cs_3)    ###################  [1,85]
np.argmax(cs_3)  ### 1
cs9 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="cosine")
cs10 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="cosine")
cs9[0][0] = 1
print(cs9)
print(cs10)
print(np.argmax(cs9))  
print(np.argmax(cs10))   
cs_4 = [cs9[0][0], cs10[0][0]]
print(cs_4)    ###################  [1,0]
np.argmax(cs_4)  ### 0
cs11 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="cityblock")
cs12 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="cityblock")
cs11[0][0] = 1
print(cs11)
print(cs12)
print(np.argmax(cs11))  ## 0
print(np.argmax(cs12))   ## 1
cs_5 = [cs11[0][0], cs12[0][0]]
print(cs_5)    ###################  [1,53]
np.argmax(cs_5)  ### 1
cs13 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="l1")
cs14 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="l1")
cs13[0][0] = 1
print(cs13)
print(cs14)
print(np.argmax(cs13))  ## 0
print(np.argmax(cs14))   ## 1
cs_6 = [cs13[0][0], cs14[0][0]]
print(cs_6)    ###################  [1,53]
np.argmax(cs_6)  ### 1
cs15 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="l2")
cs16 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="l2")
cs15[0][0] = 1
print(cs15)
print(cs16)
print(np.argmax(cs15))  ## 0
print(np.argmax(cs16))   ## 0
cs_7 = [cs15[0][0], cs16[0][0]]
print(cs_7)    ###################  [1,85]
np.argmax(cs_7)  ### 1
cs17 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="manhattan")
cs18 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="manhattan")
cs17[0][0] = 1
print(cs17)
print(cs18)
print(np.argmax(cs17))  ## 0
print(np.argmax(cs18))   ## 1
cs_8 = [cs17[0][0], cs18[0][0]]
print(cs_8)    ###################  [1,53]
np.argmax(cs_8)  ### 1
cs19 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="braycurtis")
cs20 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="braycurtis")
cs19[0][0] = 1
print(cs19)
print(cs20)
print(np.argmax(cs19))  ## 0
print(np.argmax(cs20))   ## 1
cs_9 = [cs19[0][0], cs20[0][0]]
print(cs_9)    ###################  [1,0]
np.argmax(cs_9)  ### 0
cs21 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="canberra")
cs22 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="canberra")
cs21[0][0] = 1
print(cs21)
print(cs22)
print(np.argmax(cs21))  ## 0
print(np.argmax(cs22))   ## 1
cs_10 = [cs21[0][0], cs22[0][0]]
print(cs_10)    ###################  [1,0]
np.argmax(cs_10)  ### 0
cs23 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="chebyshev")
cs24 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="chebyshev")
cs23[0][0] = 1
print(cs23)
print(cs24)
print(np.argmax(cs23))  ## 0
print(np.argmax(cs24))   ## 1
cs_11 = [cs23[0][0], cs24[0][0]]
print(cs_11)    ###################  [1,0]
np.argmax(cs_11)  ### 0
cs25 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="correlation")
cs26 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="correlation")
cs25[0][0] = 1
print(cs25)
print(cs26)
print(np.argmax(cs25))  ## 0
print(np.argmax(cs26))   ## 1
cs_12 = [cs25[0][0], cs26[0][0]]
print(cs_12)    ###################  [1,0]
np.argmax(cs_12)  ### 0
cs27 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="dice")
cs28 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="dice")
cs27[0][0] = 1
print(cs27)
print(cs28)
print(np.argmax(cs27))  ## 0
print(np.argmax(cs28))   ## 0
cs_13 = [cs27[0][0], cs28[0][0]]
print(cs_13)    ###################  [1,43]
np.argmax(cs_13)  ### 1
cs29 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="hamming")
cs30 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="hamming")
cs29[0][0] = 1
print(cs29)
print(cs30)
print(np.argmax(cs29))  ## 0
print(np.argmax(cs30))   ## 0
cs_14 = [cs29[0][0], cs30[0][0]]
print(cs_14)    ###################  [1,37]
np.argmax(cs_14)  ### 1
cs31 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="jaccard")
cs32 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="jaccard")
cs31[0][0] = 1
print(cs31)
print(cs32)
print(np.argmax(cs31))  ## 0
print(np.argmax(cs32))   ## 0
cs_15 = [cs31[0][0], cs32[0][0]]
print(cs_15)    ###################  [1,43]
np.argmax(cs_15)  ### 1
cs33 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="kulsinski")
cs34 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="kulsinski")
cs33[0][0] = 1
print(cs33)
print(cs34)
print(np.argmax(cs33))  ## 0
print(np.argmax(cs34))   ## 0
cs_16 = [cs33[0][0], cs34[0][0]]
print(cs_16)    ###################  [1,73]
np.argmax(cs_16)  ### 0
x = df1.iloc[1]
y=df2.iloc[0]
cs35 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="rogerstanimoto")
cs36 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="rogerstanimoto")
cs35[0][0] = 1
print(cs35)
print(cs36)
print(np.argmax(cs35))  ## 0
print(np.argmax(cs36))   ## 0
cs_17 = [cs35[0][0], cs36[0][0]]
print(cs_17)    ###################  [1,43]
np.argmax(cs_17)  ### 1
cs37 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="minkowski")
cs38 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="minkowski")
cs37[0][0] = 1
print(cs37)
print(cs38)
print(np.argmax(cs37))  ## 0
print(np.argmax(cs38))   ## 0
cs_18 = [cs37[0][0], cs38[0][0]]
print(cs_18)    ###################  [1,85]
np.argmax(cs_18)  ### 1
cs39 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="russellrao")
cs40 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="russellrao")
cs39[0][0] = 1
print(cs39)
print(cs40)
print(np.argmax(cs39))  ## 0
print(np.argmax(cs40))   ## 0
cs_19 = [cs39[0][0], cs40[0][0]]
print(cs_19)    ###################  [1,10]
np.argmax(cs_19)  ### 1
cs41 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="seuclidean")
cs42 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="seuclidean")
cs41[0][0] = 1
print(cs41)
print(cs42)
print(np.argmax(cs41))  ## 1
print(np.argmax(cs42))   ## 1
cs_20 = [cs41[0][0], cs42[0][0]]
print(cs_20)    ###################  [1,0]
np.argmax(cs_20)  ### 0
cs43 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="sokalmichener")
cs44 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="sokalmichener")
cs43[0][0] = 1
print(cs43)
print(cs44)
print(np.argmax(cs43))  ## 0
print(np.argmax(cs44))   ## 0
cs_21 = [cs43[0][0], cs44[0][0]]
print(cs_21)    ###################  [1,43]
np.argmax(cs_21)  ### 1
cs45 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="sokalsneath")
cs46 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="sokalsneath")
cs45[0][0] = 1
print(cs45)
print(cs46)
print(np.argmax(cs45))  ## 0
print(np.argmax(cs46))   ## 0
cs_22 = [cs45[0][0], cs46[0][0]]
print(cs_22)    ###################  [1,43]
np.argmax(cs_22)  ### 1
cs47 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="sqeuclidean")
cs48 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="sqeuclidean")
cs47[0][0] = 1
print(cs47)
print(cs48)
print(np.argmax(cs47))  ## 0
print(np.argmax(cs48))   ## 1
cs_23 = [cs47[0][0], cs48[0][0]]
print(cs_23)    ###################  [1,85]
np.argmax(cs_23)  ### 1
cs49 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df1,metric="yule")
cs50 = pairwise_distances_argmin_min(x.values.reshape(1,-1),df2,metric="yule")
cs49[0][0] = 1
print(cs49)
print(cs50)
print(np.argmax(cs49))  ## 0
print(np.argmax(cs50))   ## 0
cs_24 = [cs49[0][0], cs50[0][0]]
print(cs_24)    ###################  [1,10]
np.argmax(cs_24)  ### 1
from sklearn.metrics.pairwise import pairwise_distances_argmin
cs51 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="euclidean")
cs52 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="euclidean")
print(cs51)  ### [1]
print(cs52)  ### [85]
print(np.argmax(cs51))   ## 0
print(np.argmax(cs52))   ### 0
r = [0,0]
cs_25 = [[1],[85]]
print(cs_25)    ###################  [[1],[85]]
np.argmax(cs_25)  ### 1
cs53 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="cosine")
cs54 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="cosine")
print(cs53)  #### [1]
print(cs54)  #### [0]
print(np.argmax(cs53))   ## 0
print(np.argmax(cs54))   ## 0
cs_26 = [[1], [0]]
print(cs_26)    ###################  [[1],[0]]
np.argmax(cs_26)  ### 0
cs55 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="cityblock")
cs56 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="cityblock")
print(cs55)  #### [1]
print(cs56)  ##### [53]
print(np.argmax(cs55))  ## 0
print(np.argmax(cs56))   ## 0
cs_27 = [[1],[53]]
print(cs_27)    ###################  [[1],[53]]
np.argmax(cs_27)  ### 1
cs57 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="l1")
cs58 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="l1")
print(cs57)  [1]
print(cs58)  [53]
print(np.argmax(cs57))  ## 0
print(np.argmax(cs58))   ## 0
cs_28 = [[1],[53]]
print(cs_28)    ###################   [[1],[53]]
np.argmax(cs_28)  ### 1
cs59 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="l2")
cs60 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="l2")
print(cs59)   #### [1]
print(cs60)   #### [85]
print(np.argmax(cs59))  ## 0
print(np.argmax(cs60))   ## 0
cs_29 = [[1],[85]]
print(cs_29)    ###################  [[1],[85]]
np.argmax(cs_29)  ### 1
cs61 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="manhattan")
cs62 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="manhattan")
print(cs61)    ##### [1]
print(cs62)    ##### [53]
print(np.argmax(cs61))  ## 0
print(np.argmax(cs62))   ## 0
cs_30 = [[1],[53]]
print(cs_30)    ###################  [[1],[53]]
np.argmax(cs_30)  ### 1
cs63 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="braycurtis")
cs64 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="braycurtis")
print(cs63)  ####[1]
print(cs64)  #### [0]
print(np.argmax(cs63))  ## 0
print(np.argmax(cs64))   ## 0
cs_31 = [[1],[0]]
print(cs_31)    ###################  [[1],[0]]
np.argmax(cs_31)  ### 0
cs65 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="canberra")
cs66 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="canberra")
print(cs65)     #### [1]
print(cs66)     #### [0]
print(np.argmax(cs65))  ## 0
print(np.argmax(cs66))   ## 0
cs_32 = [[1],[0]]
print(cs_32)    ###################  [[1],[0]]
np.argmax(cs_32)  ### 0
cs67 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="chebyshev")
cs68 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="chebyshev")
print(cs67)   ### [1]
print(cs68)   ###[0]
print(np.argmax(cs67))  ## 0
print(np.argmax(cs68))   ## 0
cs_33 = [[1],[0]]
print(cs_33)    ###################  [[1],[0]]
np.argmax(cs_33)  ### 0
cs69 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="correlation")
cs70 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="correlation")
print(cs69)   ###### [1]
print(cs70)   #####  [0]
print(np.argmax(cs69))  ## 0
print(np.argmax(cs70))   ## 0
cs_34 = [[1],[0]]
print(cs_34)    ################### [[1],[0]]
np.argmax(cs_34)  ### 0
cs71 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="dice")
cs72 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="dice")
print(cs71)  #### [1]
print(cs72)  #### [43]
print(np.argmax(cs71))  ## 0
print(np.argmax(cs72))   ## 0
cs_35 = [[1],[43]]
print(cs_35)    ###################  [[1],[43]]
np.argmax(cs_35)  ### 1
cs73 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="hamming")
cs74 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="hamming")
print(cs73)    #### [1]
print(cs74)   ##### [37]
print(np.argmax(cs73))  ## 0
print(np.argmax(cs74))   ## 0
cs_36 = [[1],[37]]
print(cs_36)    ###################  [[1],[37]]
np.argmax(cs_36)  ### 1
cs75 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="jaccard")
cs76 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="jaccard")
print(cs75)   ### [1]
print(cs76)    ##### [43]
print(np.argmax(cs75))  ## 0
print(np.argmax(cs76))   ## 0
cs_37 = [[1],[43]]
print(cs_37)    ###################  [[1],[43]]
np.argmax(cs_37)  ### 1
cs77 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="kulsinski")
cs78 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="kulsinski")
print(cs77)    ##### [1]
print(cs78)    ##### [73]
print(np.argmax(cs77))  ## 0
print(np.argmax(cs78))   ## 0
cs_38 = [[1],[73]]
print(cs_38)    ###################[[1],[73]]
np.argmax(cs_38)  ### 1
x = df1.iloc[1]
##y=df2.iloc[0]
cs79 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="rogerstanimoto")
cs80 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="rogerstanimoto")
print(cs79)   #### [1]
print(cs80)   #### [43]
print(np.argmax(cs79))  ## 0
print(np.argmax(cs80))   ## 0
cs_39 = [[1],[43]]
print(cs_39)    ###################  [[1],[43]]
np.argmax(cs_39)  ### 1
cs81 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="minkowski")
cs82 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="minkowski")
print(cs81)   ##### 1
print(cs82)   ##### 85
print(np.argmax(cs81))  ## 0
print(np.argmax(cs82))   ## 0
cs_40 = [[1],[85]]
print(cs_40)    ###################  [[1],[85]]
np.argmax(cs_40)  ### 1
cs83 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="russellrao")
cs84 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="russellrao")
print(cs83)    ######## [1]
print(cs84)    ########  [10]
print(np.argmax(cs83))  ## 0
print(np.argmax(cs84))   ## 0
cs_41 = [[1],[10]]
print(cs_41)    ################### [[1],[10]]
np.argmax(cs_41)  ### 1
cs85 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="seuclidean")
cs86 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="seuclidean")
print(cs85)    #### [0]
print(cs86)    #### [0]
print(np.argmax(cs85))  ## 0
print(np.argmax(cs86))   ## 0
cs_42 = [[0],[0]]
print(cs_42)    ###################  [[0],[0]]
np.argmax(cs_42)  ### 0
cs87 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="sokalmichener")
cs88 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="sokalmichener")
print(cs87)  ##### [1]
print(cs88)   ##### [43]
print(np.argmax(cs87))  ## 0
print(np.argmax(cs88))   ## 0
cs_43 = [[1],[43]]
print(cs_43)    ###################  [[1],[43]]
np.argmax(cs_43)  ### 1
cs89 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="sokalsneath")
cs90 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="sokalsneath")
print(cs89)  ##### [1]
print(cs90)   #### [43]
print(np.argmax(cs89))  ## 0
print(np.argmax(cs90))   ## 0
cs_44 = [[1],[43]]
print(cs_44)    ###################  [[1],[43]]
np.argmax(cs_44)  ### 1
cs91 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="sqeuclidean")
cs92 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="sqeuclidean")
print(cs91)   ###### [1]
print(cs92)   ##### [85]
print(np.argmax(cs91))  ## 0
print(np.argmax(cs92))   ## 0
cs_45 = [[1],[85]]
print(cs_45)    ###################  [[1],[85]]
np.argmax(cs_45)  ### 1
cs93 = pairwise_distances_argmin(x.values.reshape(1,-1),df1,metric="yule")
cs94 = pairwise_distances_argmin(x.values.reshape(1,-1),df2,metric="yule")
print(cs93)  #### [1]
print(cs94)  ##### [10]
print(np.argmax(cs93))  ## 0
print(np.argmax(cs94))   ## 0
cs_46 = [[1],[10]]
print(cs_46)    ###################  [[1],[10]]
np.argmax(cs_46)  ### 1
from sklearn.metrics.pairwise import manhattan_distances
cs95 = manhattan_distances(x.values.reshape(1,-1),df1) 
cs96 = manhattan_distances(x.values.reshape(1,-1), df2)
cs95[0][1] = 0
print(cs95)
print(cs96)
print(np.argmax(cs95))  ## 48
print(np.argmax(cs96))   ### 80
r = [48,80]
cs_47 = [cs95[0][48], cs96[0][80]]
print(cs_47)    ################### [1099.0625, 981.3125]
np.argmax(cs_47)  ### 0
from sklearn.metrics.pairwise import cosine_distances
cs97 = cosine_distances(x.values.reshape(1,-1),df1) 
cs98 = cosine_distances(x.values.reshape(1,-1), df2)
cs97[0][1] = 0
print(cs97)
print(cs98)
print(np.argmax(cs97))  ## 56
print(np.argmax(cs98))   ### 88
r = [56,88]
cs_48 = [cs97[0][56], cs98[0][88]]
print(cs_48)    ################### [1.4412141141891894, 1.5648196693270315]
np.argmax(cs_48)  ### 1
from sklearn.metrics.pairwise import paired_euclidean_distances
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs99 = paired_euclidean_distances(x.values.reshape(1,-1),[df1]) 
cs100 = paired_euclidean_distances(x1.values.reshape(1,-1), [df2])
print(cs99)   ### [34.21532517]
print(cs100)   ### [28.53037718]
print(np.argmax(cs99))  ## 0
print(np.argmax(cs100))   ### 0
r = [0,0]
cs_49 = [[34.21532517],[28.53037718]]
print(cs_49)    #############  [[34.21532517],[28.53037718]]
np.argmax(cs_49)  ### 0
from sklearn.metrics.pairwise import paired_manhattan_distances
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs101 = paired_manhattan_distances(x.values.reshape(1,-1),[df1]) 
cs102 = paired_manhattan_distances(x1.values.reshape(1,-1),[df2])
print(cs101)   ### [126.03125]
print(cs102)   ### array[167.625]
print(np.argmax(cs101))  ## 0
print(np.argmax(cs102))   ### 0
r = [0,0]
cs_50 = [[126.03125],[167.625]]
print(cs_50)    #############  [[126.03125],[167.625]]
np.argmax(cs_50)  ### 1
from sklearn.metrics.pairwise import paired_cosine_distances
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs103 = paired_cosine_distances(x.values.reshape(1,-1),[df1]) 
cs104 = paired_cosine_distances(x1.values.reshape(1,-1),[df2])
print(cs103)   ### [0.5]
print(cs104)   ### [0.5]
print(np.argmax(cs103))  ## 0
print(np.argmax(cs104))   ### 0
r = [0,0]
cs_51 = [[0.5],[0.5]]
print(cs_51)    ############# [[0.5],[0.5]]
np.argmax(cs_51)  ### 0
from sklearn.metrics.pairwise import linear_kernel
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs105 = linear_kernel(x.values.reshape(1,-1),[df1]) 
cs106 = linear_kernel(x1.values.reshape(1,-1),[df2])
cs105[0][0] = 1
print(cs105)   ### [[1.]]
print(cs106)   ### [[0.]]
print(np.argmax(cs105))  ## 0
print(np.argmax(cs106))   ### 0
r = [0,0]
cs_52 = [cs105[0][0],cs106[0][0]]
print(cs_52)    ############# [1.0,1.0]
np.argmax(cs_52)  ### 0
from sklearn.metrics.pairwise import polynomial_kernel
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs107 = polynomial_kernel(x.values.reshape(1,-1),[df1]) 
cs108 = polynomial_kernel(x1.values.reshape(1,-1),[df2])
cs107[0][0] = 1
print(cs107)   ### [[1.]]
print(cs108)   ### [[1.]]
print(np.argmax(cs107))  ## 0
print(np.argmax(cs108))   ### 0
r = [0,0]
cs_53 = [cs107[0][0],cs108[0][0]]
print(cs_53)    #############  [1.0,1.0]
np.argmax(cs_53)  ### 0
from sklearn.metrics.pairwise import sigmoid_kernel
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs109 = sigmoid_kernel(x.values.reshape(1,-1),[df1]) 
cs110 = sigmoid_kernel(x1.values.reshape(1,-1),[df2])
cs109[0][0] = 1
print(cs109)   ### [[1.]]
print(cs110)   ### [[0.76159416]]
print(np.argmax(cs109))  ## 0
print(np.argmax(cs110))   ### 0
r = [0,0]
cs_54 = [cs109[0][0],cs110[0][0]]
print(cs_54)    ############# [1.0, 0.7615941559557649]
np.argmax(cs_54)  ### 0
from sklearn.metrics.pairwise import laplacian_kernel
x = ratings.iloc[100:250,1]
df1 = ratings.iloc[1,:150]
x1 = ratings.iloc[100:175,2]
df2 = ratings.iloc[2,:75]
cs111 = laplacian_kernel(x.values.reshape(1,-1),[df1]) 
cs112 = laplacian_kernel(x1.values.reshape(1,-1),[df2])
cs111[0][0] = 1
print(cs111)   ### [[1.]]
print(cs112)   ### [[0.10699213]]
print(np.argmax(cs111))  ## 0
print(np.argmax(cs112))   ### 0
r = [0,0]
cs_55 = [cs111[0][0],cs112[0][0]]
print(cs_55)    ############# [1.0, 0.10699212985311443]
np.argmax(cs_55)  ### 0
from sklearn.metrics.pairwise import distance_metrics
cs113 = distance_metrics() 
##cs114 = distance_metrics(x1.values.reshape(1,-1),[df2])
print(np.argmax(cs113))  ## 0
print(cs113)
from sklearn.metrics.pairwise import _parallel_pairwise
cs114 = _parallel_pairwise(x.values.reshape(1,-1),[df1], n_jobs=1,func=euclidean_distances) 
cs115 = _parallel_pairwise(x1.values.reshape(1,-1),[df2], n_jobs=2,func=euclidean_distances)
cs114[0][0] = 1
print(cs114)   ### [[1.]]
print(cs115)   ### [[28.53037718]]
print(np.argmax(cs114))  ## 0
print(np.argmax(cs115))   ### 0
r = [0,0]
cs_56 = [cs114[0][0],cs115[0][0]]
print(cs_56)    ############# [1.0, 28.530377177229887]
np.argmax(cs_56)  ### 1