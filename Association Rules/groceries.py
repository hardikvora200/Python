import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
groceries = []
# As the file is in transaction data we will be reading data directly 
with open("C:\\Users\\hardi\\OneDrive\\Documents\\Excelr_1\\Python\\Association Rules\\groceries.csv") as f:
    groceries = f.read()
##########################################################    
te=TransactionEncoder()
te_ary=te.fit(groceries).transform(groceries)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)
frequent_itemsets
###########################################################
# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter
item_frequencies = Counter(all_groceries_list)
# after sorting
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))
# barplot of top 10 
import matplotlib.pyplot as plt
plt.bar(x=items[0:11],height = frequencies[0:11],width=0.5,align='center',color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items")
plt.ylabel("Count")
# Creating Data Frame for the transactions data 
# Purpose of converting all list into Series object because to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction
groceries_series.columns = ["transactions"]
# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x=frequent_itemsets.itemsets[0:11],height = frequent_itemsets.support[0:11],width=0.5,align='center',color='rgmyk');plt.xticks(list(range(0,11),),frequent_itemsets.itemsets[0:11])
plt.xlabel('item-sets');plt.ylabel('support')
rules_1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules_1.head(20)
rules_1.sort_values('lift',ascending = False,inplace=True)
rules_1["antecedent_len"] = rules_1["antecedents"].apply(lambda x: len(x))
####we are ony interested in rules that satisfy the following criteria:
####at least 2 antecedents
#####a confidence > 0.75
#####a lift score > 1.2
rules_1[(rules_1['antecedent_len'] >= 2) &(rules_1['confidence'] > 0.65) &(rules_1['lift'] > 1.2) ]
rules_2 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules_2.head(20)
rules_2.sort_values('lift',ascending = False,inplace=True)
rules_2["antecedent_len"] = rules_2["antecedents"].apply(lambda x: len(x))
rules_2
rules_2[(rules_2['antecedent_len'] >= 2) &(rules_2['confidence'] > 0.65) &(rules_2['lift'] > 1.2) ]
########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))
ma_X = rules_1["antecedents"].apply(to_list)+rules_1["consequents"].apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 
rules_no_redudancy  = rules_1.iloc[index_rules,:]
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)
#######################################################################################
def to_list(i):
    return (sorted(list(i)))
ma_X = rules_2["antecedents"].apply(to_list)+rules_2["consequents"].apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
# getting rules without any redudancy 
rules_no_redudancy  = rules_2.iloc[index_rules,:]
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)
