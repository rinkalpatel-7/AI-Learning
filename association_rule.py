from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

#Loading the Data
dataset = [
    ['vanilla wafers', 'bananas' , 'dog food'],
    ['bananas', 'bread', 'yogurt'],
    ['bananas','apples','yogurt'],
    ['vanilla wafers','bananas','whipped cream'],
    ['bread', 'vanilla wafers' , 'yogurt'],
    ['milk', 'bread', 'bananas'],
    ['vanilla wafers', 'apples' , 'bananas'],
    ['yogurt', 'apples', 'vanilla wafers'],
    
]

Data = [['Power Bank', 'Screen Guard' , 'Travel Charger'],
 ['Screen Guard', 'Bluetooth Headset', 'Mobile Cover'],
 ['Screen Guard','Arm Band','Mobile Cover'],
 ['Power Bank','Screen Guard','Leather Pouch'],
 ['Bluetooth Headset', 'Power Bank' , 'Mobile Cover']]
#Data Transformation
# •	The first step is to create Transaction encoding of the dataset. This is performed for ease of reading the data. Each transaction is represented as a row of 1 and 0.
# •	1 represents presence of an item in that transaction. 0 represents absence.

te = TransactionEncoder()
te_ary = te.fit(Data).transform(Data)

df = pd.DataFrame(te_ary, columns=te.columns_)
# print(df)

#Frequent Items from Apriori
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
# print(frequent_itemsets)

assn_cf_rules = association_rules(frequent_itemsets,metric="confidence", min_threshold=0.7)
# print(assn_cf_rules)

# assn_cf_rules['antecedents']
assn_lift_rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1.2)
print(assn_lift_rules)

lift = assn_cf_rules[
    (assn_cf_rules['antecedents'] == frozenset({'Arm Band', 'Mobile Cover'})) &
    (assn_cf_rules['consequents'] == frozenset({'Screen Guard'}))
]['lift'].round(2).values[0]
print(f"Lift value for Arm Band, Mobile Cober -> Screen Guard: {lift}")

support = assn_cf_rules[
    (assn_cf_rules['antecedents'] == frozenset({'Leather Pouch'})) &
    (assn_cf_rules['consequents'] == frozenset({'Screen Guard'}))
]['support'].round(2).values[0]
print(f"support value for Leather Pouch -> Screen Guard: {support}")

dualton_count = assn_cf_rules[assn_cf_rules['antecedents'].apply(lambda x: len(x) == 2)].shape[0]
print(f"Number of scenarios with 2 items in antecedent set: {dualton_count}")