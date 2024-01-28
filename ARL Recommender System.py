# Is Problemi

# Asagidaki sepet bilgilerine gore Association Rule Learning ile en uygun urun onerisi yapalim
# Urun onerileri bir tane ya da daha fazla olabilir
# Karar kurallarini 2010-2011 Germany musterileri uzerinden turetecegiz

# Veri seti

# Online Retail 2 isimli veri seti Ingiltere merkezli bir perakende sirketinin 01/12/2009 - 09/12/2011 tarihleri
# arasindaki online satis islemlerini iceriyor.
# Sirket hediyelik urun satmaktadir ve cogu musterinin toptanci oldugu bilgisi mevcut.

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.expand_frame_repr', False) # Output tek bir satirda
pd.set_option('display.max_columns', None)
pd.ExcelFile("dataset/online_retail_II.xlsx").sheet_names # ['Year 2009-2010', 'Year 2010-2011']
df_ = pd.read_excel('dataset/online_retail_II.xlsx', sheet_name='Year 2010-2011')
df = df_.copy()
df.head()
df.describe() # Quantity ve Price degiskenlerinde negatif degerler var
df.info()
df.isnull().sum()

# StockCode’u POST olan satirlari drop ediyoruz. POST her faturaya eklenen bedel, urunu ifade etmiyor)
df = df[~df['StockCode'].astype(str).str.contains('POST')]
df['StockCode'].astype(str).str.contains('POST').unique() # Kontrol

# Missing valuelari siliyoruz
df.dropna(inplace=True)

# Invoice'da C bulunan degerleri veri setinden cikariyoruz. (C faturanin iptalini ifade ediyor)
df = df[~df['Invoice'].astype(str).str.contains('C')]
df['Invoice'].astype(str).str.contains('C').unique() # Kontrol

# Quantity ve Price degiskenlerinde negatifler vardi cikiyoruz
df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

# Quantity ve Price'da aykiri degerler vardi, onlari baskilayalim
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

columns = ['Quantity', 'Price']
for col in columns:
    replace_with_thresholds(df, col)


# Verisetinden sadece Germany'i secelim
df_ger = df[df['Country'] == 'Germany']
df_ger.head()

# ARL veri yapisini hazirlayalim (invoice-Product Matrix)

df_ger.groupby(['Invoice', 'Description']).agg({'Quantity':'sum'}).head()
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
df_ger.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# apply: fonksiyonu satir ve sutunlarda uygular
# applymap: tum satir ve sutunlarda uygular

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
                applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
                applymap(lambda x: 1 if x > 0 else 0)

inv_pro_df = create_invoice_product_df(df_ger, id=True)
inv_pro_df.head()

# StockCode'u kullandigimizda bize urun aciklamasini kolayca gostermesi icin bir fonksiyon yaratiyoruz
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

# Association Rules
frequent_itemsets = apriori(inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric='support',
                          min_threshold=0.01)

check_id(df, 84347)

# Sepete asamasindaki kullanicilara urun onerisinde bulunalim

product_id = 23172
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

# [0], urunu bir deger olarak istedigimiz icin. Ama gezindigi yerde birden fazla urun de olabilir, sadece kolaylik
# olmasi acisindan bir tane istiyoruz.
recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]): # enumerate, indexi de istedigimiz icin
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]



