import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cm
from cycler import cycler
sns.set(style="darkgrid")

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)

def figsize():
    return plt.figure(figsize=(20,10))
df = pd.read_csv('export.csv').dropna(how = 'all', axis = 1)




#Cleaning + arrangement des features 

todrop = ['Vendor', 'Id', 'Risk Level',
       'Source', 'Lineitem discount', 'Tax 1 Name', 'Tax 1 Value', 'Phone', 
       'Shipping Phone', 'Notes', 'Note Attributes', 'Payment Method',
       'Payment Reference','Shipping Zip', 'Shipping Province','Billing Name',
       'Billing Street', 'Billing Address1', 'Billing Address2',
       'Billing City', 'Billing Zip', 'Billing Province', 'Billing Country',
       'Billing Phone', 'Shipping Name', 'Shipping Street',
       'Shipping Address1', 'Shipping Address2',
       'Lineitem compare at price', 'Lineitem requires shipping',
       'Lineitem taxable','Taxes', 'Email', 'Financial Status', 'Paid at']

df.drop(todrop, axis = 1, inplace = True)

#ajout jours, mois, année
daterange = pd.to_datetime(df['Created at'])
df['Created at'] = pd.to_datetime(df['Created at'])
month = []
day = []
year = []

for date in daterange:
    month.append(date.month)
    day.append(date.day)
    year.append(date.year)
    
df["Month"] = month
df['Year'] = year
df['Day'] = day

#Nettoyage du nom des variantes 

items = df['Lineitem name'].apply(lambda x: x.split('#'))

models = items.apply(lambda x: x if len(x) >1 else [[float('NaN')], [float('NaN')]])

#models.dropna(inplace = True)

models = pd.DataFrame(models)
models['shoe'] = models['Lineitem name'].apply(lambda x: x[0])
models['variant'] = models['Lineitem name'].apply(lambda x: x[1])

models.drop('Lineitem name', axis = 1, inplace = True)

#On récupère les tailles

size = models.variant.str.split('-',expand = True)

size_more = size[1].str.split('/', expand = True).drop(2, axis = 1)

size_more[1].unique()

size_more = size_more.reset_index(drop= True)
#Faut rajouter les tailles dans la bonne colonnes et garder les Rouges et Jaunes

#On vire les tailles US et mets les tailles au bon endroit 

size_more.iloc[262:350,1] = float('NaN')

size_more.iloc[369,0] = size_more.iloc[369,1] 
size_more.iloc[369,1] = 'Rouge'
size_more.iloc[350,0] = size_more.iloc[350,1] 
size_more.iloc[350,1] = 'Rouge'
size_more.iloc[354,0] = size_more.iloc[354,1] 
size_more.iloc[354,1] = 'Rouge'

size_more.iloc[397,0] = size_more.iloc[397,1] 
size_more.iloc[397,1] = 'Rouge'
size_more.iloc[370:395,0] = size_more.iloc[370:395,1]
size_more.iloc[398:407,0] = size_more.iloc[398:407,1]
size_more.iloc[404:406,0] = float('NaN')
size_more.iloc[404:406,1] = float('NaN')

models.drop('variant', axis = 1, inplace = True)

size_more['variant'] = size[0]

size_more.columns = ['size', 'color', 'variant']

models = models.merge(size_more, left_index = True, right_index = True)
 
reorder = ['shoe', 'size', 'variant', 'color']


models = models[reorder]

for col in models.columns:
    models[col] = models[col].str.strip()

#Uniformisation de la colonne shoe

models['shoe'] = models['shoe'].str.lower()

listmodels = models['shoe'].drop_duplicates().to_list()

dictmodels = {listmodels[1]: 'air force one', listmodels[2]:'air force one', listmodels[3]: 'air force one shadow',
              listmodels[4]: "stan smith", listmodels[5]: "stan smith", listmodels[6]: "air force one"}

models['shoe'] = models['shoe'].replace(dictmodels)


#Uniformisation de la colonne variant

models = models.fillna("no")
models.variant = models.variant.str.lower()

models.size = models['size'].apply(lambda x: x.replace('FR ', ''))
models['size'] = models['size'].apply(lambda x: x[0:2]).apply(lambda x: x.strip())

listvariants = models.variant.drop_duplicates().to_list()

dictvariants = {'cartoon2.0': "cartoon 2.0", 'collabasaprocky': 'asaprocky', "comme des garcons": 'cdg', 
                'commedesgarçons' : 'cdg', 'commedesgarçons full': 'cdg full',
                'hunterxhunterv2': "hunterxhunter 2.0",
                'orangeshade': 'orange shade', 'swooshnéon': "swooshneon",
                'reflective louisvuitton': 'réflective louisvuitton',
                'hunterxhunter': 'hunterxhunter 1.0',
                'goku': 'furiousgoku'}

models.variant = models.variant.replace(dictvariants)







#Merging

df1 = df.merge(models, left_index = True, right_index = True)

#On supprime les lignes en trop
df1.drop([344,159,419], axis = 0, inplace = True)



reorder = ['Name', 'Fulfillment Status', 'Fulfilled at', 'Accepts Marketing',
       'Currency', 'Subtotal', 'Shipping', 'Total', 'Discount Code',
       'Discount Amount', 'Shipping Method', 'Created at','Day','Month', 'Year', 'Lineitem quantity',
       'Lineitem name', 'shoe', 'size', 'variant', 'color','Lineitem price', 'Lineitem fulfillment status',
       'Shipping City', 'Shipping Country', 'Refunded Amount']

df1 = df1[reorder]

needinfo = df1[df1['shoe'] == 'no']

needinfo.drop([236,331,199,376,384,393,129,132,156,0,380], inplace = True)
listorderneedinfos = needinfo['Name'].to_list()


#Uniformisation des formules

formules = df1[df1['shoe'] == 'no']
formules = formules[['Lineitem name', 'shoe', 'size', 'variant', 'color']]

size = formules['Lineitem name'].apply(lambda x : x.split('/'))

formules['size'] = size.apply(lambda x : x[1] if len(x) > 1 else None).str.strip()

formules.loc[331, 'size'] = 44
formules.loc[48, "size"] = 31
formules.loc[199, 'size'] = 42

formules.loc[199, 'variant'] = "Naruto X Sasuke"
formules.loc[236, 'size'] = 41
formules.loc[236, 'variant'] = "Monopoly"

formules.loc[376, 'shoe'] = 'Air Force 1'
formules.loc[376, 'size'] = float('NaN')
formules.loc[384, 'shoe'] = 'Air Force 1'
formules.loc[384, 'size'] = float('NaN')
formules.loc[392, 'shoe'] = 'Air Force 1'
formules.loc[392, 'size'] = float('NaN')
formules.loc[393, 'shoe'] = 'Air Force 1'
formules.loc[393, 'size'] = float('NaN')
formules.loc[380, 'shoe'] = 'no shoe'
formules.loc[380, 'size'] = float('NaN')

formules['shoe'] = 'Air Force 1'

formules['Lineitem name'] = formules['Lineitem name'].str.strip()

#Toutes les prestations seules deviennent des no shoe
sansshoes = [292,305,307,274,286,352,361,0,129,132,156,356,396,380]


formules.loc[sansshoes,'shoe'] = 'no shoe'



#on rajoute la mention custom perso pour les customs personnalisés

formules.loc[sansshoes,['variant', 'color']] = 'custom perso'

formules.loc[formules[formules['variant'] == 'no']['variant'].index, ['variant', 'color']] = 'custom perso'

#On passe les sizes en nombres
formules.size = pd.to_numeric(formules['size'])


#Moyenne des formules pour lesquelles on a la taille 
MeanSizeFormules = formules[formules['size'].notnull()].mean()

#Remplacement des colonnes de df1 par celles de formules pour les indexes concernés


cols = ['shoe', 'size', 'variant', 'color']

df1.loc[formules.index, cols] = formules.loc[formules.index, cols]

df1.drop(424, axis = 0, inplace = True)


#On change les no pour des NaN
#df1[df1['size'] == 'no'] = float('NaN')
#df1[df1['color'] == 'no'] = float('NaN')

#on remplace les no dans size
df1.loc[[405,404], 'size'] = float('NaN')
df1['size'] = pd.to_numeric(df1['size'])

MeanSizeAll = df1[df1['size'].notnull()].mean()

#Les moyennes des tailes sont les mêmes pour les formules ou en dehors des formules
#la moyenne est de 40. Fillna(40)
df1['size'] = df1['size'].fillna(40)

#On peut finalement droper la colonne lineitem

df1.drop('Lineitem name', axis = 1, inplace = True)

#uniformisation cdg rouge + on garde le jaune
df1['variant'] = df1['variant'].replace({'cdg': 'cdg rouge'})
df1.loc[60, 'variant'] = 'cdg jaune'

#Plus besoin de la colonne color

df1.drop('color', axis = 1, inplace = True)

df1['shoe'] = df1['shoe'].replace({'Air Force 1': 'air force one'})


#df1.to_csv('df_clean.csv', index = False)




#Evolution du CA 

CA_total = df.sort_values('Created at').groupby(['Month', 'Year'])['Total'].sum()
CA_total = CA_total.sort_index(level = 1)

figsize()
CA_total.plot(label= 'CA')
plt.xlabel('Date (mois, année)', fontsize = 20)
plt.ylabel('Euros',fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Évolution du CA', fontsize = 20)
plt.legend()

#Évolution des volumes par mois 

volume = df1.sort_values('Created at').groupby(['Month', 'Year'])['Lineitem quantity'].sum()
volume = CA_total.sort_index(level = 1)

figsize()
volume.plot(label = 'Unités vendues', color = 'gray')
plt.xlabel('Date (mois, année)', fontsize = 10)
plt.ylabel('Unités vendues', fontsize = 10)
plt.title('Évolution du nombre d\'unités vendues', fontsize = 10)
plt.legend()

#% fullfilled

fulfillment = df1['Fulfillment Status']

ratio_fulfilled = (fulfillment[fulfillment == 'fulfilled']).count()/ df1.shape[0]

ratio_unfulfilled = (fulfillment[fulfillment == 'unfulfilled']).count()/ df1.shape[0]

ratio_pie = [ratio_fulfilled, ratio_unfulfilled]

figsize()
colors = plt.cm.PuRd(np.linspace(0.2,0.8,5))
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.pie(ratio_pie, shadow = True, explode=(0,0.02), labels = ['Fulfilled', 'Unfulfilled'], 
        autopct='%1.1f%%', textprops={'fontsize': 30})
plt.title('% Fulfilled vs. Unfulfilled', fontsize = 30)

#Répartition des chaussures

shoes = df1['shoe']


ratio_af1 = (shoes[shoes == 'air force one'].count()) / (df1.shape[0] - shoes[shoes == 'no shoe'].count())
ratio_af1s = (shoes[shoes == 'air force one shadow'].count()) / (df1.shape[0] - shoes[shoes == 'no shoe'].count())
ratio_stan_smith = (shoes[shoes == 'stan smith'].count()) / (df1.shape[0] - shoes[shoes == 'no shoe'].count())

ratio_shoes = [ratio_af1, ratio_af1s, ratio_stan_smith]

figsize()
colors = plt.cm.PuRd(np.linspace(0.2,0.8,5))
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.pie(ratio_shoes, shadow = True, labels = ['AF1', 'AF1 Shadow', 'Stan Smith'], 
        autopct='%1.1f%%', textprops={'fontsize': 30})
plt.title('% ventes par modèles',fontsize = 30)
plt.text(1,1,'AF1: 390 unités\nAF1 Shadow: 9 unités\nStan Smith: 8 unités',
         bbox=dict(boxstyle ='round', alpha=0.2, color = 'grey', lw = 5), fontsize = 15)
plt.tight_layout()


#Répartition des tailles 

df1[df1['size'] == 41]['size'].count()


sizes = df1['size']

sizes_unique = sizes.sort_values().unique()
sizes_list = list((sizes_unique))


sizes = pd.DataFrame(sizes_unique, columns = ['size'])

size['total'] = 0

total_sizes = []

for size in sizes_list:
    total_sizes.append(int(df1[df1['size'] == size]['size'].count()))
    
    
figsize()   

sns.barplot(x = sizes_unique, y = np.array(total_sizes))
plt.title('Répartition des ventes par tailles',fontsize = 30)
plt.xlabel('Tailles',fontsize = 30)
plt.ylabel('Unités vendues',fontsize = 30)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15, rotation = 90)
plt.ylim([0, 100])



#Répartition des ventes par variantes 

variants = df1['variant']

variants_unique = variants.unique()

variants_list = list(variants_unique)

variants = pd.DataFrame(variants_unique, columns = ['variants'])

variants['total'] = 0 

total_variants = []

for variant in variants_list:
    total_variants.append(df1[df1['variant'] == variant]['variant'].count())
    
variants['total'] = total_variants
variants = variants.set_index('variants')

top10names = list(variants['total'].nlargest(10).index)


top10 = np.array(variants['total'].nlargest(10))



figsize()
colors = plt.cm.PuRd(np.linspace(0.2,0.8,5))
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
sns.barplot(x= top10names, y = top10)
plt.title('Répartition des ventes par modèles (top 10)',fontsize = 30)
plt.xlabel('Modèles',fontsize = 30)
plt.xticks(fontsize = 20, rotation = 35)
plt.yticks(fontsize = 15)
plt.ylabel('Unités vendues',fontsize = 30)


#Top 10 des ventes + les chiffres affichés clairement

df10top = pd.DataFrame()
df10top['variants'] = top10names
df10top['unités'] = top10

#Répartition des ventes de variantes par taille 

VarSizeTop10 = pd.DataFrame(columns = sizes_list, index = df10top['variants'])

#Regrouper les sommes de variants par taille

for var in top10names:
    x = df1[df1['variant'] == var]['size']
    for s in sizes_unique:
        VarSizeTop10.loc[var, s] = x[x == s].count()

#Prendre les top10 des sizes pour chaque variant et les visualiser


VarSizeTop10 = VarSizeTop10.T
VarSizeTop10[VarSizeTop10.columns] = VarSizeTop10[VarSizeTop10.columns].apply(pd.to_numeric)


for col in VarSizeTop10.columns:
    x = VarSizeTop10[col].nlargest(5)
    total = VarSizeTop10[col].sum()

    
    figsize()
    colors = plt.cm.PuRd(np.linspace(0.2,0.8,5))
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    plt.pie(x = x, labels= x.index, autopct='%1.1f%%', 
                                             textprops={'fontsize': 15, 'weight': 'bold'})
    plt.title('Répartition de la variante \'{!s}\' par taille'.format(col), fontsize = 15)

    





#Moyenne des ventes par mois : les modèles et les tailles

MeanSalesMonth = df1.groupby(['variant'])['Lineitem quantity'].sum() / 8

#On garde le top 10
Top10MeanSales = MeanSalesMonth.nlargest(10)

VarSizeTop10T = VarSizeTop10.T
dict ={}
for index in VarSizeTop10T.index:
    x =  pd.DataFrame(index = [index], columns = VarSizeTop10T.columns)
      
    x.loc[index]= (VarSizeTop10T.loc[index] / 8)

    dict.update({index : x.loc[index].astype(float).nlargest(10)})


#Ventes de chaussures par taille puis top ou top 10 + moyenne par mois 



SaleSizes = pd.DataFrame(index = df1['shoe'].unique(), columns = VarSizeTop10.index ).drop('no shoe', axis = 0)

listshoes = ['air force one', 'air force one shadow', 'stan smith']

#Count des ventes de chaussures par tailles
for s in listshoes:
    x = df1[df1['shoe'] == s][['size']]
    for Size in sizes_list:
        size_count = x[x['size'] == Size].count()[0]
        SaleSizes.loc[s, Size] = size_count.astype(int)
        
SaleSizesT = SaleSizes.T
SaleSizesT[SaleSizesT.columns] = SaleSizesT[SaleSizesT.columns].apply(pd.to_numeric) 

#AF1
af1_sizes = pd.DataFrame(columns = ['count', 'mean'])
af1_sizes['count'] = SaleSizesT['air force one']
af1_sizes['mean'] = (af1_sizes / 8).astype(float)

af1_top = af1_sizes['mean'].nlargest(15)

#AF1 Shadow
af1s_sizes = pd.DataFrame(columns = ['count', 'mean'])
af1s_sizes['count'] = SaleSizesT['air force one shadow']
af1s_sizes['mean'] = (af1s_sizes / 8).astype(float)

af1s_top = af1s_sizes['mean'].nlargest(10)

#Stan Smith
stans_sizes = pd.DataFrame(columns = ['count', 'mean'])
stans_sizes['count'] = SaleSizesT['stan smith']
stans_sizes['mean'] = (stans_sizes / 8).astype(float)

stans_top = stans_sizes['mean'].nlargest(10)

# La moyenne des tailles des paires vendues 
df1.groupby('shoe')['size'].mean()


#% de ventes sur lyon et ses alentours

df['Shipping City'] = df['Shipping City'].str.lower() 
cities = ['lyon', 'villeurbanne', 'caluire et cuire', 'vénissieux']
numbers =[]
for city in cities:
    x = df[df['Shipping City'] == city]['Lineitem quantity'].sum()
    numbers.append(x)

total_cities = sum(numbers)

figsize()
colors = plt.cm.PuRd(np.linspace(0.2,0.8,5))
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.pie(x = (total_cities, df1.shape[0]), labels= ['Lyon et alentours', 'Reste de la France'], autopct='%1.1f%%', 
                                         textprops={'fontsize': 20, 'weight': 'bold'})
plt.title('% des livraisons à Lyon'.format(col), fontsize = 30)
