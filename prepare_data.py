# %%
import numpy as np
import pandas as pd
from pyECLAT import ECLAT
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# %%
def get_cluster_representation(df, cluster_col, cluster, min_combination, max_combination):
    temp = df.loc[df[cluster_col] == cluster].copy()
    del temp[cluster_col]

    for c in temp.columns:
        temp[c] = temp.apply(lambda r: f"{c}={r[c]}", axis=1)

    temp.columns = np.arange(len(temp.columns))

    temp = temp.sample(frac=0.10)
    
    temp.reset_index(drop=True, inplace=True)

    eclat_instance = ECLAT(data=temp, verbose=True)

    _, supports = eclat_instance.fit(min_support=0.01,
                                     min_combination=min_combination,
                                     max_combination=max_combination,
                                     separator='&',
                                     verbose=True)

    ls = []
    for key in supports.keys():
        d = dict()
        texts = key.split('&')
        for text in texts:
            k, v = text.split('=')
            d[k] = v
        d['support'] = supports[key]
        ls.append(d)
    temp = pd.DataFrame(ls)

    temp = temp.sort_values('support', ascending=False)
    return temp

# %%
df = pd.read_csv('data/raw.csv')
df

# %%
df.columns

# %%
df.head(3).transpose()

# %%
df.describe(include='all')

# %%
df.State.value_counts()

# %%
df['Country'] = df.State.replace({'California': 'India', 
                                  'Oregon': 'Pakistan', 
                                  'Arizona': 'Philipines', 
                                  'Nevada': 'China', 
                                  'Washington': 'Other'}).tolist()
df['Country'] = df.apply(lambda r: 'UAE' if r['Coverage'] == 'Premium' else r['Country'], axis=1)

# %%
df['Location Code'].value_counts()

# %%
df['Emirate'] = df['Location Code'].replace({'Suburban': 'Dubai', 'Rural': 'Abu Dhabi', 'Urban': 'Sharjah'}).tolist()

# %%
df['EmploymentStatus'].value_counts()

# %%
df['Occupation'] = df['EmploymentStatus'].replace({'Unemployed': 'Self-Employed', 
                                                   'Medical Leave': 'Enterpreneur', 
                                                   'Disabled': 'Government',
                                                   }).tolist()

# %%
df = df.set_index('Customer')

# %%
df = df[['Country', 'Emirate', 'Education', 'Occupation', 'Gender',
             'Marital Status', 'Coverage', 'Income']].copy()

# %%
categoricals = pd.get_dummies(df.select_dtypes(include='O'), prefix_sep='=')
categoricals

# %%
numericals = df.select_dtypes(exclude='O')
cols = numericals.columns
idxs = numericals.index
mms = MinMaxScaler()
numericals = pd.DataFrame(mms.fit_transform(numericals), columns=cols, index=idxs)
numericals

# %%
train = pd.merge(categoricals, numericals, left_index=True, right_index=True, how='inner')
train

# %%
ls = []
for c in np.arange(2, 10):
    km = KMeans(n_clusters=c, n_init='auto')
    km.fit_predict(train)
    ls.append({'n_clusters': c, 'inertia': km.inertia_})
scores = pd.DataFrame(ls)
scores = scores.sort_values('inertia')
scores.set_index('n_clusters', inplace=True)
scores.plot();

# %%
km = KMeans(n_clusters=3, n_init='auto')
train['cluster'] = km.fit_predict(train)
train

# %%
df['cluster'] = train['cluster']
df.reset_index(inplace=True)
del df['Customer']
df.to_csv('data/clustered.csv', index=False)

# %%
nice_cols = ['Country', 'Emirate', 'Education', 'Occupation', 'Gender',
             'Marital Status', 'Coverage', 'Income', 'cluster']

# %%
for c in df.cluster.unique():
    temp = get_cluster_representation(df[nice_cols], 'cluster', c, 5, 5)
    temp.to_csv(f'data/cluster_{c}.csv', index=False)