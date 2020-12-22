#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # READING DATA

# In[2]:


data=pd.read_excel("Data_train.xlsx")
data.head()


# In[3]:


data.describe()


# In[4]:


data.info()


# # DATA WRANGLING

# In[5]:


data.isnull().sum()


# In[6]:


sns.heatmap(data.isnull())


# # DEALING CATEGORICAL VARIABLE

# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


data['Restaurant']=LabelEncoder().fit_transform(data['Restaurant'])
data['Location']=LabelEncoder().fit_transform(data['Location'])
data['Cuisines']=LabelEncoder().fit_transform(data['Cuisines'])
data['Average_Cost']=pd.to_numeric(data['Average_Cost'].str.replace('[^0-9]',''))
data['Minimum_Order']=pd.to_numeric(data['Minimum_Order'].str.replace('[^0-9]',''))
data['Rating']=pd.to_numeric(data['Rating'].apply(lambda x: np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))
data['Votes']=pd.to_numeric(data['Votes'].apply(lambda x: np.nan if x=='-' else x))
data['Reviews']=pd.to_numeric(data['Reviews'].apply(lambda x: np.nan if x=='-' else x))
data['Delivery_Time']=pd.to_numeric(data['Delivery_Time'].str.replace('[^0-9]',''))


# In[9]:


data.isnull().sum()


# # DEALING WITH MISSING VALUE 

# In[10]:


data['Rating']=data['Rating'].fillna(data['Rating'].median())
data['Votes']=data['Votes'].fillna(data['Votes'].median())
data['Reviews']=data['Reviews'].fillna(data['Reviews'].median())
data['Average_Cost']=data['Average_Cost'].fillna(data['Average_Cost'].median())
data.head()


# In[11]:


cor = data.corr()

mask = np.zeros_like(cor)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,10))

with sns.axes_style("white"):
    sns.heatmap(cor,annot=True,linewidth=2,
                mask = mask,cmap="magma")
plt.title("Correlation between variables")
plt.show()


# In[12]:


plt.figure(figsize=(12,8))
sns.heatmap(round(data.describe()[1:].transpose(),2),linewidth=2,annot=True,fmt="f")
plt.xticks(fontsize=20)
plt.yticks(fontsize=12)
plt.title("Variables summary")
plt.show()


# # TRAIN AND TEST DATA

# In[13]:


y_train=data.Delivery_Time
predictors_col=['Restaurant','Location','Average_Cost','Minimum_Order','Rating','Votes','Reviews']
X_train=data[predictors_col]


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


# In[15]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)


# In[16]:


y_pred = regressor.predict(X_test)


# In[17]:


regressor.score(X_train,y_train)*100


# In[18]:


print(y_pred)


# # TEST DATA 

# In[19]:


data1=pd.read_excel("Data_test.xlsx")
data2=data1
data2.head()


# In[20]:


# data preprocessing
from sklearn.preprocessing import LabelEncoder
data2['Restaurant']=LabelEncoder().fit_transform(data2['Restaurant'])
data2['Location']=LabelEncoder().fit_transform(data2['Location'])
data2['Cuisines']=LabelEncoder().fit_transform(data2['Cuisines'])
data2['Average_Cost']=pd.to_numeric(data2['Average_Cost'].str.replace('[^0-9]',''))
data2['Minimum_Order']=pd.to_numeric(data2['Minimum_Order'].str.replace('[^0-9]',''))
data2['Rating']=pd.to_numeric(data2['Rating'].apply(lambda x: np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))
data2['Votes']=pd.to_numeric(data2['Votes'].apply(lambda x: np.nan if x=='-' else x))
data2['Reviews']=pd.to_numeric(data2['Reviews'].apply(lambda x: np.nan if x=='-' else x))


# In[21]:


# fill missing value with meadian imputation
data2['Rating']=data2['Rating'].fillna(data2['Rating'].median())
data2['Votes']=data2['Votes'].fillna(data2['Votes'].median())
data2['Reviews']=data2['Reviews'].fillna(data2['Reviews'].median())
data2['Average_Cost']=data2['Average_Cost'].fillna(data2['Average_Cost'].median())
data2.head()


# In[22]:


X_test=data1[predictors_col]
predictions=regressor.predict(X_test)


# In[23]:


print(predictions)


# In[24]:


out=predictions.astype(int)
out=out.astype(int).astype(str)
for i in range(len(out)):
    out[i]=out[i]+"minutes"
y1=data1["Restaurant"].astype(int).astype(str)
for j in range(len(y1)):
    y1[j]="ID_"+y1[j]


# In[25]:


submission=(pd.DataFrame({'Restaurant':y1,'Delivery_Time':out}))


# In[26]:


submission.to_csv('submission.csv',index=False)

