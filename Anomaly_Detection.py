#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection Techniques in Python

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from IPython.display import HTML 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from matplotlib import cm
from sklearn.ensemble import IsolationForest
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
import plotly as plty
import plotly.graph_objs as go


# ## Loading Dataset

# In[2]:


def fileRead(directory_path):
    file_name = list()
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_name.append(filename)
    return file_name


    
    
directoryPath1 = "/home/mishraanurag218/Anurag/Projects/Untitled Folder/data/s1/"
directoryPath2 = "/home/mishraanurag218/Anurag/Projects/Untitled Folder/data/s2/"
s1 = fileRead(directoryPath1)
#print(s1)
s2 = fileRead(directoryPath2)
#print(s2)


# In[3]:


cols = ['time','acc_frontal','acc_vertical','acc_lateral','id','rssi','phase','frequency','activity']
def folder_to_csv(directory_path,file_name,col_name):
    df_temp = pd.DataFrame()
    for f_n in file_name:
        df = pd.read_csv(directory_path+f_n,names=col_name)
        df['device_id'] = f_n[0:-1]
        df['sex'] = f_n[-1]
        df_temp = pd.concat([df_temp, df], ignore_index=True)
    return df_temp

df_s1 = folder_to_csv(directoryPath1,s1,cols)
df_s2 = folder_to_csv(directoryPath2,s2,cols)

df = pd.concat([df_s1, df_s2], ignore_index=True)

df.head(5)


# ### Data Pre-processing`

# ### Changing the sex into binary

# In[4]:


def categorical_to_binary(x):
    if x=='F':
        return 0
    else:
        return 1

df['sex_b'] = df['sex'].apply(categorical_to_binary)
df.head(5)


# In[5]:


df


# ### removing non numerical attributes and categorical values and ids as machine learning algorithms only works on numerical values

# In[6]:


dfX = df.copy().drop(['sex','device_id','id','sex_b','time'],axis=1)
dfY = df['sex_b'].copy()
dfX.head()


# ### Making the dataset to standard scale

# In[7]:


scaler = MinMaxScaler() 
data = scaler.fit_transform(dfX)
dfX = pd.DataFrame(data, columns = dfX.columns)
dfX.head(5)


# ## Univariate Analysis on dataset

# #### Function for Histogram plot

# In[8]:


def hist_plot(x,y):
    for i in y:
        sns.distplot(x[i],bins=150)
        plt.show()
    


# In[9]:


cols = ['time','acc_frontal','acc_vertical','acc_lateral','rssi','phase']
hist_plot(df,cols)


# #### Function for joint plot

# In[10]:


def joint_plot(x,y,z):
    for i in y:
        sns.jointplot(x=i, y=z, data=x);
        plt.show()


# In[11]:


cols = ['acc_frontal','acc_vertical','acc_lateral','rssi','phase']
joint_plot(df,cols,'time')


# #### Pair Plot

# In[12]:


sns.set_style("whitegrid");
sns.pairplot(df, hue="sex_b", height=3);
plt.show()


# #### implot

# In[13]:


def implot(x,y,z):
    for i in y:
        for j in y:
            if i!=j:
                sns.lmplot(x = i, y = j, data = x, hue = z, col = z)
                plt.show()
                
implot(df,['acc_frontal','acc_vertical','acc_lateral','rssi','phase'],'sex_b')


# In[14]:


implot(df,['acc_frontal','acc_vertical','acc_lateral','rssi','phase'],'activity')


# #### countplot

# In[15]:


def countPlot(x,y):
    for i in y:
        sns.countplot(x =i, data = x)
        plt.show()
    sns.countplot(x =y[0], hue = y[1], data = x)
    plt.show()
        
countPlot(df,['sex_b','activity'])


# #### Balancing the data

# In[16]:


sm = SMOTE(random_state = 2) 
df_X, df_Y = sm.fit_sample(dfX, dfY.ravel())
df_Y = pd.DataFrame(df_Y, columns = ['sex_b'])


# In[17]:


sns.countplot(x ='sex_b', data = df_Y)


# #### Lazy predict classification

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y,test_size=.33,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# ## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

# #### This is a clustering algorithm (an alternative to K-Means) that clusters points together and identifies any points not belonging to a cluster as outliers. It’s like K-means, except the number of clusters does not need to be specified in advance.
# #### The method, step-by-step:
# #### Randomly select a point not already assigned to a cluster or designated as an outlier. Determine if it’s a core point by seeing if there are at least min_samples points around it within epsilon distance.
# #### Create a cluster of this core point and all points within epsilon distance of it (all directly reachable points).
# #### Find all points that are within epsilon distance of each point in the cluster and add them to the cluster. Find all points that are within epsilon distance of all newly added points and add these to the cluster. Rinse and repeat. (i.e. perform “neighborhood jumps” to find all density-reachable points and add them to the cluster).

# ### Sklearn Implementation of DBSCAN:

# In[19]:


outlier_detection = DBSCAN(
 eps = .2, 
 metric='euclidean', 
 min_samples = 5,
 n_jobs = -1)
clusters = outlier_detection.fit_predict(dfX)
cmap = cm.get_cmap('Set1')


# #### DBSCAN will output an array of -1’s and 0’s, where -1 indicates an outlier. Below, I visualize outputted outliers in red by plotting two variables.

# In[20]:


df.plot.scatter(x='time',y='acc_vertical', c=clusters, cmap=cmap,
 colorbar = False)
plt.show()


# In[21]:


# fig = go.Figure(data=go.Scatter(x=df['time'],
#                                 y=df['acc_vertical'],
#                                 mode='markers',
#                                 marker_color=clusters,
#                                 text=clusters)) # hover text goes here

# fig.update_layout(title='Scatter Plot to identify the outliers')
# fig.show()


# In[22]:


import plotly.express as px
fig = px.scatter(df, x="time", y="acc_vertical", color=clusters,
                 hover_data=['time'])
fig.show()
fig = px.scatter(df[clusters==-1], x="time", y="acc_vertical",
                 hover_data=['time'])
fig.show()


# In[23]:


outliers = np.where(clusters==-1)
df_X_db = df_X.drop(list(outliers[0])) 
df_Y_db = df_Y.drop(list(outliers[0]))
df_dbScan = result = pd.concat([df_X_db,df_Y_db], axis=1, sort=False)
df_dbScan.to_csv (r'Filtered_DBSCAN.csv', index = False, header=True)
print(df_dbScan.head())


# In[24]:


sns.countplot(x ='sex_b', data = df_dbScan)

fig = px.histogram(df_dbScan, x="sex_b", color="sex_b")
fig.update_layout(barmode='group')
fig.show()


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df_X_db, df_Y_db,test_size=.33,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


# ## Isolation Forests

# #### Randomly select a feature and randomly select a value for that feature within its range.
# #### If the observation’s feature value falls above (below) the selected value, then this value becomes the new min (max) of that feature’s range.
# #### Check if at least one other observation has values in the range of each feature in the dataset, where some ranges were altered via step 2. If no, then the observation is isolated.
# #### Repeat steps 1–3 until the observation is isolated. The number of times you had to go through these steps is the isolation number. The lower the number, the more anomalous the observation is.

# ## Sklearn Implementation of Isolation Forests:

# In[26]:


rs=np.random.RandomState(0)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1) 
clf.fit(dfX)
if_scores = clf.decision_function(dfX)
if_anomalies=clf.predict(dfX)
if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])


# In[27]:


fig = px.scatter(dfX, x="phase", y="acc_vertical", color=if_anomalies,
                 hover_data=['phase'])
fig.show()
fig = px.scatter(dfX[if_anomalies==1], x="phase", y="acc_vertical",
                 hover_data=['phase'])
fig.show()


# In[28]:


df_X_if = dfX[if_anomalies!=1]
df_Y_if = dfY[if_anomalies!=1]
df_lf = pd.concat([df_X_if,df_Y_if], axis=1, sort=False)
df_lf.to_csv (r'Filtered_lf.csv', index = False, header=True)
print(df_lf.head())


# In[29]:


sns.countplot(x ='sex_b', data = df_lf)
fig = px.histogram(df_lf, x="sex_b", color="sex_b")
fig.update_layout(barmode='group')
fig.show()


# In[30]:


import h2o
from h2o.automl import H2OAutoML

h2o.init()
df_train, df_test = train_test_split(df_lf, test_size=0.26)
df_train.to_csv (r'train_lf.csv', index = False, header=True)
df_test.to_csv (r'test_lf.csv', index = False, header=True)
# Import a sample binary outcome train/test set into H2O

train = h2o.import_file("train_lf.csv")
test = h2o.import_file("test_lf.csv")

# Identify predictors and response
x = train.columns
y = "sex_b"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  


# ## Local Outlier Factor

# In[31]:


plt.figure(figsize=(12,8))
plt.hist(if_scores);
plt.title('Histogram of Avg Anomaly Scores: Lower => More Anomalous');


# #### LOF uses density-based outlier detection to identify local outliers, points that are outliers with respect to their local neighborhood, rather than with respect to the global data distribution. The higher the LOF value for an observation, the more anomalous the observation.
# #### This is useful because not all methods will not identify a point that’s an outlier relative to a nearby cluster of points (a local outlier) if that whole region is not an outlying region in the global space of data points.
# #### A point is labeled as an outlier if the density around that point is significantly different from the density around its neighbors.

# In[32]:


clf = LocalOutlierFactor(n_neighbors=30, contamination=.1)
y_pred = clf.fit_predict(df_X)
LOF_Scores = clf.negative_outlier_factor_
LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies=df[LOF_pred==1]


# In[33]:


LOF_anomalies


# In[34]:


cmap=np.array(['white','red'])
plt.scatter(dfX.iloc[:,0],dfX.iloc[:,2],c='white',s=20,edgecolor='k')
plt.scatter(LOF_anomalies.iloc[:,1],LOF_anomalies.iloc[:,2],c='red')
 #,marker=’x’,s=100)
plt.title('Local Outlier Factor — Anomalies')
plt.xlabel('time')
plt.ylabel('acc_vertical')


# In[35]:


df_X


# In[36]:



fig = px.scatter(df_X, x="phase", y="acc_vertical", color=LOF_pred,
                 hover_data=['phase'])
fig.show()


# In[37]:


df_X_lof = df_X[LOF_pred!=1]
df_Y_lof = df_Y[LOF_pred!=1]
df_out =  pd.concat([df_X[LOF_pred==1],df_Y[LOF_pred==1]], axis=1, sort=False)
fig = px.scatter(df_out, x="phase", y="acc_vertical",
                 hover_data=['phase'])
fig.show()


# In[38]:


df_lof = pd.concat([df_X_lof,df_Y_lof], axis=1, sort=False)
df_lof.to_csv (r'Filtered_lof.csv', index = False, header=True)
print(df_lof.head())


# In[39]:


sns.countplot(x ='sex_b', data = df_lof)
fig = px.histogram(df_lof, x="sex_b", color="sex_b")
fig.update_layout(barmode='group')
fig.show()


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(df_X_lof, df_Y_lof,test_size=.33,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models

