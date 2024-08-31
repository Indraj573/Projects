#!/usr/bin/env python
# coding: utf-8

# # Obective :-

#  Predict whether a customer will change telecommunications provider or not

# ## Import Library

# In[1]:


import pandas as pd
import numpy as np

# Graph modules.
import matplotlib.pyplot as plt
from scipy.stats import norm

# Data encoding module
from sklearn.preprocessing import OneHotEncoder

# Classifier modules.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ## Loading dataset as df

# In[2]:


df = pd.read_csv(r"A:\Dataset\Coustomer church\train.csv")
df.head()


# ## Data Identification

# In[3]:


df.keys()


# In[4]:


# Pie chart function
def plot_pie_chart(column, title):
    count = df[column].value_counts()
    # Define custom colors for the pie chart
    custom_colors = ['#70a288', '#d5896f', '#33B168']
    # Calculate the percentages
    percentages = count / count.sum() * 100

    # Plotting the pie chart
    plt.figure(figsize=(4, 4))
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=90, colors=custom_colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(title)
    plt.show()


# In[5]:


# The function above requires the following input format: (the name of the class column, the title of the pie chart).
plot_pie_chart("churn", "Percentage Distribution of Class data")


# In[6]:


# View information about the variable column used in the training dataset, such as the variable's data type.
df.info()


# ## Data Preprocessing

# In[7]:


# Get the features with categorical data type.
categorical = list(df.select_dtypes(['object']).columns)
categorical


# In[8]:


# Get the features with numerical data type.
numerical = list(df.select_dtypes(['float64','int64']).columns)
numerical


# In[9]:


# View how many null values are in each feature of the training data.
df.isnull().sum()


# In[10]:


# Function of z-score visualisation.
def zscore_visualization(df, column, zscores, outliers):
    # Plot the z-scores with detailed bell-shaped curve.
    plt.figure(figsize=(8, 4))

    # Plot the histogram of z-scores.
    plt.hist(zscores, bins=30, density=True, alpha=0.6, color='#70a288', label='Z-Scores')

    # Plot the bell-shaped curve.
    x = np.linspace(zscores.min(), zscores.max(), 100)
    plt.plot(x, norm.pdf(x), 'r-', label='Bell Curve')

    # Add threshold lines for outliers.
    plt.axvline(-3, color='r', linestyle='--', label='Threshold')
    plt.axvline(3, color='r', linestyle='--')

    # Highlight individual outliers.
    plt.scatter(outliers, np.zeros_like(outliers), color='red', label='Individual Outliers')

    plt.xlabel('Z-Score')
    plt.ylabel('Density')
    plt.title(f'Z-Score Distribution of {column} with Bell-Shaped Curve')

    # Display the count of outliers.
    plt.text(0.02, 0.9, f'Outliers: {len(outliers)} Data', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='square'))

    plt.legend()
    plt.grid(True)
    plt.show()


# In[11]:


# Copy  data into new variable.
df_copy = df.copy()


# ### account_length

# In[12]:


# FORMULA: Z-score = (observed value - population mean)/standard deviation.
df_copy["zscore"] = (df_copy["account_length"] - df_copy["account_length"].mean())/ df_copy["account_length"].std()
df_copy.head()


# In[13]:


# Retrieve data with a z-score value that is less than -3 or greater than 3.
account_length_outliers = df_copy[(df_copy.zscore<-3) | (df_copy.zscore>3)]
account_length_outliers


# In[14]:


# z-score distribution of account_length feature.
zscore_visualization(df_copy,"account_length", df_copy['zscore'], account_length_outliers['zscore'])


# ###  number_vmail_messages

# In[15]:


# FORMULA: Z-score = (observed value - population mean)/standard deviation.
df_copy["zscore"] = (df_copy["number_vmail_messages"] - df_copy["number_vmail_messages"].mean())/ df_copy["number_vmail_messages"].std()
df_copy.head()


# In[16]:


# Retrieve data with a z-score value that is less than -3 or greater than 3.
number_vmail_messages_outliers = df_copy[(df_copy.zscore<-3) | (df_copy.zscore>3)]
number_vmail_messages_outliers


# In[17]:


# z-score distribution of number_vmail_messages column.
zscore_visualization(df_copy,"number_vmail_messages", df_copy['zscore'], number_vmail_messages_outliers['zscore'])


# #### NOTES ON HANDLING DATA OUTLIER 
# In this case, the outlier data is retained
# When dealing with outliers, the first step is to check if they are contaminated data. Check the accuracy of the data and calculations. If the outliers are indeed contaminated, exclude them from the study. If possible, replace the contaminated data with the correct values. Outliers may sometimes represent valid and uncontaminated data, but they have extreme values compared to the majority of the data points in the group. There are two approaches to handling outliers: eliminate them or preserve them. The decision to retain or discard the outliers depends on the specific context and goals of the study.

# ### Featur encoding
# Change the categorical data to numerical data
# 
# The feature encoding method used:
# 
# * state: Frequency Encoding
# * area_code: One-hot Encoding
# * international_plan: Label Encoding
# * voice_mail_plan: Label Encoding

# #### Encode stat column using Frequency encoding

# In[18]:


frq_dis = df.groupby('state').size()/len(df)
frq_dis.head()


# In[19]:


df["state"] = df.state.map(frq_dis)
df.head()


# ####  Encode area_code column using One-hot encoding

# In[20]:


# Call the One-hot encoding module
enc_one = OneHotEncoder()


# In[21]:


x = df["area_code"].values.reshape(-1,1)
val = enc_one.fit_transform(x).toarray()


# In[22]:


# Lets assign the column name to each one hot vector
df_onehot = pd.DataFrame(val,columns=['is_'+str(enc_one.categories_[0][i]) for i in range(len(enc_one.categories_[0]))])
df_onehot.head()


# In[23]:


df_c = df.copy()


# In[24]:


df_enc = pd.concat([df_onehot, df_c],axis=1)
df_enc.head()


# In[25]:


#droping the area_code column
df_enc.drop("area_code",inplace=True,axis=1)
df_enc.head()


# #### Encode international_plan column using Lable encoding

# In[26]:


international_plan = {"no" : 0, "yes" : 1}
df_enc['international_plan'] = df_enc['international_plan'].map(international_plan)


# In[27]:


df_enc['international_plan'].value_counts()


# ####  Encode voice_mail_plan column using Lable encoding

# In[28]:


voice_mail_plan = {"no" : 0, "yes" : 1}
df_enc['voice_mail_plan'] = df_enc['voice_mail_plan'].map(voice_mail_plan)


# In[29]:


df_enc['voice_mail_plan'].value_counts()


# In[30]:


df_enc.head()


# #### Normalisation using MinMax Normalisation

# In[31]:


# The minimum value in each feature
numeric_min = df[numerical].min().to_dict()
numeric_min


# In[32]:


# The maximum value in each feature
numeric_max = df[numerical].max().to_dict()
numeric_max


# In[33]:


# Function of data normalisation
def normalize(df, decimal_places=2):
    for key in numeric_min.keys():
        df[key] = round((df[key] - numeric_min[key])/ (numeric_max[key]-numeric_min[key]),decimal_places)
    return df


# In[34]:


df_norml = normalize(df_enc, decimal_places=3)
df_norml.head()


# ## Modelling

# In[48]:


# independent and dependent variables separation
data_churn = df_norml.drop('churn',axis = 1 )
target = df_norml['churn']


# In[49]:


# Show training data without dependent attribute (class data)
data_churn.head()


# In[50]:


# Show dependent attribute (class data) of training data
target.head()


# In[51]:


# Assuming df_churn is your DataFrame with features and target is your Series with the target labels
tt = target
# Standardize the features
scaler = StandardScaler()
df_churn_scaled = scaler.fit_transform(data_churn)

# Perform PCA to reduce to 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_churn_scaled)

# Create a DataFrame with the principal components
df_pca = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = tt

# Plot the PCA result
plt.figure(figsize=(10, 6))
targets = df_pca['Target'].unique()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can define more colors if you have more classes

for targe, color in zip(targets, colors):
    indicesToKeep = df_pca['Target'] == targe
    plt.scatter(df_pca.loc[indicesToKeep, 'Principal Component 1'],
                df_pca.loc[indicesToKeep, 'Principal Component 2'],
                c=color,
                s=50)
plt.title('PCA of Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(targets)
plt.grid()
plt.show()


# ### Naive Bayes model performance with 10 Fold Cross Validation

# In[52]:


k_fold = KFold(n_splits=10 , shuffle=True, random_state=0)

clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, data_churn, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
mean_score_nb = score.mean()
print(f"The mean score of cross validation using Naive Bayes (NB) algorithm is {mean_score_nb * 100:.2f}%")


# ### Decision Tree model performance with 10 Fold Cross Validation

# In[53]:


k_fold = KFold(n_splits=10 , shuffle=True, random_state=0)

clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, data_churn, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
mean_score_dt = score.mean()
print(f"The mean score of cross validation using Decision tree (DT) algorithm is {mean_score_dt * 100:.2f}%")


# ### K-Nearest Neighbor model performance with 10 Fold Cross Validation

# In[54]:


# Convert them to NumPy arrays
data_churn_np = np.array(data_churn)
target_np = np.array(target)

# Ensure the data is in C-contiguous order
data_churn_np = np.ascontiguousarray(data_churn_np)
target_np = np.ascontiguousarray(target_np)

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, data_churn_np, target_np, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
mean_score_knn = score.mean()
print(f"The mean score of cross validation using K-Nearest Neighbor algorithm (KNN) is {mean_score_knn * 100:.2f}%")


# ### Support Vector Machine model performance with 10 Fold Cross Validation

# In[55]:


k_fold = KFold(n_splits=10 , shuffle=True, random_state=0)

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, data_churn, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
mean_score_svm = score.mean()
print(f"The mean score of cross validation using Support Vector Machine (SVM) algorithm is {mean_score_svm * 100:.2f}%")


# In[56]:


# Add the mean score to an array.
result_modelling = [{mean_score_nb*100},{mean_score_dt*100},{mean_score_knn*100},{mean_score_svm*100}]
# Add the model name to an array.
model = ["NB","DT","KNN","SVM"]

# Convert result_modeling and model into a DataFrame using Pandas.
result_modelling = pd.DataFrame(result_modelling,columns=["Results"])
model = pd.DataFrame(model,columns=["Model"])


# In[57]:


# Merge the model table with the result_modeling table.
modelling = pd.concat([model,result_modelling],axis=1)
modelling


# In[58]:


# Visualize model performance in a bar chart.
model = modelling.Model
results = modelling.Results
fig = plt.figure(figsize = (8,  4))
plt.bar(model, results, color = "royalblue", width = 0.7)
plt.xlabel('Model')
plt.ylabel('Results')
plt.title('Performance of modelling')
plt.show()


# ##  The Decision Tree Classifier demonstrates higher accuracy than Naive Bayes, K-Nearest Neighbor, and Support Vector Machine based on cross-validation results with k=10. The obtained accuracy is 92.4%.

# In[ ]:





# In[ ]:




