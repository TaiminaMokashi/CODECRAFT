#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[8]:


df = pd.read_csv('train.csv') 


# In[9]:


df.shape


# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


df.info()


# In[ ]:





# In[13]:


df.nunique()


# In[17]:


df.columns


# In[18]:


new_df = df[['GrLivArea', 'SalePrice']]
new_df


# In[19]:


new_df.plot(x='GrLivArea', y='SalePrice', kind='scatter')


# In[21]:


df.plot(x='OverallQual', y='SalePrice', kind='scatter')


# In[22]:


x = new_df['GrLivArea']
y = new_df['SalePrice']


# In[23]:


df.plot(x='YearBuilt', y='SalePrice', kind='scatter')


# In[24]:


df.plot(x='GarageArea', y='SalePrice', kind='scatter')


# In[25]:


df.plot(x='TotalBsmtSF', y='SalePrice', kind='scatter')


# In[ ]:


from sklearn


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
data = {
    'SquareFootage': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'Bathrooms': [2, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

# Print the coefficients of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# In[ ]:





# In[ ]:




