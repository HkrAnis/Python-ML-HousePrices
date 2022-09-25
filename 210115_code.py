#!/usr/bin/env python
# coding: utf-8

# ## 1- Dowload the data 

# #### import the important packages

# In[2]:


import pandas as pd 


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import numpy as np


# In[5]:


import sklearn.model_selection as skl


# In[6]:


import seaborn as sns


# #### create a dataframe 

# In[7]:


housing= pd.read_csv("/Users/AA/Desktop/housing.csv")


# ## 2- Take a quick look at the data structure 

# #### Look at the top rows of the dataset to have a better understanding 
# - Each row represents one district
# - There are 10 columns/attributes
# - The top 5 rows of ocean_proximity are repetitive which probably means that it is a categorical attribute and we can use the value_counts() attribute to understand it more.

# In[8]:


housing.head()


# #### Use the info function to have a description of the total number of rows, the presence of null values and the type of each column
# - We can see that there are 20,640 rows in the data set
# - The number total_bedrooms has 20,433 values (207 missing)
# - All attributes are numerical except for the ocean_proximity which is an object
#      - This means that it can hold any type kind of python object, but since the data was loaded from a csv, you know that it must be a text attribute. 

# In[9]:


housing.info()


# #### Use the value_counts() to have a better grasp of the categorical column
# - We can see that there are 5 different categories, ISLAND accounts for 0.024% of the total values 
# 

# In[10]:


housing["ocean_proximity"].value_counts()


# #### Use the describe function to have a summary of each numerical column
# - Note that the null values are ignored
# - 25th precentile represents the value under which 25% of numbers fall below (ex: 25% of housing_median age fall below 18.0)
# - To have a better feel of the numerical value, you can plot histograms of all the numerical values in the dataset

# In[11]:


housing.describe()


# #### Use the .hist function to have a summary of each numerical column
# - If you call hist on the entire dataset, histograms will be plotted for all the numerical variables in the dataset
# - The alternative is to plot each graph alone 
# - You'll need to change the figsize argument to have a better view of the graphs 
# - You can change the bin sizes to have a better feel of the distributions 

# #### Interpretation of the graphs
# - You can feel that the median income is too small (the measures are in 10,000)
# - housing_median_age and median_house_value were capped (they have a limit)
# - The attributes have very different scales
# - Many histograms are tail heavy, they extend much farther to the right of the median than to the left (we may have to transform these attributes for them to have bell shaped distributions)

# In[12]:


housing.hist(bins=50 ,figsize= (20,12), grid=False)


# ## 3- Create a test set

# #### Split the dataset 
# - It is important to split the dataset from the start so that the model creator doesn't make assumptions of the model (data snooping bias)
# - You would want to use a random seed in order to split the data the same way everytime you run the model
# - Dependently on the data, you may need to use stratify split which will divide the dataset based on a percentage criteria (for ex: the data set picked for the study 50% male and 50% female, thus it might be important to split the dataset randomly in the same way)
# - In our example, median income is very important to predict housing prices 
#     - On that note, you might want to use stratify sampling based on the income levels as opposed to randomly 
#         - Since median_income is a continuous variable, you need to divide the values in categories
#         - It is important to have sufficient number of instances in each stratum or else the estimate of the stratum's importance may be biased
#             - This mean that you shouldn't have many stratum and each stratum should be large enough 
#         - Since most of the values are ranged between 1.5 and 6, we will create the following split
# - When splitting is done, you can compare the distribution of the values between the training, testing and orginal dataset
# - Finally, you should remove the income_cat column from the test and train dataset

# the pd.cut() function will split the dataset between 5 bins, from 0 to 1.5; 1.5 to 3; ...

# In[13]:


housing["income_cat"]= pd.cut(housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels= [1, 2, 3, 4, 5])


# In[14]:


housing["income_cat"].hist(grid=False)


# Stratifying sampling

# In[15]:


split = skl.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# In[16]:


for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[17]:


train_set_dist= strat_test_set["income_cat"].value_counts()/len(strat_test_set)
test_set_dist=strat_train_set["income_cat"].value_counts()/len(strat_train_set)
orig_set_dist= housing["income_cat"].value_counts()/len(housing)
comparison_df= pd.DataFrame(zip(train_set_dist,test_set_dist,orig_set_dist), columns=["training","testing","orginal"])


# In[18]:


comparison_df


# In[19]:


for i in (strat_test_set, strat_train_set):
    i.drop("income_cat", axis=1, inplace=True)


# In[20]:


strat_test_set


# ## 4- Discover and visualize data to gain insights
# - If the data set is quite large you may want to sample an exploration set, to make manipulations easy and fast (this dataset is considered to be small)
# - You can use the .copy function to create a copy of the dataset, to leave the original one intact

# In[21]:


housing= strat_train_set.copy()


# #### visualizing geographical data
# - Using the parameter alpha (between 0 and 1) can make the values more transparent and thus help us understand the density better

# In[22]:


plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(10,6), s=100)


# #### visualizing housing prices
# - The radius of each circle represents district's population (parameter s) 
# - The color represents the price (parameter c)
# - We will use a predefined color map called jet (option cmap) -> colors range from blue to red
# - This graph shows us that the houses that are near the cost are prices higher and have more population

# In[26]:


housing.plot(kind="scatter", x="longitude", y="latitude", s=housing["population"]/10, 
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar= True, figsize=(20,10))


# #### looking for correlations
# - Since the dataset is not too large, you can compute the standard correlation coefficient (also called Pearson's r) between every pair of attribute using the corr() method
#     - Close to -1/1 -> highly linearly correlated 
#     - Close to 0 -> not linearly correlated 
# - Note that the standard correlation coefficient only measured linear correlation and may miss out on non linear relationships (for example, when x close to 0, y goes up)
#     - In this example, note how all the plots of the bottom row have a correlation coefficient equal to 0, despite the fact that their axes are clearly not independent: these are examples of nonlinear relationships
# - You can also use a heatmap to visualize the correlations faster

# Correlation using table 

# In[263]:


corr_matrix= housing.corr()


# In[264]:


corr_matrix["median_house_value"].sort_values(ascending= False)


# Correlation using heatmap

# In[1]:


corrmat = housing.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corrmat, vmax=1, square=True, annot= True)


# - Correlation using pandas scatter matrix function
# - Note that there are 11 attributes, 11^2= 121 scatter plots 
# - A good idea would be to plot only the ones that may be of interest 
# - Note that since the main diagonal would be filled with straight lines, pandas automatically replaces it with a histogram of each attribute

# In[266]:


from pandas.plotting import scatter_matrix
attributes=["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# #### zooming in on correlated attributes
# - since median_income and median_house_value are very correlated, it would be interesting to zoom in on the graph representing their correlation 
# - Interpretation of scatter plot
#     - Correlation is very strong: you can clearly see the upward trend, and the points are not too dispersed
# - You can clearly see price caps (500k, 350k, maybe 280k ...)
#     - You may want to remove the representing district to prevent the algorithm to reproduce these data quirks
# 

# In[267]:


#Use this to increase the font size of the axis labels
plt.rcParams.update({'font.size':20})
housing.plot(kind="scatter", x="median_income", y="median_house_value", figsize=(15,9))


# ## 5- Experimenting with attributes combinations
# - Some attributes may provide more value if they are calculated along with other attributes: try out various attribute combinations
# - In our example:
#     - The total number of rooms in a district is not very useful if you don't know how many households there are (what you really want is the number of rooms per household)
#     - The total number of bedrooms by it self is not very useful, you totaly want to compare it to the total number of rooms
# - The new bedrooms_per_room attribute is much more correlated with the median house value than the total number of rooms or bedrooms. 
# - The number of rooms per household is also more informative than the total number of rooms in a district

# In[268]:


housing["rooms_per_household"]= housing["total_rooms"]/housing["households"]
housing["bedrooms_per_rooms"]= housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]= housing["population"]/housing["households"]


# In[269]:


housing


# In[270]:


corr_matrix= housing.corr()
corr_matrix["median_house_value"].sort_values(ascending= False)


# ## 6- Prepare the data for machine learning algorithms
# - Revert back to a clean training set (by copying strat_train_set once again)
#     - Note that by using drop, you are creating a copy of the data set and not affecting the original one
# - Seperate the attribute you're trying to predict (label) from the rest of the attributes
# 

# In[271]:


housing= strat_train_set.drop("median_house_value", axis=1)


# In[392]:


housing_labels= strat_train_set["median_house_value"].copy()


# In[393]:


housing


# ## 7- Data cleaning
# - Most machine learning algorithms can't work with missing values
# - We noticed earlier that total_bedrooms_attribute has some missing values, let's fix that
#     - We can remove the entire attribute -> housing.drop("total_bedrooms", axis=1)
#     - Remove all the corresponing rows (disctrict) -> housing.dropna(subset=["total_bedrooms"])
#     - Replace the missing values with some other values (mean, median zero ...) -> housing["total_bedrooms"].fillna(median, inplace= True)
#         - If you chose this option you should compute the median value of the training set and replace the missing values of the training set 
#         - Don't forget to save the median value you calculated. You will need it later on with the testing set
#         
#         - Scikit-Learn provides a useful class to take care of missing values: SimpleImputer
#             - import
#             - Create a simple imputer instance specifying that you want to replace each attribute's missing value with the median of the attribute
#             - Remove the categorical values from the dataset 
#             - Fit the imputer instance to the training data using the fit() method
#             - Since we cannot make sure that later on, the other attributes won't have missing values, we should apply the imputer to all the numerical attributes 
#             - Replace the missing values with the learned medians
#             - Transform the numpy array back into a dataframe
#         
#     
#    

# In[273]:


from sklearn.impute import SimpleImputer 


# In[274]:


imputer= SimpleImputer(strategy="median")


# In[275]:


housing_num= housing.drop("ocean_proximity", axis=1)


# In[276]:


imputer.fit(housing_num)


# In[277]:


X= imputer.transform(housing_num)


# In[278]:


housing_tr= pd.DataFrame(X, columns= housing_num.columns, index= housing_num.index)


# ## 8- Handling text and categorical attributes
# - We should transform the categorical attributes into numerical attributes
# - One way would be to use the OrdinalEncoder class. 
#     - However, this class assumes that the categorical variables are ordinal 
# - For our case, we should use one hot encoding by using the OneHotEncoder class 
#     - For every attribute, create a column and replace one with the value of this specific attribute and 0 for the others 
#     - Note that if we have many attributes, we may want to try and replace the categorical variable with a numerical one (in our example we can change it to distance from the ocean)
#     - Notice that the output of the OneHotEncoder class is a SciPy sparce matrix instead of a NumPy array 
#         - This is very useful when you have categorical attributes with thousands of categories as it saves up memory by only storing the values of one and disregarding the positions of the zeroes
#         - if you want to convert it to a NumPy array, you just need to call the function .toarray()

# In[279]:


housing_cat= housing[["ocean_proximity"]]


# In[280]:


from sklearn.preprocessing import OneHotEncoder


# In[288]:


cat_encoder= OneHotEncoder()


# In[289]:


housing_cat_1hot= cat_encoder.fit_transform(X=housing_cat, y=None)


# In[290]:


housing_cat_1hot


# In[357]:


housing_cat_1hot.toarray()


# ## 9- Custom tranfomers
# - Sklearn provides many useful transformers
#     - Transformers such as custom cleanup operations and combining specific attributes have to be created by the user
# - You'll want your transformers to work seamlessly with Sklearn functionalities (such as piplines)
# - When you create a class you'll need to implement three methods: fit(), transform(), and fit_transform()
#     - If you add TransfomerMixin as a base class, you can get the fit_transformer() for free
#     - If you add BaseEstimator as a base class (and avoid *args and *kargs in your constructor), you will also get two extra methods (get_params() and set_params()) that will be useful for automatic hyperparameter tuning
#     
# - Here is a small transfomer that adds combined attributes we implemented above
# 

# In[396]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# ## 10- Feature scaling
# - With few exceptions, machine learning algorithms don't perform well when the scale is different 
# - You can use:
#     - min-max scaling 
#     - standardization (much less affected by outliers)
# 
# - As with all the transformations, it is important to fir the scalars to the training set only 

# In[403]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)
 


# In[370]:


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)


# In[373]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[375]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


# In[377]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
68628.19819848922


# In[378]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[397]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[398]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[399]:


display_scores(tree_rmse_scores)


# In[400]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

