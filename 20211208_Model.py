# %% [markdown]
# # 1. Import Data

# %%
# import libraries
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
 
def open_json_data(fileLocation = 'train-1.json'):
    # Opening JSON file
    openFile = open(fileLocation)
    dictionary = json.load(openFile)
    # return pd.DataFrame(dictionary)[:1000]
    return pd.DataFrame(dictionary)
    # return np.array(dictionary)

# %% [markdown]
# # 2. Process data
# Before feeding the data to the machine, we want to process the data:
# - Adding the mean contribution per topic
# - Adding the mean contribution per Venue
# - Adding the mean contribution per field of study
# - Adding the years grouped
# - fill Not a numbers (Nan)
# 

# %%
def find_venue_in_dataFrame(dataLine):
    venue_dictionary = pd.read_pickle("venue_data.pkl")
    dataLine["class_Venue"] = 0
    return dataLine
    return

def get_venue(data):

    return data
    


# %%
def get_contribution_topic(data):
    return

# %%
def get_contribution_FieldOfStudy(data):
    return

# %% [markdown]
# ## Dummy builders

# %% [markdown]
# ### Get sparseDummies

# %% [markdown]
# ### Get Dummies from single input

# %%
# def dummy(data, dummy_column, delete_cat = True):
#     ## create dummy
#     dummy = pd.get_dummies(data[dummy_column],prefix=dummy_column, drop_first=True)
#     data = pd.concat([data, dummy], axis=1)
#     ## drop the original categorical column
#     if delete_cat : return data.drop(dummy_column, axis=1)
#     return data

# %%
def dummy(data, dummy_column, delete_cat = True, spase= False):
    ## create dummy
    dummy = pd.get_dummies(data[dummy_column],prefix=dummy_column, drop_first=True, sparse=spase)
    print('dummy', dummy.shape)
    data = pd.concat([data, dummy], axis=1)
    ## drop the original categorical column
    if delete_cat : return data.drop(dummy_column, axis=1)
    return data

# %% [markdown]
# ### Get Dummies from list

# %%
def dummies_from_nestedList(data, dummy_column):
    ## create dummy
    print(data)
    dummy = pd.get_dummies(data[dummy_column, 1].apply(pd.Series).stack()).sum(level=0)
    print('dummy', dummy.shape)
    data = pd.concat([data, dummy], axis=1)
    ## drop the original categorical column
    # if delete_cat : return data.drop(dummy_column, axis=1)
    return data

# %% [markdown]
# ## Year Data

# %% [markdown]
# ### Adding the categorical years
# 

# %%
def get_years(data, return_with_dummy= False):
    conditions = [
        (data['year'] < 2000),
        (data['year'] >= 2000) & (data['year'] <= 2010),
        (data['year'] > 2010) & (data['year'] < 2016),
        (data['year'] >= 2016)
    ]
    values = [1, 2, 3, 4]
    data['class_year'] = np.select(conditions, values)
    if return_with_dummy : return dummy(data,"class_year")
    return data

# %% [markdown]
# ## References

# %% [markdown]
# ### Adding Categorical References

# %%
def get_ref(file, return_with_dummy= False):
    conditions = [
        (file['references'] == 0),
        (file['references'] > 0) & (file['references'] <= 30),
        (file['references'] > 30) & (file['references'] < 60),
        (file['references'] >= 60)
    ]
    values = [1, 2, 3, 4]
    file['class_ref'] = np.select(conditions, values)
    if return_with_dummy : return dummy(file,"class_ref")
    return file

# %% [markdown]
# ## Fill Nan

# %%
# def fill_Nan(data):
#     for column in data:
#         if data[column].dtype == np.float64 or data[column].dtype == np.int64:data[column].fillna(data[column].mean())
#     return data
def fill_Nan(data):
    data['year'] = data['year'].fillna(0)
    data['references'] = data['references'].fillna(0)
    data["fields_of_study"] = data["fields_of_study"].fillna("")
    data["fields_of_study"] = data["fields_of_study"].fillna("")
    return data

# %% [markdown]
# ## Evaluation

# %%
from sklearn import metrics

def eval(predicted, y_test):
    ## Kpi
    print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((y_test - predicted) / predicted)), 2))
    print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):",
            "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
    ## residuals
    residuals = y_test - predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(
        residuals).index(min(residuals))
    # max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))
    return(residuals,max_error,max_idx)

# %% [markdown]
# ### Title length
# 

# %%
def get_titleLength(data):

# %% [markdown]
# # Throw everthing together
# The function will consist of two steps, first the program will load all the data and process this. This is necessary in order to know the relative contribution of different variables. Then it will split the data file in two sections, train and test in order to know the respective progress of the learning task.

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, LinearRegression

def model():
    
    data = open_json_data()
    data = fill_Nan(data)

    data = get_ref(data, False)
    data = get_years(data, False)
    data = data.drop('year', axis=1)
    
    get_venue(data)

    print(data.shape)
    X = data.drop(["citations", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values
    y = data["citations"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=2021)
    # model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, early_stopping=True, validation_fraction=0.33))
    
    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))
    model.fit(X_train, y_train)
    print(model.score(X_val, y_val))

    # predictions = model.predict(X_val)

    # print(model.predict(X_val))
    # print(model.score(X_val, y_val))
    # print(eval(predictions, y_val))

    # huber = HuberRegressor().fit(X_train, y_train)
    # print(huber.score(X_val, y_val))

    # linear = LinearRegression()
    # linear.fit(X_train, y_train)
    # print(linear.score(X_val,y_val))

    # return model
    # return model.score(X_val, y_val)
    return
    # return model.fit(X_train, y_train).score(X_val, y_val)
    
    # return cross_val_score(model, X, y, cv=10)
    # cross_val_score(model, X, y, cv=10)
    # return cross_val_score(model, X, y, cv=10)

model()


