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
    # return pd.DataFrame(dictionary)[:10]
    return pd.DataFrame(dictionary)

# %% [markdown]
# # 2. Process data
# 

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
    data['year'] = data['year'].fillna(data['references'].mean())
    data['references'] = data['references'].fillna(0)
    data["fields_of_study"] = data["fields_of_study"].fillna("")
    data["title"] = data["title"].fillna("")
    return data

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
def get_titleLen(data):
    data["title_len"] = data.apply(lambda x:len(x["title"]),axis=1)
    return data

# %% [markdown]
# ### Authors length

# %%
def get_AuthorsLen(data):
    data["authors_len"] = data.apply(lambda x:len(x["authors"]),axis=1)
    return data

# %% [markdown]
# ### Topics Length

# %%
def get_TopicsLen(data):
    data["topics_len"] = data.apply(lambda x:len(x["topics"]),axis=1)
    return data

# %% [markdown]
# ### Fields of Study Length

# %%
def get_FieldsOfStudyLen(data):
    data["fields_of_study_len"] = data.apply(lambda x:len(x["fields_of_study"]),axis=1)
    return data

# %% [markdown]
# ### Find venue data

# %%
def find_venue_in_dataFrame(venue, venues_dict):
    venue_cat = venues_dict[venues_dict["venue"].str.decode("utf-8")  == venue]
    if len(venue_cat)>0 : return venue_cat.iloc[0]['class_Venue']
    return 0

def get_venue(data):
    venue_dictionary = pd.read_pickle("venue_data.pkl")
    data["venue_cat"] = data.apply(lambda x:find_venue_in_dataFrame(x.venue, venue_dictionary),axis=1)
    return data


# %% [markdown]
# # Plot line

# %% [markdown]
# ## Plot single line

# %%
import matplotlib.pyplot as plt
def plotY(Y):
    Y = np.array(Y)
    Y = np.sort(Y)
    plt.plot(Y)
    return plt.show()

# %% [markdown]
# ## Plot 2 lines

# %%
import matplotlib.pyplot as plt
def plotY2(Ypred,Yreal):
    length = len(Ypred)
    plt.plot(Yreal,'b')
    plt.plot(Ypred,'r')
    return plt.show()

# %% [markdown]
# # Model

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# ## Evaluation

# %%
def process_data(data):
    data = fill_Nan(data)
    data = get_ref(data, True)
    data = get_years(data, True)    
    data = get_titleLen(data)
    data = get_FieldsOfStudyLen(data)
    data = data.assign(authorsLen2 = lambda x: x['fields_of_study_len'] * x['fields_of_study_len'])
    data = data.assign(authorsLen3 = lambda x: x['fields_of_study_len'] * x['fields_of_study_len'] * x['fields_of_study_len'])
    data = get_TopicsLen(data)
    data = data.assign(authorsLen2 = lambda x: x['topics_len'] * x['topics_len'])
    data = data.assign(authorsLen3 = lambda x: x['topics_len'] * x['topics_len'] * x['topics_len'])
    data = get_AuthorsLen(data)
    data = data.assign(authorsLen2 = lambda x: x['authors_len'] * x['authors_len'])
    data = data.assign(authorsLen3 = lambda x: x['authors_len'] * x['authors_len'] * x['authors_len'])
    return data

# %% [markdown]
# # Throw everthing together
# The function will consist of two steps, first the program will load all the data and process this. This is necessary in order to know the relative contribution of different variables. Then it will split the data file in two sections, train and test in order to know the respective progress of the learning task.

# %%
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
# from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn import linear_model

def model():
    
    data = open_json_data()
    
    data = fill_Nan(data)
    data = get_ref(data, True)
    data = get_years(data, True)    
    data = get_titleLen(data)
    data = get_FieldsOfStudyLen(data)
    data = get_TopicsLen(data)
    data = get_AuthorsLen(data)
    data = get_venue(data)

    data = data.assign(fields_of_study_len2 = lambda x: x['fields_of_study_len'] * x['fields_of_study_len'])
    data = data.assign(fields_of_study_len3 = lambda x: x['fields_of_study_len'] * x['fields_of_study_len'] * x['fields_of_study_len'])
    data = data.assign(topicsLen2 = lambda x: x['topics_len'] * x['topics_len'])
    data = data.assign(topicsLen3 = lambda x: x['topics_len'] * x['topics_len'] * x['topics_len'])
    data = data.assign(topicsLen4 = lambda x: x['topics_len'] * x['topics_len'] * x['topics_len']* x['topics_len'])
    data = data.assign(topicsLen5 = lambda x: x['topics_len'] * x['topics_len'] * x['topics_len']* x['topics_len']* x['topics_len'])
    data = data.assign(authorsLen2 = lambda x: x['authors_len'] * x['authors_len'])
    data = data.assign(authorsLen3 = lambda x: x['authors_len'] * x['authors_len'] * x['authors_len'])
    data = data.assign(topicsFoS = lambda x: x['topics_len'] * x['fields_of_study_len'])
    data = data.assign(topicsRef = lambda x: x['topics_len'] * x['references'])
    data = data.assign(topicsVenueCat = lambda x: x['topics_len'] * x['venue_cat'])
    data = data.assign(references2 = lambda x: x['references'] * x['references'])
    data = data.assign(references3 = lambda x: x['references'] * x['references']* x['references'])
    data = data.assign(references4 = lambda x: x['references'] * x['references']* x['references']* x['references'])

    print(data.shape)
        
    X = data.drop(["citations", "year", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values
    y = data["citations"].values

    # plotY(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=2021)
    # model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))
    # model = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=1.1))
    model = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=1.1, selection='random'))
    # model = make_pipeline(StandardScaler(), RandomForestRegressor(max_depth=2, random_state=0))
    # model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2,degree=2))

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    predictions = model.predict(X_val)

    Ypred = np.array(predictions)
    Ypred = np.sort(predictions)
    Yreal = np.array(y_val)
    Yreal = np.sort(y_val)
    # plotY(Ypred)
    plotY2(predictions, y_val)

    print(Ypred)
    print(Yreal)
    print(eval(predictions, y_val))
    print(score)
    # print(score(y_val, predictions))
    # print(model.score(X_val, y_val))
    return 
    # huber = HuberRegressor().fit(X_train, y_train)
    # print(huber.score(X_val, y_val))

    # linear = LinearRegression()
    # linear.fit(X_train, y_train)
    # print(linear.score(X_val,y_val))

    # return model
    # return model.score(X_val, y_val)
    # return predictions
    # return model.fit(X_train, y_train).score(X_val, y_val)
    
    # return cross_val_score(model, X, y, cv=10)
    # cross_val_score(model, X, y, cv=10)
    # return cross_val_score(model, X, y, cv=10)

model()

# %% [markdown]
# ## Test model

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn import linear_model

def model():
    
    data = open_json_data()
    data = fill_Nan(data)
    data = get_ref(data, False)
    data = get_years(data, False)    
    data = get_titleLen(data)
    data = get_FieldsOfStudyLen(data)
    data = get_TopicsLen(data)
    data = get_AuthorsLen(data)
    
    print(data.shape)

    data_test = open_json_data('test.json')
    data_test = fill_Nan(data_test)
    data_test = get_ref(data_test, False)
    data_test = get_years(data_test, False)    
    data_test = get_titleLen(data_test)
    data_test = get_FieldsOfStudyLen(data_test)
    data_test = get_TopicsLen(data_test)
    data_test = get_AuthorsLen(data_test)

    print(data_test.shape)

    
    X_train = data.drop(["citations", "year", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values
    y_train = data["citations"].values
    plotY(y_train)

    X_test = data_test.drop(["year", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values

    model = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=1.1))

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    
    Ypred = pd.Series(predictions.astype(int))
    Ypred = Ypred.where(Ypred > 0, 0)
    
    data_test["citations"] = predictions.astype('int') 
    
    # Ypred = np.array(Ypred)
    YpredSorted = np.sort(Ypred)
    print(YpredSorted)

    
    return data_test

model().to_json("output.json", orient='records')


