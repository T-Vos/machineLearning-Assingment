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
    # print('dummy', dummy.shape)
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
        (file['references'] <12),
        (file['references'] > 12) & (file['references'] <= 18),
        (file['references'] > 18) & (file['references'] < 21),
        (file['references'] > 21) & (file['references'] < 25),
        (file['references'] > 25) & (file['references'] < 29),
        (file['references'] > 29) & (file['references'] < 33),
        (file['references'] > 33) & (file['references'] < 38),
        (file['references'] > 38) & (file['references'] < 45),
        (file['references'] > 45) & (file['references'] < 55),
        (file['references'] >= 55)
    ]
    values = [1, 2, 3, 4, 5, 6 ,7 ,8,9,10]
    file['class_ref'] = np.select(conditions, values)
    if return_with_dummy : return dummy(file,"class_ref")
    return file

# %% [markdown]
# ## Generic Data process

# %% [markdown]
# ### Fill Nan

# %%
def fill_Nan(data):
    data['year'] = data['year'].fillna(data['references'].mean())
    data['references'] = data['references'].fillna(0)
    data["fields_of_study"] = data["fields_of_study"].fillna("")
    data["title"] = data["title"].fillna("")
    return data

# %% [markdown]
# ## Length calculators

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
# 
# ## Venue

# %% [markdown]
# ### Generate Venue Data

# %%
data_set = np.array([ [x['venue'], x['citations']] for x in json.load(open('train-1.json')) if x['venue']])

def get_cat(dataFrame):
    conditions = [
        (dataFrame['cumsum'] == 0),
        (dataFrame['cumsum'] > 0) & (dataFrame['cumsum'] <= 10),
        (dataFrame['cumsum'] > 10) & (dataFrame['cumsum'] <= 20),
        (dataFrame['cumsum'] > 20) & (dataFrame['cumsum'] <= 30),
        (dataFrame['cumsum'] > 30) & (dataFrame['cumsum'] <= 40),
        (dataFrame['cumsum'] > 40) & (dataFrame['cumsum'] <= 50),
        (dataFrame['cumsum'] > 50) & (dataFrame['cumsum'] <= 60),
        (dataFrame['cumsum'] > 60) & (dataFrame['cumsum'] <= 70),
        (dataFrame['cumsum'] > 70) & (dataFrame['cumsum'] <= 80),
        (dataFrame['cumsum'] > 80) & (dataFrame['cumsum'] <= 90),
        (dataFrame['cumsum'] > 90) & (dataFrame['cumsum'] <= 92),
        (dataFrame['cumsum'] > 92) & (dataFrame['cumsum'] <= 94),
        (dataFrame['cumsum'] > 94) & (dataFrame['cumsum'] <= 96),
        (dataFrame['cumsum'] > 96) & (dataFrame['cumsum'] <= 98),
        (dataFrame['cumsum'] >= 98)
    ]
    values = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16]
    dataFrame['class_Venue'] = np.select(conditions, values)
    return dataFrame

def get_venue_dictionary(XY):
    #split data 
    X, _ = np.split(XY, 2, 1)
    
    unique_X = np.unique(X,return_counts=True)
    unique_X_length = len(unique_X[0])
    print(unique_X_length)
    zero_array = np.zeros((3,unique_X_length))

    merge = [unique_X[0], unique_X[1].astype(int), zero_array[0].astype(int), zero_array[1].astype(int), zero_array[2].astype(float)]
    
    # Compute citations per topic
    for x in XY :
        merge[2][np.where(merge[0] == x[0])[0][0]] += int(x[1])

    # Compute citations per topic divided by the amount of articles
    for (i, j) in enumerate(merge[3]):
        merge[3][i] = (merge[2][i]/merge[1][i])

    # Compute new summed citations
    summed_citations = np.sum(merge[3])
    result = []
    for (i, j) in enumerate(merge[3]):
        merge[4][i] = 100/summed_citations*merge[3][i]
        result.append((merge[0][i],merge[1][i],merge[2][i],merge[3][i],merge[4][i]))
    
    dtype = [('venue', 'S100'), ('count', int), ('summed_citations', int), ('average_citations', float), ('contribution', float)]
    
    structured_array = np.array(result, dtype=dtype)
    sorted = np.sort(structured_array, order='contribution')
    df = pd.DataFrame(sorted)
    df['cumsum'] = df['contribution'].cumsum(axis=0)
    df = get_cat(df)
    return df
get_venue_dictionary(data_set).to_pickle("venue_data.pkl")

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
# ## Topics

# %% [markdown]
# ### Generate Topic Data

# %%
data_set = np.array([ [x['topics'], x['citations']] for x in json.load(open('train-1.json')) if x['topics']])

def get_cat(dataFrame):
    conditions = [
        (dataFrame['cumsum'] == 0),
        (dataFrame['cumsum'] > 0) & (dataFrame['cumsum'] <= 10),
        (dataFrame['cumsum'] > 10) & (dataFrame['cumsum'] <= 20),
        (dataFrame['cumsum'] > 20) & (dataFrame['cumsum'] <= 30),
        (dataFrame['cumsum'] > 30) & (dataFrame['cumsum'] <= 40),
        (dataFrame['cumsum'] > 40) & (dataFrame['cumsum'] <= 50),
        (dataFrame['cumsum'] > 50) & (dataFrame['cumsum'] <= 60),
        (dataFrame['cumsum'] > 60) & (dataFrame['cumsum'] <= 70),
        (dataFrame['cumsum'] > 70) & (dataFrame['cumsum'] <= 80),
        (dataFrame['cumsum'] > 80) & (dataFrame['cumsum'] <= 90),
        (dataFrame['cumsum'] > 90)
    ]
    values = [0,1, 2, 3, 4,5,6,7,8,9,10]
    dataFrame['class_topic'] = np.select(conditions, values)
    return dataFrame

def unique_Array_of_nested_array(x):
    topics_array = list(map((lambda z: z[0]), x))
    topics_array = np.concatenate(topics_array)
    return np.unique(topics_array,return_counts=True)

def get_venue_dictionary(XY):
    #split data 
    X, _ = np.split(XY, 2, 1)

    # Get unique topics
    unique_topics = unique_Array_of_nested_array(X)
    unique_topics_lenght = len(unique_topics[0])

    # Generate 0 arrays in order to fill them with the counts
    zero_arrays = np.zeros((3,unique_topics_lenght))

    # Merge arrays together
    merge = [unique_topics[0], unique_topics[1].astype(int), zero_arrays[0].astype(int), zero_arrays[1].astype(int), zero_arrays[2].astype(float)]

    # Compute citations per topic
    for x in XY :
        for topic in x[0]:
            merge[2][np.where(merge[0] == topic)[0][0]] += int(x[1])


    # Compute citations per topic divided by the amount of articles
    for (i, j) in enumerate(merge[3]):
        merge[3][i] = (merge[2][i]/merge[1][i])

    # Compute new summed citations
    summed_citations = np.sum(merge[3])

    result = []
    # Compute percentage of new total sum
    for (i, j) in enumerate(merge[3]):
        merge[4][i] = 100/summed_citations*merge[3][i]
        result.append((merge[0][i],merge[1][i],merge[2][i],merge[3][i],merge[4][i]))

    dtype = [('topic', 'str'), ('count', int), ('summed_citations', int), ('average_citations', float), ('contribution', float)]
    structured_array = np.array(result, dtype=dtype)

    sorted = np.sort(structured_array, order='contribution')
    df = pd.DataFrame(sorted)
    df['cumsum'] = df['contribution'].cumsum(axis=0)
    df = get_cat(df)
    return df
get_venue_dictionary(data_set).to_pickle("topic_data.pkl")

# %% [markdown]
# ### Extract topic data

# %% [markdown]
# 

# %% [markdown]
# ## Abstract
#  

# %% [markdown]
# ### Abstract Vector

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def create_countVector():
    train_data = [ x['abstract'] for x in json.load(open('train-1.json')) if x['abstract']]
    tfIdfTransformer = TfidfTransformer(use_idf=True)
    countVectorizer = CountVectorizer()
    wordCount = countVectorizer.fit_transform(train_data)
    newTfIdf = tfIdfTransformer.fit_transform(wordCount)
    df = pd.DataFrame(newTfIdf[0].T.todense(), index = countVectorizer.get_feature_names(), columns = ["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending = False)

create_countVector()


# %% [markdown]
# # Figures

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
# ## Histogram

# %%
def PlotHistY(Y):
    plt.hist(x=Y, bins=20)
    plt.show()

# %% [markdown]
# # Model

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
# ## Models

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR,LinearSVR
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor,VotingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

def modelCalculators(X_train, y_train, X_val, y_val, train=True):
    
    sdgRegres = make_pipeline(StandardScaler(), SGDRegressor(max_iter=100000, tol=1e-3))

    if train:
        sdgRegres.fit(X_train, y_train)
        sdgRegres_score = sdgRegres.score(X_val, y_val)
        # sdgRegres_transformed = sdgRegres.predict(X_val)
        # sdgRegres_pred = np.exp(sdgRegres_transformed)
        print('SDG',sdgRegres_score)


    linearRidg = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=1.1))

    if train:
        linearRidg.fit(X_train, y_train)
        linearRidg_score = linearRidg.score(X_val, y_val)
        # linearRidg_transformed = linearRidg.predict(X_val)
        # linearRidg_pred = np.exp(linearRidg_transformed)
        print('Ridg',linearRidg_score)
    
    linearLasso = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=1.1, selection='random'))

    if train:
        linearLasso.fit(X_train, y_train)
        linearLasso_score = linearLasso.score(X_val, y_val)
        # linearLasso_transformed = linearLasso.predict(X_val)
        # linearLasso_pred = np.exp(linearLasso_transformed)
        print('Lasso',linearLasso_score)

    svrRegress = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2, kernel='rbf', degree=4))

    if train:
        svrRegress.fit(X_train, y_train)
        svrRegress_score = svrRegress.score(X_val, y_val)
        # svrRegress_transformed = svrRegress.predict(X_val)
        # svrRegress_pred = np.exp(svrRegress_transformed)
        print('SVR Regress',svrRegress_score)

    linearSVR = make_pipeline(StandardScaler(), LinearSVR(random_state=0,max_iter=100000, tol=1e-5))

    if train:
        linearSVR.fit(X_train, y_train)
        linearSVR_score = linearSVR.score(X_val, y_val)
        # linearSVR_transformed = linearSVR.predict(X_val)
        # linearSVR_pred = np.exp(linearSVR_transformed)
        print('SVR Lin',linearSVR_score)

    adaBoostReg = AdaBoostRegressor(random_state=0, n_estimators=100)

    if train:
        adaBoostReg.fit(X_train, y_train)
        adaBoostReg_score = adaBoostReg.score(X_val, y_val)
        adaBoostReg_transformed = adaBoostReg.predict(X_val)
        # adaBoostReg_pred = np.exp(adaBoostReg_transformed)
        print('Ada',adaBoostReg_score)

    regressionVoted = VotingRegressor([('Ridg', linearRidg),('SVR', svrRegress),('ada', adaBoostReg), ('linearSVR', linearSVR)],n_jobs=-1)
    regressionVoted.fit(X_train, y_train)
    
    if train : 
        regressionVoted_score = regressionVoted.score(X_val, y_val)
        print('Voted',regressionVoted_score)

    prediction = regressionVoted.predict(X_val)

    return prediction

# %% [markdown]
# ## Data transformation

# %%
def transformData(data):
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
    data = data.assign(topicsFoS = lambda x: x['topics_len'] * x['fields_of_study_len'])
    data = data.assign(topicsVenueCat = lambda x: x['topics_len'] * x['venue_cat'])
    data = data.assign(references2 = lambda x: x['references'] * x['references'])
    data = data.assign(references3 = lambda x: x['references'] * x['references']* x['references'])
    data = data.assign(references4 = lambda x: x['references'] * x['references']* x['references']* x['references'])
    return data

# %% [markdown]
# ## Create X

# %%
def return_x(data, train=True):
    if train : return data.drop(["citations", "year", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values
    return data.drop(["year", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values

# %% [markdown]
# ## Training
# 
# The function will consist of two steps, first the program will load all the data and process this. This is necessary in order to know the relative contribution of different variables. Then it will split the data file in two sections, train and test in order to know the respective progress of the learning task.

# %%
def model(): 
    data = open_json_data()

    data = transformData(data)
    print(data.shape)
            
    X = return_x(data)
    y = data["citations"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=2000)
    
    y_train_transformed = np.log1p(y_train)
    y_val_transformed = np.log1p(y_val)

    predY = modelCalculators(X_train, y_train_transformed, X_val, y_val_transformed)
    # pred_y = modelCalculators(X_train, y_train, X_val, y_val)

    pred_y = np.exp(predY)

    plotY(pred_y)
    plotY(y_val)
    plotY2(pred_y, y_val)
    eval(pred_y, y_val)
    YpredSorted = np.sort(pred_y)
    print(YpredSorted)
    return

model()

# %% [markdown]
# # Test generator

# %%
def model(): 
    data = open_json_data()
    data = transformData(data)
    print(data.shape)

    data_test = open_json_data('test.json')
    data_test = transformData(data_test)
    print(data_test.shape)
  
    data_test_final = open_json_data('test.json')
    
    X = return_x(data)
    y = data["citations"].values

    X_test = return_x(data_test, False)

    predictions = modelCalculators(X, y, X_test, [], False)

    Ypred = pd.Series(predictions.astype(int))
    Ypred = Ypred.where(Ypred > 0, 0)
    
    data_test_final["citations"] = Ypred.astype('int')

    YpredSorted = np.sort(Ypred)
    print(YpredSorted)
    plotY(Ypred)
   
    return data_test_final.drop(["year", "references", "is_open_access","title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1)

model().to_json("predicted.json", orient='records')

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


