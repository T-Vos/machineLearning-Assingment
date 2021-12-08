import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

def open_json_data(fileLocation = 'train-1.json'):
    # Opening JSON file
    openFile = open(fileLocation)
    dictionary = json.load(openFile)
    # return pd.DataFrame(dictionary)[:10]
    return pd.DataFrame(dictionary)

def get_contribution_topic(data):
    return

def get_contribution_FieldOfStudy(data):
    return

def dummy(data, dummy_column, delete_cat = True, spase= False):
    ## create dummy
    dummy = pd.get_dummies(data[dummy_column],prefix=dummy_column, drop_first=True, sparse=spase)
    print('dummy', dummy.shape)
    data = pd.concat([data, dummy], axis=1)
    ## drop the original categorical column
    if delete_cat : return data.drop(dummy_column, axis=1)
    return data

def dummies_from_nestedList(data, dummy_column):
    ## create dummy
    print(data)
    dummy = pd.get_dummies(data[dummy_column, 1].apply(pd.Series).stack()).sum(level=0)
    print('dummy', dummy.shape)
    data = pd.concat([data, dummy], axis=1)
    ## drop the original categorical column
    return data

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

def fill_Nan(data):
    data['year'] = data['year'].fillna(data['references'].mean())
    data['references'] = data['references'].fillna(0)
    data["fields_of_study"] = data["fields_of_study"].fillna("")
    data["title"] = data["title"].fillna("")
    return data

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

def get_titleLen(data):
    data["title_len"] = data.apply(lambda x:len(x["title"]),axis=1)
    return data

def get_AuthorsLen(data):
    data["authors"] = data.apply(lambda x:len(x["authors"]),axis=1)
    return data

def get_TopicsLen(data):
    data["topics"] = data.apply(lambda x:len(x["topics"]),axis=1)
    return data

def get_FieldsOfStudyLen(data):
    data["fields_of_study"] = data.apply(lambda x:len(x["fields_of_study"]),axis=1)
    return data

def find_venue_in_dataFrame(venue, venues_dict):
    venue_cat = venues_dict[venues_dict["venue"].str.decode("utf-8")  == venue]
    if len(venue_cat)>0 : return venue_cat.iloc[0]['class_Venue']
    return 0

def get_venue(data):
    venue_dictionary = pd.read_pickle("venue_data.pkl")
    data["venue_cat"] = data.apply(lambda x:find_venue_in_dataFrame(x.venue, venue_dictionary),axis=1)
    return data

def score(Y_true, Y_pred):
    y_true = np.log1p(np.maximum(0, Y_true))
    y_pred = np.log1p(np.maximum(0, Y_pred))
    return 1 - np.mean((y_true-y_pred)**2) / np.mean((y_true-np.mean(y_true))**2)

def evaluate(gold_path, pred_path):
    gold = { x['doi']: x['citations'] for x in json.load(open(gold_path)) }
    pred = { x['doi']: x['citations'] for x in json.load(open(pred_path)) }
    y_true = np.array([ gold[key] for key in gold ])
    y_pred = np.array([ pred[key] for key in gold ])
    return score(y_true, y_pred)

def model():
    data = open_json_data()
    data = fill_Nan(data)

    data = get_ref(data, False)
    data = get_years(data, False)    
    data = get_titleLen(data)
    data = get_venue(data)
    data = get_FieldsOfStudyLen(data)
    data = get_TopicsLen(data)
    data = get_AuthorsLen(data)
    print(data.shape)
    # print(data)
    
    X = data.drop(["citations", "year", "doi", "title", "abstract", "authors", "topics","fields_of_study", "venue"], axis=1).values
    y = data["citations"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=2021)
    model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=10000, tol=1e-3))
    # model = make_pipeline(StandardScaler(), linear_model.Ridge(alpha=1.1))
    # model = make_pipeline(StandardScaler(), linear_model.Lasso(alpha=1.1))
    # model = make_pipeline(StandardScaler(), RandomForestRegressor(max_depth=2, random_state=0))

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    predictions = model.predict(X_val)

    print(score)
    print(predictions)
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


