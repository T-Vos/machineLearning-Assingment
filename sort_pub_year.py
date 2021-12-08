import load_file
import numpy as np
import pandas as pd

def get_years(file):
    conditions = [
        (file['year'] < 2000),
        (file['year'] >= 2000) & (file['year'] <= 2010),
        (file['year'] > 2010) & (file['year'] < 2016),
        (file['year'] >= 2016)
    ]

    values = [1, 2, 3, 4]
    file['class_year'] = np.select(conditions, values)

    return file

def get_cit(file):
    conditions = [
        (file['citations'] == 0),
        (file['citations'] > 0) & (file['citations'] <= 20),
        (file['citations'] > 20) & (file['citations'] < 40),
        (file['citations'] >= 40)
    ]

    values = [1, 2, 3, 4]
    file['class_cit'] = np.select(conditions, values)
    return file

def get_ref(file):
    conditions = [
        (file['references'] == 0),
        (file['references'] > 0) & (file['references'] <= 30),
        (file['references'] > 30) & (file['references'] < 60),
        (file['references'] >= 60)
    ]

    values = [1, 2, 3, 4]
    file['class_ref'] = np.select(conditions, values)
    return file

def main():
   pd.set_option('display.max_columns', None)
   file = load_file.main()
   file = get_years(file)
   file = get_cit(file)
   file = get_ref(file)

   return file

#main()

