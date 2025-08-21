import pandas as pd


def categoric_to_numeric_bool(serie):
    """
    Converts a pandas Series containing categorical boolean values ('Yes'/'No') to numeric values (1/0).
    Parameters:    
    serie : pandas.Series
        Series containing categorical values 'Yes' and 'No'.
    Returns:    
    pandas.Series
        Series with 'Yes' mapped to 1 and 'No' mapped to 0.
    """
    serie = serie.map(
        {'Yes': 1, 'No': 0})

    return serie
