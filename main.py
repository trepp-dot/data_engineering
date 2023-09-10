"""
A program that loads a csv file and preforms data cleaning and analysis.
The program will then output a csv file with the cleaned data and plots.
The cleaning will include:
    - Removing rows with missing data
    - Removing rows with duplicate data
    - filling in missing data with the mean of the column for numerical data and the mode for categorical data
    - plot data statistics and correlations
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
from sklearn import linear_model
import xgboost
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.over_sampling import SMOTENC, SMOTEN


# create a class to fetch and clean data
class DataOrganizer:
    """A class to fetch and clean data"""
    def __init__(self, path, target):
        self.path = path
        self.target = target
        self.df = None
        self.cols = None
        self.num_cols = None
        self.nan_cols = None

    def load_data(self):
        """Load data from a csv file"""
        try:
            self.df = pd.read_csv(self.path)
            self.cols = self.df.columns.tolist()
            self.num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            self.nan_cols = self.df.columns[self.df.isna().any()].tolist()
            return self.df
        except FileNotFoundError as e:
            logging.error(e)
            sys.exit(1)

    def clean_data(self):
        """Clean data by removing rows with missing data, invalid data, and duplicates"""
        # Remove rows with missing data more than 30%
        self.df = self.df.dropna(thresh=len(self.df)*0.7, axis=1)
        print(f'The columns with missing data more than 30% are: {[i for i in self.cols if i not in self.df.columns.tolist()]}')

        # Remove rows with duplicate data
        print(f'The number of duplicate rows is: {self.df.duplicated().sum()}')
        self.df = self.df.drop_duplicates()

        # Remove id column
        id_col = [col for col in self.df.columns if 'id' in col.lower()]
        if len(id_col) == 1:
            if self.df[id_col[0]].nunique() == len(self.df):
                print(f'The {id_col[0]} column is an id column and will be removed')
                self.df = self.df.drop(id_col[0], axis=1)
        elif len(id_col) > 1:
            print('found more than one id column')
        else:
            print('no id column found')
        return self.df

    def set_data_type(self):
        # set date column to datetime if it exists
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])

    def fill_missing_data(self, method='default', col_missing_value=None):
        # fill in missing data with the mean of the column for numerical data and the mode for categorical data
        for col in self.df.columns:
            if col in self.nan_cols and col is not self.target:
                if method == 'defualt':
                    print('For the defualt method fill in missing data with the mean of the column for numerical data and the mode for categorical data')
                    if self.df[col].dtype == 'object':
                        print(f'For {col} replace Nan values with the mode of {self.df[col].mode()[0]}')
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    else:
                        print(f'For {col} replace Nan values with the mean of {self.df[col].mean()}')
                        self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif method == 'ffill':
                    print('For the ffill method fill in missing data with the previous value')
                    self.df[col] = self.df[col].fillna(method='ffill')
                elif method == 'bfill':
                    print('For the bfill method fill in missing data with the next value')
                    self.df[col] = self.df[col].fillna(method='bfill')
                elif method == 'col_missing_value':
                    print(f'For the col_missing_value method fill in missing data with {col_missing_value}')
                    self.df[col] = self.df[col].fillna(col_missing_value[col])
                elif method == 'regresion':
                    print('For the regresion method fill in missing data with a regresion model')
                    regression_cols = [i for i in self.df.columns.tolist() if
                                       i in self.num_cols and i not in self.target and i not in self.nan_cols]
                    if col in self.num_cols:
                        regression_df = self.df[regression_cols].dropna(subset=[col])
                        x = regression_df.drop(col, axis=1)
                        y = self.df[col]
                        reg = linear_model.LinearRegression(fit_intercept=True)
                        reg.fit(x, y)
                        print('Intercept: \n', reg.intercept_)
                        print('Coefficients: \n', reg.coef_)
                        print('R2: \n', reg.score(x, y))
                        self.df[col] = self.df[col].fillna(reg.predict(x))
                    else:
                        print(f'For {col} replace Nan values with the mode of {self.df[col].mode()[0]}')
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif method == 'xgboost':
                    print(f'For the {col} column fill in missing data with a xgboost model')
                    model_cols = [i for i in self.df.columns.tolist() if i not in self.target and i not in self.nan_cols]
                    if col in self.num_cols:
                        model_df = self.df[model_cols + [col]]
                        x = model_df.dropna(subset=[col])[model_cols]
                        y = model_df.dropna(subset=[col])[col]
                        xgb = xgboost.XGBRegressor(obgective='reg:squarederror', n_estimators=1000, seed=49, n_jobs=-1,
                                                   max_depth=5, learning_rate=0.1)
                        xgb.fit(x, y)
                        print('R2: \n', xgb.score(x, y))
                        print('rmse: \n', np.sqrt(np.mean((y - xgb.predict(x)) ** 2)))
                        self.df.loc[self.df[col].isnull(), col] = xgb.predict(
                            model_df[model_df[col].isnull()][model_cols])
                    else:
                        print(f'For {col} replace Nan values with the mode of {self.df[col].mode()[0]}')
                        self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def imbalance_data(self, method='SMOTE'):
        # check if data is imbalanced
        if self.target:
            print(f'The number of unique values in the target column is: {self.df[self.target].value_counts()}')
            print(f'The percentage of unique values in the target column is: {self.df[self.target].value_counts(normalize=True)}')
            # check if the target label is not balanced
            balanced_df = self.df[self.target].value_counts(normalize=True)
            if len(self.df[self.target].value_counts()) > 2:
                print('The target column is not binary')
            if balanced_df[0] < 0.3 or balanced_df[0] > 0.7:
                print('The target column is not balanced')
                if method == 'SMOTE':
                    # create samples for imbalanced data useing SMOTE
                    sm = SMOTE(random_state=42)
                elif method == 'ADASYN':
                    # create samples for imbalanced data useing ADASYN -- focus on the samples which are difficult to classify with a nearest-neighbors rule
                    sm = ADASYN(random_state=42)
                elif method == 'SVMSMOTE':
                    # create samples for imbalanced data using SVMSMOTE
                    sm = SVMSMOTE(random_state=42)
                elif method == 'BorderlineSMOTE':
                    # create samples for imbalanced data useing BorderlineSMOTE -- detect which point to select which are in the border between two classes.
                    sm = BorderlineSMOTE(random_state=42)
                elif method == 'KMeansSMOTE':
                    # create samples for imbalanced data using KMeansSMOTE
                    sm = KMeansSMOTE(random_state=42, k_neighbors=5, )
                elif method == 'SMOTENC':
                    # create samples for imbalanced data using SMOTENC -- mixed of continuous and categorical features
                    sm = SMOTENC(random_state=42, categorical_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                elif method == 'SMOTEN':
                    # create samples for imbalanced data using SMOTEN -- only categorical features
                    sm = SMOTEN(random_state=42)
                X_res, y_res = sm.fit_resample(self.df.drop(self.target, axis=1), self.df[self.target])
                self.df = pd.concat([X_res, y_res], axis=1)
                print(f'The percentage of unique values in the target column is: {self.df[self.target].value_counts(normalize=True)}')

    def print_stats(self):
        """Print statistics about the data"""
        print(self.df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])
        print('\n')
        # print for each column the number of unique values and null values percentage
        for col in self.df.columns:
            print(
                f'{col}: {self.df[col].nunique()} unique values, {round(self.df[col].isnull().sum() / len(self.df), 2)} null values, {self.df[col].dtypes} type')
        print('\n')
        print(self.df.head())

    def plot_data(self):
        """Plot data statistics and correlations"""
        # Plot histogram of the data
        self.df.hist(bins=50, figsize=(20,15))
        plt.show()

        # Plot correlation matrix for numerical columns
        sns.heatmap(self.df[[col for col in self.df if col in self.num_cols]].corr(), annot=True)
        plt.show()

    def save_data(self):
        """Save cleaned data to a csv file"""
        self.df.to_csv('cleaned_data.csv', index=False)

    def extract_features_from_text(self, method ,col, new_col, drop_col=False):
        """Extract features from text data"""
        if 'second_word' in method:
            self.df[new_col] = self.df[col].str.split().str[1]
        if 'len' in method:
            self.df[new_col] = self.df[col].str.len()
        if 'numbers' in method:
            self.df[new_col] = self.df[col].str.count('\d+') # count numbers
        if 'upper' in method:
            self.df[new_col] = self.df[col].str.count('[A-Z]')
        if drop_col:
            self.df = self.df.drop(col, axis=1)

    def extract_features_from_date(self, method, col, new_col, drop_col=False):
        """Extract features from date data"""
        if 'year' in method:
            self.df[new_col] = self.df[col].dt.year
        if 'month' in method:
            self.df[new_col] = self.df[col].dt.month
        if 'day' in method:
            self.df[new_col] = self.df[col].dt.day
        if 'hour' in method:
            self.df[new_col] = self.df[col].dt.hour
        if 'minute' in method:
            self.df[new_col] = self.df[col].dt.minute
        if 'second' in method:
            self.df[new_col] = self.df[col].dt.second
        if 'weekday' in method:
            self.df[new_col] = self.df[col].dt.weekday
        if 'weekofyear' in method:
            self.df[new_col] = self.df[col].dt.weekofyear
        if 'quarter' in method:
            self.df[new_col] = self.df[col].dt.quarter
        if 'is_month_start' in method:
            self.df[new_col] = self.df[col].dt.is_month_start
        if 'is_month_end' in method:
            self.df[new_col] = self.df[col].dt.is_month_end
        if 'is_quarter_start' in method:
            self.df[new_col] = self.df[col].dt.is_quarter_start
        if 'is_quarter_end' in method:
            self.df[new_col] = self.df[col].dt.is_quarter_end
        if 'is_year_start' in method:
            self.df[new_col] = self.df[col].dt.is_year_start
        if 'is_year_end' in method:
            self.df[new_col] = self.df[col].dt.is_year_end
        if 'is_leap_year' in method:
            self.df[new_col] = self.df[col].dt.is_leap_year
        if 'days_in_month' in method:
            self.df[new_col] = self.df[col].dt.days_in_month
        if 'daysinmonth' in method:
            self.df[new_col] = self.df[col].dt.daysinmonth
        if 'quarter' in method:
            self.df[new_col] = self.df[col].dt.quarter
        if 'is_month_start' in method:
            self.df[new_col] = self.df[col].dt.is_month

    def one_hot_encoder(self):
        """One hot encode categorical data"""
        col_to_remove = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if 2 < self.df[col].unique().size < (len(self.df)/10):
                    print(f'For {col} one hot encode')
                    self.df = pd.get_dummies(self.df, columns=[col], prefix=[col], drop_first=True)
                elif self.df[col].unique().size == 2:
                    print(f'For {col} replace {self.df[col].unique()[0]} with 1 and {self.df[col].unique()[1]} with 0')
                    self.df[col] = self.df[col].map({self.df[col].unique()[0]: 1, self.df[col].unique()[1]: 0})
                else:
                    col_to_remove.append(col)
        self.df = self.df.drop(col_to_remove, axis=1)


if __name__ == "__main__":
    # Create a Data object
    data = DataOrganizer('Titanic.csv', 'Survived')
    data.load_data()
    data.clean_data()
    data.set_data_type()
    data.print_stats()
    data.extract_features_from_text(method='second_word', col='Name', new_col='title', drop_col=True)
    data.extract_features_from_text(method='numbers', col='Ticket', new_col='ticket_numbers')
    data.extract_features_from_text(method='len', col='Ticket', new_col='ticket_len')
    data.one_hot_encoder()
    data.fill_missing_data(method='xgboost')
    data.imbalance_data(method='SMOTE')

    data.plot_data()
    data.print_stats()

    # Save data
    data.save_data()
