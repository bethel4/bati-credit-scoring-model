import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
# Remove invalid import, WOETransformer does not exist in xverse

# Placeholder for WOE/IV feature engineering (to be implemented)
# from xverse.transformer import WOETransformer
# from woe import WoE

class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create aggregate features per customer and merge them back to the transaction level.
    """
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_features_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('customer_total_amount', 'sum'),
            ('customer_avg_amount', 'mean'),
            ('customer_transaction_count', 'count'),
            ('customer_std_amount', 'std')
        ]).reset_index()
        X = X.merge(agg_df, on=self.customer_id_col, how='left')
        return X

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract date/time features from TransactionStartTime.
    """
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X

class WOEFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply WOE encoding to categorical variables using xverse.
    """
    def __init__(self, categorical_cols, target_col='FraudResult'):
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.woe = None
        self.woe_cols = None

    def fit(self, X, y=None):
        from xverse.transformer import WOE
        self.woe = WOE()
        self.woe.fit(X, X[self.target_col])
        self.woe_cols = self.woe.features
        return self

    def transform(self, X):
        X = X.copy()
        X_woe = self.woe.transform(X)
        # Drop original categorical columns and add WOE columns
        X = X.drop(columns=self.categorical_cols)
        X = pd.concat([X.reset_index(drop=True), X_woe.reset_index(drop=True)], axis=1)
        return X

def build_preprocessing_pipeline():
    # Define columns
    categorical_cols = ['ProviderId', 'ProductCategory', 'ChannelId', 'ProductId']
    numerical_cols = ['Amount', 'Value', 'customer_total_amount', 'customer_avg_amount',
                     'customer_transaction_count', 'customer_std_amount',
                     'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']

    # Pipelines for different column types
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols),
        ('num', numerical_pipeline, numerical_cols)
    ])

    # Full pipeline
    pipeline = Pipeline([
        ('aggregate_features', AggregateCustomerFeatures()),
        ('datetime_features', DateTimeFeatureExtractor()),
        ('woe', WOEFeatureTransformer(categorical_cols=categorical_cols, target_col='FraudResult')),
        ('preprocessor', preprocessor)
    ])
    return pipeline

def process_data(input_path):
    df = pd.read_csv(input_path)
    pipeline = build_preprocessing_pipeline()
    processed = pipeline.fit_transform(df)
    return processed

if __name__ == "__main__":
    processed = process_data("data/raw/data.csv")
    print(processed)
