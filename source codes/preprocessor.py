import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Optional

class DataPreprocessor:
    """
    A class for preprocessing datasets: handling nulls, encoding categorical columns, and scaling numerical features.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the preprocessor with a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to preprocess.
        """
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}

    def handle_nulls(self) -> pd.DataFrame:
        """
        Handles missing values in the DataFrame.

        - For object (categorical) columns: fills with mode.
        - For numeric columns: fills with mean.

        Returns:
            pd.DataFrame: DataFrame with no missing values.

        Example:
            >>> df = prep.handle_nulls()
        """
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        return self.df

    def encode_columns(self, columns: List[str], method: str = 'label') -> pd.DataFrame:
        """
        Encodes categorical columns using label or one-hot encoding.

        Args:
            columns (List[str]): List of column names to encode.
            method (str): Encoding method. Options:
                          - 'label' for LabelEncoder
                          - 'onehot' for OneHotEncoder or get_dummies (based on cardinality)

        Returns:
            pd.DataFrame: DataFrame with encoded columns.

        Raises:
            ValueError: If an unsupported encoding method is provided.

        Example:
            >>> df = prep.encode_columns(['Gender', 'Country'], method='onehot')
        """
        for col in columns:
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le

            elif method == 'onehot':
                cardinality = self.df[col].nunique()

                if cardinality <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)

                else:
                    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    transformed = ohe.fit_transform(self.df[[col]])
                    ohe_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out([col]))
                    self.df = pd.concat([self.df.drop(columns=[col]), ohe_df], axis=1)
                    self.encoders[col] = ohe

            else:
                raise ValueError("Unsupported encoding method. Use 'label' or 'onehot'.")

        return self.df

    def scale_columns(self, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Scales numerical columns using the specified method.

        Args:
            columns (List[str]): List of column names to scale.
            method (str): Scaling method. Options:
                          - 'standard': StandardScaler
                          - 'minmax': MinMaxScaler
                          - 'robust': RobustScaler

        Returns:
            pd.DataFrame: DataFrame with scaled columns.

        Raises:
            ValueError: If an unsupported scaling method is provided.

        Example:
            >>> df = prep.scale_columns(['Age', 'Income'], method='minmax')
        """
        scaler = None

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaling method. Choose from: 'standard', 'minmax', 'robust'.")

        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scalers[method] = scaler
        return self.df
