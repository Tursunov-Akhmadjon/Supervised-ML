import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import pandas as pd

class DataVisualizer:
    """
    A class to perform common data visualization tasks for exploratory data analysis.

    Attributes:
        df (pd.DataFrame): The input DataFrame to visualize.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataVisualizer with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to visualize.
        """
        self.df = df.copy()

    def plot_histogram(self, column: str, bins: int = 30) -> None:
        """
        Plots histogram for a numerical column.

        Args:
            column (str): Column name to plot.
            bins (int): Number of bins for the histogram. Default is 30.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column].dropna(), bins=bins, kde=True)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    def plot_boxplot(self, column: str) -> None:
        """
        Plots a boxplot for a numerical column.

        Args:
            column (str): Column name to plot.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=self.df[column])
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.show()

    def plot_correlation_heatmap(self) -> None:
        """
        Plots a heatmap of the correlation matrix for numerical columns.
        """
        corr = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_scatter(self, x_col: str, y_col: str, hue: Optional[str] = None) -> None:
        """
        Plots a scatter plot between two columns with optional hue.

        Args:
            x_col (str): Column name for x-axis.
            y_col (str): Column name for y-axis.
            hue (Optional[str]): Column name for color grouping. Default is None.
        """
        for col in [x_col, y_col] + ([hue] if hue else []):
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue)
        plt.title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.show()
