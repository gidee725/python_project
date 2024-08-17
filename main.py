import pandas as pd
from sqlalchemy import create_engine
class DataLoadError(Exception):
    """Exception raised for errors in loading data."""
    pass

class MappingError(Exception):
    """Exception raised for errors during the mapping of test data."""
    pass


class BaseDataLoader:
    """Base class for loading data into a DataFrame."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataframe = None

    def load_data(self):
        """Load data from the given file path."""
        try:
            self.dataframe = pd.read_csv(self.filepath)
        except Exception as e:
            raise DataLoadError(f"Failed to load data from {self.filepath}: {e}")

    def save_to_db(self, table_name, engine):
        """Save the loaded data to an SQLAlchemy database."""
        if self.dataframe is not None:
            self.dataframe.to_sql(table_name, engine, if_exists='replace', index=False)
        else:
            raise DataLoadError("No data to save. Please load the data first.")
class TrainDataLoader(BaseDataLoader):
    """Class for loading and handling training data."""
    pass

class IdealFunctionsLoader(BaseDataLoader):
    """Class for loading and handling ideal functions data."""
    pass

class TestDataLoader(BaseDataLoader):
    """Class for loading and handling test data."""
    pass
import numpy as np

class FunctionMapper:
    """Class for mapping test data to the best-fit ideal functions."""

    def __init__(self, train_df, ideal_funcs_df):
        self.train_df = train_df
        self.ideal_funcs_df = ideal_funcs_df
        self.best_fits = {}

    def least_squares(self, y_true, y_pred):
        """Calculate the least squares error."""
        return np.sum((y_true - y_pred) ** 2)

    def find_best_fit(self):
        """Find the best-fit ideal functions for the training data."""
        for col in self.train_df.columns[1:]:  # Skip the 'x' column
            y_train = self.train_df[col].values
            errors = [(func, self.least_squares(y_train, self.ideal_funcs_df[func].values)) for func in self.ideal_funcs_df.columns[1:]]
            best_fit = min(errors, key=lambda x: x[1])
            self.best_fits[col] = best_fit[0]
        return self.best_fits

    def map_test_data(self, test_df, engine, threshold_factor=np.sqrt(2)):
        """Map the test data to the best-fit ideal functions."""
        results = []
        for i, row in test_df.iterrows():
            x_val, y_test = row['x'], row['y']
            deviations = []
            for col, func in self.best_fits.items():
                try:
                    y_pred = self.ideal_funcs_df.loc[self.ideal_funcs_df['x'] == x_val, func].values[0]
                except IndexError:
                    raise MappingError(f"No matching ideal function found for x={x_val}")
                deviation = abs(y_test - y_pred)
                deviations.append((func, deviation))
            best_fit, min_deviation = min(deviations, key=lambda x: x[1])
            if min_deviation <= threshold_factor * np.std([d[1] for d in deviations]):
                results.append({'x': x_val, 'y': y_test, 'delta_y': min_deviation, 'ideal_func': int(best_fit[-1])})
        
        # Convert results to DataFrame and store in database
        results_df = pd.DataFrame(results)
        results_df.to_sql('test_data_results', engine, if_exists='replace', index=False)
        return results_df