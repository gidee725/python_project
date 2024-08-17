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
