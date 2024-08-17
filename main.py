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
    
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    """Class for visualizing the results using Matplotlib and Seaborn."""

    def __init__(self, train_df, ideal_funcs_df, test_results_df):
        self.train_df = train_df
        self.ideal_funcs_df = ideal_funcs_df
        self.test_results_df = test_results_df

    def plot_results(self, filename="results.png"):
        """Create and save the plot using Matplotlib and Seaborn."""
        plt.figure(figsize=(14, 8))
        
        # Plot training data
        for col in self.train_df.columns[1:]:
            sns.lineplot(x=self.train_df['x'], y=self.train_df[col], label=f"Train {col}", linewidth=2)
        
        # Plot ideal functions
        for func in self.ideal_funcs_df.columns[1:]:
            sns.lineplot(x=self.ideal_funcs_df['x'], y=self.ideal_funcs_df[func], label=f"Ideal {func}", linestyle='--', linewidth=2)
        
        # Plot test results
        plt.scatter(self.test_results_df['x'], self.test_results_df['y'], color='red', s=100, label="Test Mapped to Ideal")
        
        plt.title("Function Mapping and Deviations")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper left")
        
        # Save the plot to the current directory
        plt.savefig(filename)
        
        # Display the plot
        plt.show()
if __name__ == "__main__":
    # Setup SQLite engine
    engine = create_engine('sqlite:///functions.db')

    # Load data
    train_loader = TrainDataLoader('D:/kazi/Bin it/Dataset2/train.csv')
    train_loader.load_data()
    train_loader.save_to_db('training_data', engine)

    ideal_loader = IdealFunctionsLoader('D:/kazi/Bin it/Dataset2/ideal.csv')
    ideal_loader.load_data()
    ideal_loader.save_to_db('ideal_functions', engine)

    test_loader = TestDataLoader('D:/kazi/Bin it/Dataset2/test.csv')
    test_loader.load_data()
    test_loader.save_to_db('test_data', engine)

    # Map functions
    mapper = FunctionMapper(train_loader.dataframe, ideal_loader.dataframe)
    best_fits = mapper.find_best_fit()
    test_results_df = mapper.map_test_data(test_loader.dataframe, engine)

    # Print the best-fit functions
    print("Best Fit Functions:")
    for train_col, ideal_func in best_fits.items():
        print(f"Training Column '{train_col}' is best fit by Ideal Function '{ideal_func}'")

    # Visualize results
    viz = Visualization(train_loader.dataframe, ideal_loader.dataframe, test_results_df)
    viz.plot_results()

import unittest

class TestFunctionMapping(unittest.TestCase):

    def setUp(self):
        # Create sample data
        self.train_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [2, 4, 6],
            'y2': [3, 6, 9]
        })
        self.ideal_funcs_df = pd.DataFrame({
            'x': [1, 2, 3],
            'func1': [2.1, 4.1, 6.1],
            'func2': [3.1, 6.1, 9.1]
        })

        self.mapper = FunctionMapper(self.train_df, self.ideal_funcs_df)

    def test_least_squares(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 2.9])
        self.assertAlmostEqual(self.mapper.least_squares(y_true, y_pred), 0.01)

    def test_find_best_fit(self):
        best_fits = self.mapper.find_best_fit()
        self.assertEqual(len(best_fits), 2)  # Ensure two functions are selected

    def test_map_test_data(self):
        test_df = pd.DataFrame({'x': [1], 'y': [2.1]})
        test_results_df = self.mapper.map_test_data(test_df, engine)
        self.assertGreater(len(test_results_df), 0)  # Ensure test data is mapped

if __name__ == '__main__':
    unittest.main()