import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10],
    'fit_prior': [True, False]
}

class NB:
    def __init__(self, name, data):
        self.name = name
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.model = None

    def _target_func(self, value):

        # variables = np.array(variable_values)

        # X_train_subset = self.X_train[:, variables.astype(int)]
        # X_test_subset = self.X_test[:, variables.astype(int)]
        # y_train_subset = self.y_train

        # model = BernoulliNB()
        # model.fit(X_train_subset, y_train_subset)
        # y_pred = model.predict(X_test_subset)

        # return -accuracy_score(self.y_test, y_pred)
        nb = BernoulliNB()
        nb.fit(self.X_train[:, value], self.y_train)
        accuracy = nb.score(self.X_test[:, value], self.y_test)

        return -accuracy
    
    def perform_SVD(self, n_components=4000):
        shape_before = self.X_train.shape

        svd = TruncatedSVD(n_components=n_components)
        svd.fit(self.X_train)
        self.X_train = svd.transform(self.X_train)
        self.X_test = svd.transform(self.X_test)

        explained_variance_ratio = svd.explained_variance_ratio_.sum()
        print(f"Variance Ratio {self.name}: ", explained_variance_ratio)
        print("Shape Before: ", shape_before)
        print("Shape After: ", self.X_train.shape)
    
    def train(self, param_grid=param_grid, export=False, export_link=None):

        print(f"{self.name} training started")
        # Ensure random aspect
        np.random.seed(42)
        
        model = BernoulliNB()
        grid_model = GridSearchCV(model, param_grid=param_grid, cv=5)
        grid_model.fit(self.X_train, self.y_train)

        print("Best Parameters: ", grid_model.best_params_)
        print("Model Runtime: ", grid_model.cv_results_['mean_fit_time'][grid_model.best_index_])

        if export is True:
            path = export_link if export_link is not None else ''

            with open(f'{path}{self.name}_model.pkl', 'wb') as file:
                pickle.dump(grid_model, file)

            print(f"{self.name} model exported")

        print(f"{self.name} training completed")

        self.model = grid_model

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained. Call the 'train' method first.")
        return self.model.predict(X)
    
    def evaluate(self):
        if self.model is None:
            raise RuntimeError("Model not trained. Call the 'train' method first.")
        
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        report = classification_report(self.y_test, y_pred_test, output_dict=True)
        training_accuracy = accuracy_score(y_pred_train, self.y_train)
        testing_accuracy = accuracy_score(y_pred_test, self.y_test)

        print(f"---- {self.name}'s evaluation ----")
        print("Training Accuracy:", training_accuracy)
        print("Testing Accuracy:", testing_accuracy)
        print("Classification Report:")

        report['training_accuracy'] = accuracy_score(y_pred_train, self.y_train)
        report['testing_accuracy'] = accuracy_score(y_pred_test, self.y_test)

        print(classification_report(self.y_test, y_pred_test))

        return report

