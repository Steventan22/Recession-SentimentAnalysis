import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'brute'],
    'metric': ['euclidean', 'manhattan']
}   

class KNN:
    def __init__(self, name, data):
        self.name = name
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.model = None

    def _target_func(self, variable_values):

        variables = np.array(variable_values)

        X_train_subset = self.X_train[:, variables.astype(int)]
        X_test_subset = self.X_test[:, variables.astype(int)]

        model = KNeighborsClassifier()
        model.fit(X_train_subset, self.y_train)
        y_pred = model.predict(X_test_subset)

        return -accuracy_score(self.y_test, y_pred)
    
    def train(self, param_grid=param_grid, export=False, export_link=None):

        print(f"{self.name} training started")
        #Ensure random aspect
        np.random.seed(42)
        
        model = KNeighborsClassifier()
        grid_model = GridSearchCV(model, param_grid=param_grid, cv=5)
        grid_model.fit(self.X_train, self.y_train)

        print("Best Paramaters: ", grid_model.best_params_)
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

        print(f"---- {self.name}'s evaluation----")
        print("Training Accuracy :", training_accuracy)
        print("Testing Accuracy :", testing_accuracy)
        print("Classification Report:")

        report['training_accuracy'] = accuracy_score(y_pred_train, self.y_train)
        report['testing_accuracy'] = accuracy_score(y_pred_test, self.y_test)

        print(classification_report(self.y_test, y_pred_test))

        return report


# what is cahat 
        