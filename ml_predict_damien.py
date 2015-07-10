

#import math
#import numpy
#import sys
#from sklearn import preprocessing
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

############################################################
### Model 1: Use the average speed and angle in a window ###
############################################################

class MLPredictor:
    """
    The ML predictor class.
    This predictor takes the training data and a prediction horizon:
        (1) normalize the (x,y) coordinate to [0, 1] scale, with 1 representing
        maximium x or y.
        (2) for each point, train a knn regressor based on preivous x points'
        positions, velocities, and angles. x = 0:61 so that we can predict up
        to 60 future points.
        (3) use the knn regressors to predict future points in testing data
    """

    # initialize the predictor
    def __init__(self, n_neighbors, training_file_path):
        # read training data
        self.prediction_horizon = 60
        self.read_training(training_file_path)
        # train the knn classifier
        self.knn_x = [0] * self.prediction_horizon
        self.knn_y = [0] * self.prediction_horizon
        for i in range(self.prediction_horizon):
            self.knn_x[i] = KNeighborsRegressor(n_neighbors, algorithm='kd_tree')
            self.knn_y[i] = KNeighborsRegressor(n_neighbors, algorithm='kd_tree')
            self.knn_x[i].fit(self.training_features, self.training_labels['next_x' + str(i + 1)])
            self.knn_y[i].fit(self.training_features, self.training_labels['next_y' + str(i + 1)])

    def predict(self, testing_features, horizon):
        predictions = pd.DataFrame(columns =  ['x', 'y'])    
        predictions['x'] = [self.knn_x[i].predict(testing_features)[0] for i in range(horizon)]
        predictions['y'] = [self.knn_y[i].predict(testing_features)[0] for i in range(horizon)]      
        return self.denormalize(predictions)

    # read training data
    def read_training(self, training_file_path):
        training_data = pd.read_csv(training_file_path, names =  ['x', 'y'], index_col=None)
        self.maxX , self.maxY = training_data.max(axis = 0)
        self.minX , self.minY = training_data.min(axis = 0)
        self.training_features = self.normalize(training_data)
        self.training_labels = pd.DataFrame(index= self.training_features.index)
        for i in range(self.prediction_horizon):
            x_name = 'next_x' + str(i + 1)
            y_name = 'next_y' + str(i + 1)
            self.training_labels[x_name] = self.training_features['x'].shift(-(i+1))
            self.training_labels[y_name] = self.training_features['y'].shift(-(i+1))
        self.training_labels = self.training_labels.head(len(self.training_features.index) - self.prediction_horizon)
        self.training_features = self.training_features.head(len(self.training_features.index) - self.prediction_horizon)
   

    def read_testing(self, testing_file_path, horizon):
        testing_data = pd.read_csv(testing_file_path, header=None, index_col=None)
        self.testing_data = self.normalize(testing_data.head(len(testing_data) - horizon))
        self.actual_data = self.normalize(testing_data.tail(horizon))

    def denormalize(self, data):
        data['x'] = data['x'] * (self.maxX - self.minX) + self.minX
        data['y'] = data['y'] * (self.maxY - self.minY) + self.minY
        return data

    # normalize (x,y) data to [0, 1]^2 domain and compute velocity and angle of each point (compared to
    # previous point
    def normalize(self, data):
        data.is_copy = False
        data['x'] = (data['x'] - self.minX) / (self.maxX - self.minX)
        data['y'] = (data['y'] - self.minY) / (self.maxY - self.minY)
#        data['vel_x'] = data['x'] - data['x'].shift(1)
#        data['vel_y'] = data['y'] - data['y'].shift(1)
#        data['acc_x'] = data['x'] - 2 * data['x'].shift(1) + data['x'].shift(2)
#        data['acc_y'] = data['y'] - 2 * data['y'].shift(1) + data['y'].shift(2)
        lag = 10
        for i in range(1,lag):
            x_name = 'x' + str(i)
            y_name = 'y' + str(i)
            data[x_name] = data['x'].shift(i)
            data[y_name] = data['y'].shift(i)
        return data.iloc[lag+1:len(data.index),:].reset_index(drop=True)

    # The main interface of the class.
    def make_prediction(self, testing_file_path, horizon = 0):
        testing_data = pd.read_csv(testing_file_path, names =  ['x', 'y'], index_col=None)
        testing_data_features = self.normalize(testing_data.head(len(testing_data) - horizon))
        actual_data = testing_data.tail(horizon)
        predictions = self.predict(testing_data_features.tail(1),self.prediction_horizon)
        return [predictions, actual_data]
        
        
        
        
        