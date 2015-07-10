
import naive_predict
import ml_predict_damien as ml_predict
import math
import numpy

class Performance:
    """
    This class tests the performance of existing algorithms on the given 10
    testing files.
    """
    def __init__(self):
        self.naive_predictor = naive_predict.NaivePredictor(15, 60)
        self.ml_predictor = ml_predict.MLPredictor(20, "./inputs/training_data.txt")

    def prediction_accuracy(self, prediction, actual):
        accuracy = 0
        for x, y in zip(prediction, actual):
            accuracy += (x[0] - y[0])**2 + (x[1] - y[1])**2
        return math.sqrt(accuracy)
        
    def prediction_accuracy_pandas(self, prediction, actual):
        prediction = prediction.reset_index(drop=True)
        actual = actual.reset_index(drop=True)
        accuracy = ((prediction['x'] - actual['x'])**2 + (prediction['y'] - actual['y'])**2).sum(axis=0) 
        return math.sqrt(accuracy)

    def test_naive_predictor(self):
        input = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        input_files = ["./inputs/test"+x+".txt" for x in input]
        accuracies = []
        for file in input_files:
            [prediction, testing] = self.naive_predictor.make_prediction(file)
            accuracy = self.prediction_accuracy(prediction, testing)
            accuracies.append(accuracy)
        return accuracies

    def test_ml_predictor(self):
        input = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        input_files = ["./inputs/test"+x+".txt" for x in input]
        accuracies = []
        for file in input_files:
            [prediction, testing] = self.ml_predictor.make_prediction(file, 60)
            accuracy = self.prediction_accuracy_pandas(prediction, testing)
            accuracies.append(accuracy)
        return accuracies

    def print_stats(self, accuracies):
        print """
                Predictoin acurracy over 10 testing files for Naive Predictor:
                Average of Errors:            %d
                Standard deviation of Errors: %d
                Maximum Error:                %d
                Minimum Error:                %d
              """ % (numpy.mean(accuracies), numpy.std(accuracies),
                     numpy.min(accuracies), numpy.max(accuracies))

    def run(self):
        print "Performance testing for algorithms"
        print "Algorithm 1: Naive Predictor"
        accuracies = self.test_naive_predictor()
        self.print_stats(accuracies)
        print "Algorithm 2: KNN ML Predictor"
        accuracies = self.test_ml_predictor()
        self.print_stats(accuracies)
