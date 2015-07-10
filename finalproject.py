

import performance
import ml_predict_damien as ml_predict
import argparse

# code for testing the performance
performance_test = performance.Performance()
performance_test.run()

parser = argparse.ArgumentParser(description="Predict a robot movements in next 2 seconds")
parser.add_argument("--training", help = "file pattern to training data", default = "./inputs/training_data.txt")
parser.add_argument("--input", help = "file pattern to input data", default = "./inputs/test01.txt")
parser.add_argument("--output", help = "file pattern to output data", default = "./output.txt")


# KNN ML predictor performs much better. Therefore, we use it for our final
# output.
args = parser.parse_args()
predictor = ml_predict.MLPredictor(20, args.training)
predictions = predictor.make_prediction(args.input)[0]
print predictions

# output the predictions
predictions.to_csv(args.output)
#with open(args.output, 'w') as f:
 #   for point in predictions:
 #       f.write(str(point[0])+','+str(point[1])+'\n')