

import math
import numpy

############################################################
### Model 1: Use the average speed and angle in a window ###
############################################################

class NaivePredictor:
    """
    The naive predictor class.
    This predictor takes the training data and a prediction horizon:
        (1) compute the boundary of the box
        (2) for each point in the prediction horizon
            (2.1) takes prior 20 frames.
            (2.2) compute the average velocity, angle, and turning angle in the last 20 frames.
            (2.3) predict the point.
            (2.4) add the predicited point to data (for future prediction)
    (2.3) prediction step consists of 2 steps: (1) predict the point basd on the robot's current position,
    current angle, and current velocity; (2) If the robot is going to hit the wall, bounce the
    robot back based on law of refleciton (incident angle =  reflecting angle).
    """

    # initialize the predictor
    def __init__(self, window_size, horizon):
        self.window_size = window_size
        self.horizon = horizon

    # compute the average velocity in the past window of data
    def compute_velocity(self, past_data):
        velocities = []
        for i in range(0, len(past_data) - 1):
            velocities.append(math.sqrt((past_data[i][0] - past_data[i+1][0])**2 +\
                                        (past_data[i][1] - past_data[i+1][1])**2))
        return numpy.mean(velocities)

    # compute the average turning angle in the past window of data
    def compute_truning_angle(self, past_data):
        turning_angles = []
        past_angle = math.atan2(past_data[0][0] - past_data[1][0], past_data[0][1] - past_data[1][1])
        for i in range(1, len(past_data) - 1):
            current_angle = math.atan2(past_data[i+1][0] - past_data[i][0], past_data[i+1][1] - past_data[i][1])
            turning_angles.append(current_angle - past_angle)
            past_angle = current_angle
        return numpy.mean(turning_angles)

    # compute the last angle the car goes in last frame
    def compute_angle(self, past_data):
        x = past_data[-1]
        y = past_data[-2]
        return math.atan2(x[0] - y[0], x[1] - y[1])

    def next_position(self, distance, angle, turning_angle, prev_position):
        delta_x = distance * math.sin(angle + turning_angle)
        delta_y = distance * math.cos(angle + turning_angle)
        return [int(prev_position[0] + delta_x), int(prev_position[1] + delta_y)]

    # Check whether we reach a boundary, if so, return
    # the new angle and turning angle
    def check_boundary(self, position, angle, boundaries):
        if position[0] < boundaries['left']:
            if angle > math.pi / 4 and angle < math.pi / 2:
                return math.pi / 2 - angle
            if angle > math.pi / 2 and angle < math.pi / 4 * 3:
                return math.pi / 2 * 3 - angle
        if position[0] > boundaries['right']:
            if angle < math.pi / 4:
                return math.pi / 2 - angle
            if angle > math.pi / 4 * 3:
                return math.pi / 2 * 3 - angle
        if position[1] < boundaries['down']:
            if angle > math.pi / 4 * 3:
                return math.pi  - angle
            if angle > math.pi / 2 and angle < math.pi / 4 * 3:
                return math.pi  - angle
        if position[1] > boundaries['up']:
            if angle > math.pi / 4 and angle < math.pi / 2:
                return math.pi  - angle
            if angle < math.pi / 4:
                return math.pi  - angle
        return 0

    # given past data, we predict the future n frames
    def prediction(self, prediction_horizon, window_size, data):
        if len(data) < window_size:
            return []
        past_data = data[-window_size:]
        angle = self.compute_angle(past_data)
        turning_angle = self.compute_truning_angle(past_data)
        distance = self.compute_velocity(past_data)
        predictions = []
        boundaries = {"left" : min([x[0] for x in data]),\
                      "right" : max([x[0] for x in data]),\
                      "up" : max([x[1] for x in data]),\
                      "down" : min([x[1] for x in data])}
        count = 0
        current_position = past_data[-1]
        for i in range(0, prediction_horizon):
            prediction = self.next_position(distance, angle, turning_angle, current_position)
            new_angle = self.check_boundary(prediction, angle, boundaries)
            if new_angle != 0 and count <= 0:
                count = 5
                angle = new_angle
            count -= 1
            predictions.append(prediction)
            current_position = prediction
        return predictions

    # The main interface of the funciton. This returns the prediction over the horizon
    def make_prediction(self, input_file_path):
        self.data = [[int(x) for x in line.rstrip('\r\n').split(',')] for line in open(input_file_path)]
        training_data = self.data[0:len(self.data)-self.horizon]
        testing_data = self.data[-self.horizon:]
        prediction_data = self.prediction(self.horizon, self.window_size, training_data)
        return [prediction_data, testing_data]