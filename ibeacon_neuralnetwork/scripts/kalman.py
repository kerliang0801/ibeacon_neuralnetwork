import random
import numpy as np
import pylab
import pandas as pd

position_1 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/unprocessed_data/position_1.csv', header=None, sep= ',')
# Implements a linear Kalman filter.
class KalmanFilterLinear:
  def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
    self.A = _A                      # State transition matrix.
    self.B = _B                      # Control matrix.
    self.H = _H                      # Observation matrix.
    self.current_state_estimate = _x # Initial state estimate.
    self.current_prob_estimate = _P  # Initial covariance estimate.
    self.Q = _Q                      # Estimated error in process.
    self.R = _R                      # Estimated error in measurements.
  def GetCurrentState(self):
    return self.current_state_estimate
  def Step(self,control_vector,measurement_vector):
    #---------------------------Prediction step-----------------------------
    predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_vector
    predicted_prob_estimate = (self.A * self.current_prob_estimate) * np.transpose(self.A) + self.Q
    #--------------------------Observation step-----------------------------
    innovation = measurement_vector - self.H*predicted_state_estimate
    innovation_covariance = self.H*predicted_prob_estimate*np.transpose(self.H) + self.R
    #-----------------------------Update step-------------------------------
    kalman_gain = predicted_prob_estimate * np.transpose(self.H) * np.linalg.inv(innovation_covariance)
    self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
    # We need the size of the matrix so we can make an identity matrix.
    size = self.current_prob_estimate.shape[0]
    # eye(n) = nxn identity matrix.
    self.current_prob_estimate = (np.eye(size)-kalman_gain*self.H)*predicted_prob_estimate

numsteps = 100

A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.008])
xhat = np.matrix([3])
P    = np.matrix([1])

filter = KalmanFilterLinear(A,B,H,xhat,P,Q,R)

position_1 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/unprocessed_data/position_1.csv', header=None, sep= ',')
p1 = np.array(position_1.values)
p1_c0= p1[0:100,0]
p1_c1= p1[0:100,1]
p1_c2= p1[0:100,2]
p1_c3= p1[0:100,3]

kalman =  []
kalman1 = []
kalman2 = []
kalman3 = []


for i in range(numsteps):
    kalman.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p1_c0[i]]))

for i in range(numsteps):
    kalman1.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p1_c1[i]]))

for i in range(numsteps):
    kalman2.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p1_c2[i]]))

for i in range(numsteps):
    kalman3.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p1_c3[i]]))

print(kalman)

pylab.plot(range(numsteps),p1_c0,'y',range(numsteps),kalman,'r')
pylab.xlabel('Time')
pylab.ylabel('RSSI')
pylab.title('RSSI Measurement with Kalman Filter')
pylab.legend(('measured','kalman'))
pylab.show()