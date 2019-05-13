import random
import numpy as np
import pylab
import pandas as pd
import seaborn as sns
import csv


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

numsteps = 1499

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


p1_c0= p1[0:1499,0]
p1_c1= p1[0:1499,1]
p1_c2= p1[0:1499,2]
p1_c3= p1[0:1499,3]

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


#must add a line of code here to get rid of first value
new_kalman = np.empty([1499,4])
#writting to csv file

for i in range(0,numsteps):
  new_kalman[i,0] = kalman[i]
  new_kalman[i,1] = kalman1[i]
  new_kalman[i,2] = kalman2[i]
  new_kalman[i,3] = kalman3[i]


np.savetxt('kalman_position1.csv',new_kalman.astype(int), delimiter=",")
print (new_kalman)


#print(kalman)


#pylab.title('RSSI Measurement with Kalman Filter')


sns.distplot(kalman[100:1499], label='mint', color='springgreen')
sns.distplot(kalman1[100:1499], label='blueberry', color='rebeccapurple')
sns.distplot(kalman2[100:1499], label='coconut', color='gold')
sns.distplot(kalman3[100:1499], label='icy', color='aqua')

#ax1 = pylab.subplot(2,2,1)
#ax1.plot(range(numsteps),p1_c0,'y',range(numsteps),kalman,'r')
#ax1.set_xlabel('Samples')
#ax1.set_ylabel('RSSI')
#ax1.legend(('measured','kalman'))

#ax2 = pylab.subplot(2,2,2)
#ax2.plot(range(numsteps),p1_c1,'y',range(numsteps),kalman,'r')
#ax2.set_xlabel('Samples')
#ax2.set_ylabel('RSSI')
#ax2.legend(('measured','kalman'))

#ax3 = pylab.subplot(2,2,3)
#ax3.plot(range(numsteps),p1_c2,'y',range(numsteps),kalman,'r')
#ax3.set_xlabel('Samples')
#ax3.set_ylabel('RSSI')
#ax3.legend(('measured','kalman'))

#ax4 = pylab.subplot(2,2,4)
#ax4.plot(range(numsteps),p1_c3,'y',range(numsteps),kalman,'r')
#ax4.set_xlabel('Samples')
#ax4.set_ylabel('RSSI')
#ax4.legend(('measured','kalman'))


pylab.show()


