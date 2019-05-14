import random
import numpy as np
import pylab
import pandas as pd
import seaborn as sns
import csv
from kalman_filter_linear import KalmanFilterLinear


numsteps = 1499

A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.008])
xhat = np.matrix([3])
P    = np.matrix([1])

filter = KalmanFilterLinear(A,B,H,xhat,P,Q,R)

position_1 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_1.csv', header=None, sep= ',')
position_2 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_2.csv', header=None, sep= ',')
position_3 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_3.csv', header=None, sep= ',')
position_4 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_4.csv', header=None, sep= ',')

p1 = np.array(position_1.values)
p2 = np.array(position_2.values)
p3 = np.array(position_3.values)
p4 = np.array(position_4.values)

p1_c0= p1[0:1499,0]
p1_c1= p1[0:1499,1]
p1_c2= p1[0:1499,2]
p1_c3= p1[0:1499,3]

p2_c0= p2[0:1499,0]
p2_c1= p2[0:1499,1]
p2_c2= p2[0:1499,2]
p2_c3= p2[0:1499,3]

p3_c0= p3[0:1499,0]
p3_c1= p3[0:1499,1]
p3_c2= p3[0:1499,2]
p3_c3= p3[0:1499,3]

p4_c0= p4[0:1499,0]
p4_c1= p4[0:1499,1]
p4_c2= p4[0:1499,2]
p4_c3= p4[0:1499,3]

kalman =[]
kalman1=[]
kalman2=[]
kalman3 =[]
kalman_p2=[]
kalman1_p2=[]
kalman2_p2=[]
kalman3_p2=[]
kalman_p3=[]
kalman1_p3=[]
kalman2_p3=[]
kalman3_p3=[]
kalman_p4=[]
kalman1_p4=[]
kalman2_p4=[]
kalman3_p4=[]



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

#_______________________________________________________________
for i in range(numsteps):
    kalman_p2.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p2_c0[i]]))

for i in range(numsteps):
    kalman1_p2.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p2_c1[i]]))

for i in range(numsteps):
    kalman2_p2.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p2_c2[i]]))

for i in range(numsteps):
    kalman3_p2.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p2_c3[i]]))

#______________________________________________________________    
for i in range(numsteps):
    kalman_p3.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p3_c0[i]]))

for i in range(numsteps):
    kalman1_p3.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p3_c1[i]]))

for i in range(numsteps):
    kalman2_p3.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p3_c2[i]]))

for i in range(numsteps):
    kalman3_p3.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p3_c3[i]]))

#_______________________________________________________________
for i in range(numsteps):
    kalman_p4.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p4_c0[i]]))

for i in range(numsteps):
    kalman1_p4.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p4_c1[i]]))

for i in range(numsteps):
    kalman2_p4.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p4_c2[i]]))

for i in range(numsteps):
    kalman3_p4.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([p4_c3[i]]))



#must add a line of code here to get rid of first value
new_kalman = np.empty([1499,5])
new_kalman1 = np.empty([1499,5])
new_kalman2 = np.empty([1499,5])
new_kalman3 = np.empty([1499,5])
#writting to csv file

for i in range(0,numsteps):
  new_kalman[i,0] = kalman[i]
  new_kalman[i,1] = kalman1[i]
  new_kalman[i,2] = kalman2[i]
  new_kalman[i,3] = kalman3[i]

for i in range(0,numsteps):
  new_kalman[i,4]=1

for i in range(0,numsteps):
  new_kalman1[i,0] = kalman_p2[i]
  new_kalman1[i,1] = kalman1_p2[i]
  new_kalman1[i,2] = kalman2_p2[i]
  new_kalman1[i,3] = kalman3_p2[i]

for i in range(0,numsteps):
  new_kalman1[i,4]=2

for i in range(0,numsteps):
  new_kalman2[i,0] = kalman_p3[i]
  new_kalman2[i,1] = kalman1_p3[i]
  new_kalman2[i,2] = kalman2_p3[i]
  new_kalman2[i,3] = kalman3_p3[i]

for i in range(0,numsteps):
  new_kalman2[i,4]=3

for i in range(0,numsteps):
  new_kalman3[i,0] = kalman_p4[i]
  new_kalman3[i,1] = kalman1_p4[i]
  new_kalman3[i,2] = kalman2_p4[i]
  new_kalman3[i,3] = kalman3_p4[i]

for i in range(0,numsteps):
  new_kalman3[i,4]=4


np.savetxt('kalman_position1.csv',new_kalman.astype(int), delimiter=",")
np.savetxt('kalman_position2.csv',new_kalman1.astype(int), delimiter=",")
np.savetxt('kalman_position3.csv',new_kalman2.astype(int), delimiter=",")
np.savetxt('kalman_position4.csv',new_kalman3.astype(int), delimiter=",")




