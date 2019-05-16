#!/usr/bin/python

import time
from beacontools import BeaconScanner, IBeaconFilter
from kalman_filter_linear import KalmanFilterLinear
import numpy as np
import matplotlib.pyplot as plt


#Kalman filter properties
kalman =[]
A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.008])
xhat = np.matrix([-80])
P    = np.matrix([1])

filter = KalmanFilterLinear(A,B,H,xhat,P,Q,R)

def callback(bt_addr, rssi, packet, additional_info):
    #print(additional_info)
    kalman.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]),np.matrix([rssi]))
    print(kalman)
    plt.plot(kalman)
    plt.pause(0.1)

def main():
    
    while True:
       
        # scan for all iBeacon advertisements from beacons with the specified uuid, major and minor
        scanner = BeaconScanner(callback, device_filter=IBeaconFilter(uuid="b9407f30-f5f8-466e-aff9-25556b57fe6d", major=30174, minor=5667))
        scanner.start()
        time.sleep(20)
        scanner.stop()

    plt.show()


if __name__ == "__main__":
    main()




