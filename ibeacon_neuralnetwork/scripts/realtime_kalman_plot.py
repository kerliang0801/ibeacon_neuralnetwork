import time
from beacontools import BeaconScanner, IBeaconFilter
from kalman_filter_linear import KalmanFilterLinear
import numpy as np


#rssi_value = 0
kalman =[]

A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.008])
xhat = np.matrix([3])
P    = np.matrix([1])

def callback(bt_addr, rssi, packet, additional_info):
    #print("<%s, %d>" % (additional_info, rssi))
    print(additional_info)
    #rssi_value = rssi

def main():
    while True:
        # scan for all iBeacon advertisements from beacons with the specified uui
        filter = KalmanFilterLinear(A,B,H,xhat,P,Q,R)
        scanner = BeaconScanner(callback)
        scanner.start()
        time.sleep(1)
        #print(uuid)
        #print(rssi_value)
        print("\n")
        scanner.stop()


#kalman.append(filter.GetCurrentState()[0,0])
#filter.Step(np.matrix([0]),np.matrix([rssi))


if __name__ == "__main__":
    main()







#time.sleep(5)
#scanner.stop()