#!/usr/bin/python
import time
from beacontools import BeaconScanner, IBeaconFilter
from kalman_filter_linear import KalmanFilterLinear
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading



### Variables for log - distance pathloss model
# RSSI = -10*n*log10(d/d0)+A0
# n - propagation exponent
# A0 - RSSI value at distance - d0 (d0 is set to 1)
# Hence, d = 10**((RSSI-A0)/-10n)
A0 = -77 #RSSI for 1M
n = 3 # n

#For plotting
plt.style.use('seaborn-darkgrid')
plt.ion()
kalman_arr = []
raw_arr = []
meters_arr = []
size = 50 # length of plot array

#Kalman filter properties
A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.0001])
xhat = np.matrix([-60])
P    = np.matrix([1])
filter = KalmanFilterLinear(A,B,H,xhat,P,Q,R)


class scanner_thread(threading.Thread):
    def run(self):
        while True:
            scanner = BeaconScanner(callback, device_filter=IBeaconFilter(uuid='636f3f8f-6491-4bee-95f7-d8cc64a863b5'))#"b9407f30-f5f8-466e-aff9-25556b57fe6d", major=30174, minor=5667)) #coconut2
            #my rpi beacon uuid 636f3f8f-6491-4bee-95f7-d8cc64a863b5
            scanner.start()
            time.sleep(5)
            scanner.stop()

def callback(bt_addr, rssi, packet, additional_info):
    
    #print(additional_info)
    kalman_value = filter.GetCurrentState()[0,0]
    print ("Kalman : " + str(kalman_value))
    meters = 10**((kalman_value - A0)/(-10*n))
    meters_arr.append(meters)

    kalman_arr.append(filter.GetCurrentState()[0,0])
    raw_arr.append(rssi)
    print("RSSI : "  + str(rssi))
    print("Packet : " + str(packet))

    if len(kalman_arr) > size:
        counter = len(kalman_arr) -size
        del kalman_arr[:counter]
    if len(raw_arr) > size:
        counter = len(raw_arr) - size
        del raw_arr[:counter]
    if len(meters_arr) >size:
        counter = len(meters_arr) - size
        del meters_arr[:counter]

    filter.Step(np.matrix([0]),np.matrix([rssi]))
    time.sleep(0.01)


class plot_thread(threading.Thread):
    def run(self):
        while True:
            p1 = plt.subplot(2,1,1)
            p1.set_title("Distance vs samples", weight='bold')
            p1.set_ylim(0,10)
            p1.set_xlim(0,50)
            p1.plot(meters_arr, 'k')
            
            p2 = plt.subplot(2,1,2)
            p2.set_title("Raw RSSI / Kalman Filtered vs samples", weight='bold')
            p2.set_ylim(-100,0)
            p2.set_xlim(0,50)
            p2.plot(raw_arr, 'yellow', label= "raw RSSI")
            p2.plot(kalman_arr, 'red', label= "Kalman Filtered")
            
            plt.legend()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()


def main():
    scanning = scanner_thread()
    plotting=  plot_thread()
    scanning.start()
    plotting.start()



if __name__ == "__main__":
    main()




