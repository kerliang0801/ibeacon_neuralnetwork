import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#import data
position_1 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork//data/raw/kalman_position_1.csv', header=None, sep= ',')
position_2 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork//data/raw/kalman_position_2.csv', header=None, sep= ',')
position_3 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork//data/raw/kalman_position_3.csv', header=None, sep= ',')
position_4 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork//data/raw/kalman_position_4.csv', header=None, sep= ',')


def main():

    fig = plt.figure()
    # converting to numpy array
    position_1_np= np.array(position_1.values)
    position_2_np= np.array(position_2.values)
    position_3_np= np.array(position_3.values)
    position_4_np= np.array(position_4.values)

    #extracting each column
    p1_column_0 = position_1_np[:,0]
    p1_column_1 = position_1_np[:,1]
    p1_column_2 = position_1_np[:,2]
    p1_column_3 = position_1_np[:,3]

    p2_column_0 = position_2_np[:,0]
    p2_column_1 = position_2_np[:,1]
    p2_column_2 = position_2_np[:,2]
    p2_column_3 = position_2_np[:,3]

    p3_column_0 = position_3_np[:,0]
    p3_column_1 = position_3_np[:,1]
    p3_column_2 = position_3_np[:,2]
    p3_column_3 = position_3_np[:,3]

    p4_column_0 = position_4_np[:,0]
    p4_column_1 = position_4_np[:,1]
    p4_column_2 = position_4_np[:,2]
    p4_column_3 = position_4_np[:,3]

    #visualising
    sns.set_style('darkgrid')

    p1=plt.subplot(2,2,1)
    p1.set_title("Position 1 (Kalman)", weight='bold')
    p1.set_xlabel("RSSI")
    p1.set_ylabel("Probability")
    sns.distplot(p1_column_0, label='mint', color='springgreen')
    sns.distplot(p1_column_1, label='blueberry', color='rebeccapurple')
    sns.distplot(p1_column_2, label='coconut', color='gold')
    sns.distplot(p1_column_3, label='icy', color='aqua')
    plt.legend()

    p2=plt.subplot(2,2,2)
    p2.set_title("Position 2 (Kalman)", weight='bold')
    p2.set_xlabel("RSSI")
    p2.set_ylabel("Probability")
    sns.distplot(p2_column_0, label='mint', color='springgreen')
    sns.distplot(p2_column_1, label='blueberry', color='rebeccapurple')
    sns.distplot(p2_column_2, label='coconut', color='gold')
    sns.distplot(p2_column_3, label='icy', color='aqua')
    plt.legend()

    p3=plt.subplot(2,2,3)
    p3.set_title("Position 3 (Kalman)", weight='bold')
    p3.set_xlabel("RSSI")
    p3.set_ylabel("Probability")
    sns.distplot(p3_column_0, label='mint', color='springgreen')
    sns.distplot(p3_column_1, label='blueberry', color='rebeccapurple')
    sns.distplot(p3_column_2, label='coconut', color='gold')
    sns.distplot(p3_column_3, label='icy', color='aqua')
    plt.legend()

    p4=plt.subplot(2,2,4)
    p4.set_title("Position 4 (Kalman)", weight='bold')
    p4.set_xlabel("RSSI")
    p4.set_ylabel("Probability")
    sns.distplot(p4_column_0, label='mint', color='springgreen')
    sns.distplot(p4_column_1, label='blueberry', color='rebeccapurple')
    sns.distplot(p4_column_2, label='coconut', color='gold')
    sns.distplot(p4_column_3, label='icy', color='aqua')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()


