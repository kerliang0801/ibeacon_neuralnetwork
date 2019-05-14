import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


position_1 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_1.csv', header=None, sep= ',')
position_2 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_2.csv', header=None, sep= ',')
position_3 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_3.csv', header=None, sep= ',')
position_4 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data/raw/position_4.csv', header=None, sep= ',')


def main():
    p1 = np.array(position_1.values)
    p2 = np.array(position_2.values)
    p3 = np.array(position_3.values)
    p4 = np.array(position_4.values)
    
    p1_c0 = p1[1:6000,0]
    p2_c0 = p2[1:6000,0]
    p3_c0 = p3[1:6000,0]
    p4_c0 = p4[1:6000,0]

    a1=plt.subplot(2,2,1)
    a1.set_title("position 1", weight='bold')
    plt.plot(p1_c0)

    a2=plt.subplot(2,2,2)
    a2.set_title("position 2", weight='bold')
    plt.plot(p2_c0)

    a3=plt.subplot(2,2,3)
    a3.set_title("position 3", weight='bold')
    plt.plot(p3_c0)

    a4=plt.subplot(2,2,4)
    a4.set_title("position 4", weight='bold')
    plt.plot(p4_c0)

    plt.show()



if __name__ == "__main__":
    main()

