from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing csv files
position_1 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data_augmentation/data/position_1.csv', header=None, sep= ',')
position_2 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data_augmentation/data/position_2.csv', header=None, sep= ',')
position_3 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data_augmentation/data/position_3.csv', header=None, sep= ',')
position_4 = pd.read_csv('/home/christopherlau/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data_augmentation/data/position_4.csv', header=None, sep= ',')

def main():


    # converting to numpy array
    position_1_np= np.array(position_1.values)
    position_2_np= np.array(position_2.values)
    position_3_np= np.array(position_3.values)
    position_4_np= np.array(position_4.values)

    ps1_x = position_1_np[:,0]
    ps1_y = position_1_np[:,1]
    ps1_z = position_1_np[:,2]

    ps2_x = position_2_np[:,0]
    ps2_y = position_2_np[:,1]
    ps2_z = position_2_np[:,2]
    
    ps3_x = position_3_np[:,0]
    ps3_y = position_3_np[:,1]
    ps3_z = position_3_np[:,2]

    ps4_x = position_4_np[:,0]
    ps4_y = position_4_np[:,1]
    ps4_z = position_4_np[:,2]

    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.set_xlim3d(-70,-50)
    #ax.set_ylim3d(-70,-50)
    #ax.set_zlim3d(-70,-50)
    #ax = plt.axes(projection='3d')
    
    ax.scatter3D(ps1_x, ps1_y, ps1_z, c=ps1_z, cmap='Greens')
    #ax.scatter3D(ps2_x, ps2_y, ps2_z, c=ps2_z, cmap='Blues')
    ax.scatter3D(ps3_x, ps3_y, ps3_z, c=ps3_z, cmap='Reds')
    #ax.scatter3D(ps4_x, ps4_y, ps4_z, c=ps4_z, cmap='Purples')

    #position 1 and position 2 quite near
    
    plt.show()





if __name__ == "__main__":
    main()