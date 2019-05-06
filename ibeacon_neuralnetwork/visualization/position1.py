import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#import data
position_1 = pd.read_csv('/home/christopher/projects/ibeacon_neuralnetwork/ibeacon_neuralnetwork/data_augmentation/data/position_4.csv', header=None, sep= ',')



def main():
    # converting to numpy array
    position_1_np= np.array(position_1.values)

    #extracting each column
    column_0 = position_1_np[:,0]
    column_1 = position_1_np[:,1]
    column_2 = position_1_np[:,2]
    column_3 = position_1_np[:,3]

    #visualising
    sns.set_style('darkgrid')
    sns.distplot(column_0, label='mint')#, color='#78C850')
    sns.distplot(column_1, label='blueberry')#, color='#705898')
    sns.distplot(column_2, label='coconut')
    sns.distplot(column_3, label='icy')#, color='#98D8D8')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


