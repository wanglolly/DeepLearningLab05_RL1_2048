import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse

row = 100
filename = 'Results/AfterStatePercentage100k.csv'

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta, col):
    loss = []
    for r in range(dta.shape[0]):
        if len(dta[r]) > col:
            loss.append(float(dta[r][col]))
        else:
            loss.append(0)
    return loss

table = read_table(open(filename, 'r'))
col1 = get_train_loss(table, 2)
col2 = get_train_loss(table, 4)
col3 = get_train_loss(table, 6)
#col4 = get_train_loss(table, 8)
#Plot the training loss
plt.subplots()
plt.plot(range(row), col1, label= '2048')
plt.plot(range(row), col2, label= '4096')
plt.plot(range(row), col3, label= '8192')
#plt.plot(range(row), col4, label= '16384')
plt.legend()
#plt.yscale('log')
#plt.ylim([0., 100])
plt.xlabel("episode(k)")
plt.ylabel('percentage that terminated in this stage')
plt.savefig('Results/TDStatePecentage.png', dpi=400, bbox_inches='tight')
plt.close()
