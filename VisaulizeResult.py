import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse

row = 100
filename = 'Results/AfterStateScore.csv'

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta, col):
    loss = []
    for l in dta[:, col]:
        loss.append(float(l))
    return loss

table = read_table(open(filename, 'r'))
col1 = get_train_loss(table, 1)
#Plot the training loss
plt.subplots()
plt.plot(range(row), col1, label= 'V(after-state) Reward')
plt.legend()
#plt.yscale('log')
#plt.ylim([0., 100])
plt.xlabel("episode(k)")
plt.ylabel('Reward')
plt.savefig('Results/AfterStateReward.png', dpi=400, bbox_inches='tight')
plt.close()
