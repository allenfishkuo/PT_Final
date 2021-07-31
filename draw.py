# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:05:50 2021

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
font1 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 12
}

t1 = pd.read_csv("2013-2016NewNew.csv",usecols= ["loss","acc","winrate"])
t2 = pd.read_csv("CNN.csv",usecols = ["loss","acc"])
t3 = pd.read_csv("test_single.csv",usecols = ["loss","acc"])
print(t1.winrate)
print(t1)
"""
plt.figure()

plt.plot(t1["loss"] , label = "Multi-scale ResNet")
plt.plot(t2["loss"], label = "CNN")
plt.plot(t3["loss"], label = "Single-scale ResNet")
#plt.title("Multi-scale ResNet vs Single-scale ResNet vs Simple CNN")
plt.xlabel('Epoch',font1)
plt.ylabel('Training loss',font1)
plt.legend(prop = font1)

plt.tight_layout()
#plt.savefig('loss.pdf')
plt.show()        
plt.close()

"""

plt.figure()
#plt.title("Multi-scale ResNet vs Single-scale ResNet vs Simple CNN")

plt.xlabel('Epoch',font1)
plt.ylabel('Training accuracy',font1)
plt.plot(t1["acc"] , label = "Multi-scale ResNet")
plt.plot(t2["acc"] , label = "CNN")
plt.plot(t3["acc"] , label = "Single-scale ResNet")

#plt.savefig('Pair Trading ResNet loss(2016).png')
plt.legend(prop = font1)

plt.tight_layout()
plt.savefig('acc.pdf')
plt.show()        
plt.close()

"""
t1 = pd.read_csv("2015-2016NewNew_wegiht_decay005.csv",usecols= ["loss","acc","winrate"])
t2 = pd.read_csv("2013-2014_anasgrad_0120.csv")
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Train_accuracy')
plt.plot(t1["winrate"] , label = "winrate")
plt.plot(t2["winrate"] /5, label = "profit")
plt.legend()

plt.tight_layout()
plt.show()        
plt.close()
"""