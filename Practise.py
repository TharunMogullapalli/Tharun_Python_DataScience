from filecmp import cmp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
birth = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\births.csv",sep=',')
birth["gender"].head()
Y = birth[birth["gender"]=="girl"]
X = birth[birth["gender"]=="boy"]

plt.hist(X['births'],color = 'maroon', alpha = 0.3, orientation='vertical', label='Boy', histtype='bar', bins=30, rwidth = 0.95, edgecolor = 'black')
plt.hist(Y['births'],color = 'green', alpha = 0.5, orientation='vertical', label='Girl', histtype='bar', bins=30, rwidth = 0.95, edgecolor = 'white')

plt.title("Overlapping of Boy and Girl")
plt.legend(loc = "upper right")

print('pandas:{}'.format(pd.__version__))

c=list('Hel(ll0)')

a = [1,2,3,2]
b = ["abhilash","abhilash",'2']
b.count('abhilash')


x = np.linspace(2 * np.pi, 20 * np.pi, 1000)
y = np.linspace(3 * np.pi, 30 * np.pi, 1000)
z = np.linspace(4 * np.pi, 40 * np.pi, 1000)
p = np.linspace(5 * np.pi, 50 * np.pi, 1000)
q = np.linspace(6 * np.pi, 60 * np.pi, 1000)
r = np.linspace(7 * np.pi, 70 * np.pi, 1000)
C,S = np.cos(x), np.sin(x)
D,T = np.cos(y), np.sin(y)
E,U = np.cos(z), np.sin(z)
F,V = np.cos(p), np.sin(p)
plt.figure(2)
plt.plot(x,C, color = "blue", linewidth = 2.0, linestyle = "-")
plt.plot(x,S, color = "green", linewidth = 3.0, linestyle = "-")

plt.plot(y,D, color = "black", linewidth = 4.0, linestyle = '-' )
plt.plot(y,T, color = "pink", linewidth = 4.0, linestyle = '-' )

x = np.arange(0,1,0.01)
freqs = range(1,10)
nfreqs = len(freqs)
y = np.zeros((nfreqs,len(x)))
i=0
for f in freqs:
    y[i,:] = np.sin(f*2*np.pi*x)
    #z[i,:] = np.cos(f*2*np.pi*x)
    i += 1

#plt.figure()
plt.plot(x,y.T)
#plt.plot(x,z.T)
#plt.show()

#a = np.arange(0,1,0.01)
freqs = range(1,10)
nfreqs = len(freqs)
z = np.zeros((nfreqs,len(x)))
i=0
for f in freqs:
    #y[i,:] = np.sin(f*2*np.pi*a)
    z[i,:] = np.cos(f*2*np.pi*x)
    i += 1

plt.figure(1)
#plt.plot(x,y.T)
plt.plot(x,z.T)
plt.show()





