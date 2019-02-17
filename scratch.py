import sys
print('Python: {}'.format(sys.version))

import numpy as np
np.sqrt(10)

import matplotlib.pyplot as plt
Hist1 = (1,1,2,2,2,2,2,3,3,3,4,4,5)
plt.hist(Hist1, color= "black",bins=5)

z2 = (1,2,3,4,5,6,7,8,9)
z3 = (6,8,3,5,7,1,8,6,4)
plt.scatter(z2,z3, color='red', s=80, marker='x')
plt.title("My first Scatter Plot")

def CircleArea( Rad ):
    pi = 3.14
    area = pi * Rad * Rad
    print(area)

var1 = (2.3, 4.5, 6, 10, 1.3, 7.7)
sum1= np.sum(var1)

squares1 = np.square(var1)

squarert = np.sqrt(var1)

squareroot = np.sqrt(sum1)

meanval = np.mean(var1)

list1= [2,4, 'c', 2.714]
weekday = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
list_in_list = [1,2,3,['a','b','c']]
list_col = list('Hello')

tup1 = ('TN', 'TS', 'AP', 'UP', 'WB')
tup2 = (3,4,5,6)
tup3 = ('history', 'Science', 2017, 2018)
tup_null= ();

tup_new= ('MP',)
tup4 = tup1+tup_new

dict1 = {'phy':94, 'chem': 92, 'Math':98, 'Eng':90, 'CS':90}
dict2 = {'phy':96, 'chem': 98, 'Math':98, 'Eng':92, 'CS':94}

dict3 = {'phy':[94,96], 'chem':[92,98], 'Math':[98,98], 'Eng':[90,92], 'CS':[90,94]}
print(dict3)

marks = (34, 60 , 32, 65, 67, 78, 92)
if marks[3] >= 50:
    print('Passed')

    marks = (34, 60, 32, 65, 67, 78, 92)
    if marks[2] >= 50:
        print('Passed')
    else:
         print('Failed')

         marks = (34, 60, 32, 65, 67, 78, 92)
         if marks[2] >= 45:
             print('Passed')
         elif marks[2] <= 30:
             print('Failed')
         else:
             print('Retest')

             #Example For
             marks = (34, 60, 32, 65, 67, 78, 92)
             for i in range(len(marks)):
                 if marks[i] >= 50:
                     print('Passed')
                 else:
                     print('Failed')
#Example while
             marks = (34, 60, 32, 65, 67, 78, 92)
             i=0
             while(i < len(marks)):
                 if marks[i]>50:
                     print("Passed")

                 else:
                     print('Failed')
                 i+=1




#numpy libary
        import numpy as np
    np.array([1,4,2,5,3])

np.full((3,5),3.14)

#Generating Random Nos

np.arange(10,20,2)
np.linspace(10,20,5)
np.random.random((4,4))

np.random.normal(3,1,(3,3))
np.random.randint(0,12,8)
np.random.randint(0,12,(3,4))

np.eye(3)

x = np.random.randint(10, size = (3,4,5))
print(x.shape, x.ndim, x.size)

np.zeros((3,3));
np.random.randint(10, size =(3,4,5))

z = np.random.randint(0,10,(3,3));print(z)
z[1,2]


y = np.random.randint(0,10,6)
y[:3]

z = np. random.randint(0,10,(3,3))
z[:2,:3]
z[:2,::3]
z[::-1,::-1]
z[:,2]
z[2,:]

np.sort(y)
np.sort(z,axis=1)
np.sort(z,axis=0)

import numpy as np
y = np.random.randint(0,10,(3,2))
y.reshape(2,3)

y.reshape(3,2)

y = np.random.randint(0,10,(3,3))
y.reshape(-1,1)