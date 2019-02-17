import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0,5,0.05)
f = 2 * np.pi*np.sin(2*np.pi*t)
plt.plot(t,f)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('First Plot')
plt.show()

g = 2 * np.pi*np.cos(2*np.pi*t)
plt.plot(t,g, color = 'green')

x = np.linspace(-2*np.pi, 2*np.pi, 200)
plt.plot(x, np.sin(x), color = 'blue', linewidth = 2.5, linestyle = '--', marker = '^', markeredgecolor = 'brown');
plt.plot(x, np.cos(x), color = 'red', linewidth = 2.5, linestyle = '-', marker = 'o', markeredgecolor = 'black', markerfacecolor='white');

z = np.random.normal(0,1,250)

plt.hist(z, color= 'purple', alpha = 0.8, orientation = 'vertical', label = 'First Histogram', histtype = 'bar', bins= 30, rwidth = 0.95)
plt.title('First Histogram')

import pandas as pd
birth = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\births.csv",sep=',')

a = birth['births']
x = birth.iloc[:,2]
plt.hist(a,color= 'purple', alpha = 0.8, orientation = 'vertical', label = 'First Histogram', histtype = 'bar', bins= 30, rwidth = 0.95 )

# Overlapping Histograms

X = np.random.normal(0,1,1000)
Y = np.random.normal(2,1,1000)

plt.hist(Y, color = 'maroon', alpha = 0.3, orientation='vertical', label='Norm2', histtype='bar', bins=50, rwidth = 0.95, edgecolor = 'white')
plt.hist(X, color = 'green', alpha = 0.7, orientation='vertical', label='Norm1', histtype='bar', bins=50, rwidth = 0.95, edgecolor = 'white')
plt.title(" First Histogram")
plt.legend()

### Pie Charts

FoodType = ['Grain Products', 'Added fats and Oils', 'Meat,Eggs and Nuts', 'Caloric Sweetners', 'Dairy', 'Fruit and Vegetables']
Intake = [582,589,525,367,275,205]
Daily_calories = pd.DataFrame(Intake,FoodType,columns=['Intake(gms)'])

type(Daily_calories)

plt.pie(Intake, labels=FoodType, autopct='%1.1f%%', startangle=90, explode=[0,0.3,0,0,0,0], shadow = True)
plt.legend(loc="best")
plt.axis('equal')
plt.title('Daily Calories per capita by food group, 2010', size=14)
Daily_calories.plot.pie(subplots = True, figsize= (10,10))

rng = np.random
a = rng.randn(50)
b = rng.randn(50)
plt.scatter(a,b,c=rng.randn(50), s=60)

plt.scatter(a,b,c=rng.randn(50), s=np.random.rand(50)*200,cmap='rainbow')
plt.suptitle("cmap = rainbow")
plt.colorbar()


Q= np.random.randn(100)
plt.boxplot(Q)

from mpl_toolkits import mplot3d
fig=plt.figure
ax=plt.axes(projection='3d')
x=np.linspace(-4,4,50)
y=np.linspace(-4,4,50)
x,y=np.meshgrid(x,y)
z=np.sin(np.sqrt(x**2+y**2+x*y))
ax.plot_surface(x,y,z,cmap='rainbow')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.suptitle('Surface')


plt.figure(figsize=(8,6), dpi= 80)
plt.subplot(111)

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
C,S = np.cos(x), np.sin(x)
plt.plot(x, C, color = "blue", linewidth = 1.0, linestyle = "-", label = 'cosine')
plt.plot(x, S, color = "green", linewidth = 1.0, linestyle = "-", label = 'sine')
plt.legend(loc = "lower left" , frameon = True)

#Set X limits & X ticks
plt.xlim(-3.0*np.pi,+3.0*np.pi)
plt.xticks(np.linspace(-3.0*np.pi,+3.0*np.pi,9,endpoint=True))

#Set Y limits & Y ticks
plt.ylim(-1.0,1.0)
plt.yticks(np.linspace(-1,1,5,endpoint=True))

fig, ax = plt.subplots(2,1) ## grid is split into 2 rows and 1 column
ax[0].plot(x, np.cos(x), '--', color = 'blue')
ax[1].plot(x, np.sin(x), '-', color = 'green')


#Generate Random data set
a1 = np.random.randn(50)
a2 = np.random.randn(50)

#split the graph panel into 2 rows & 2 cols
fig,axes = plt.subplots(2,2)

#4 in 1 subplots for the above
axes[0,0].scatter(a1,a2, c= np.random.randn(50), s = np.random.rand(50)*200, cmap ='rainbow')
axes[0,0].set_title("cmap=rainbow")

axes[0,1].scatter(a1,a2,c=np.random.randn(50), s= np.random.randn(50)*200, cmap = 'RdGy')
axes[0,1].set_title("cmap=RdGy")

axes[1,0].scatter(a1,a2,c=np.random.randn(50), s= np.random.randn(50)*200, cmap = 'jet')
axes[1,0].set_title("cmap=jet")

axes[1,1].scatter(a1,a2,c=np.random.randn(50), s= np.random.randn(50)*200, cmap = 'PuOr')
axes[1,1].set_title("cmap=PuOr")

import  seaborn as sns
x = np.random.normal(7,3,1000)
y = np.random.normal(15,4,1000)
sns.distplot(x)
sns.distplot(y)
plt.title('Distribution Plots with Seaborn')

sns.kdeplot(x,y, hue = 'blue')
plt.suptitle('2d Kernel Density Plots with Seaborn')
plt.xlabel('X')
plt.ylabel('Y')

sns.jointplot(x,y, kind ='kde')
plt.suptitle('Joint Plot with Seaborn')

sns.jointplot(x,y, kind ='reg')

#Pair Plots
iris = sns.load_dataset("iris")
iris.head()
sns.pairplot(iris, hue= 'species', size =2)

MtCars = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\mtcars.csv")
MtCars.head()
grid = sns.FacetGrid(MtCars, row ='am', col = 'cyl', margin_titles=True)
grid.map(plt.scatter,'disp','mpg')
grid2 = sns.factorplot('cyl','mpg',data = MtCars,kind ='box')

list1 = [1,'tharun','xyz']

list1[1][-1]

tup1 = (1,)
tup2 = (2,3)
print(tup1+tup2)

a=4.5
b=2
print(a//b)