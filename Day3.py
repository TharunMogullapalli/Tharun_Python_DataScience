import numpy as np
y = np.random.randint(0,10,(3,2))

x = np.random.randint(0,5,(3,2))

np.vstack([y,y])
np.hstack([y,y])

y1,y2,y3 = np.split(y,[2,4])

# change axis
a1 = np.arange(4)
a2 = np.arange(4)[:,np.newaxis]

#Trigonometric functions

a = np.linspace(0,np.pi,4)
a_sin = np.sin(a)
a_cos = np.cos(a)

## Array Addition
z= np.random.randint(0,5,(4,3))
z.sum()
z.sum(axis = 1)
z.sum(axis = 0)

## Creating panda dataframe
import pandas as pd
dat1 = pd.Series([0.25,1.2,0.5,0.75,1.75,2])
print(dat1.values)
print(dat1.index)

data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'])
print(df)

data = {'Name':['Tom','Jack','Steve','Ricky'],'Age':[28,34,29,42]}
df1 = pd.DataFrame(data)
print(df1)


birth = pd.read_csv("D:\\MBA\\Term-6\\Python & SPSS\\births.csv",sep=',')
birth.head()
birth.head(10)

birth.describe()['births']
birth.sample(15)

birth[birth['births']>= 150000]

birth[(birth['births']>= 150000) & (birth['gender']=='girl')]

birth[:5]

x = birth.iloc[:,0:3]
x.head()

y = birth.iloc[:,[3]]
y.head()

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                    index = ['ohio','Colorado','Utah','New York'],
                    columns = ['a','b','c','d'])

print(data)
data.drop(['Colorado','ohio'])

df1 = pd.DataFrame({'Name':['Ak','Bk','Ck','Dk'],'Dept':['Quality','Prod','Fin','WH']})
df2 = pd.DataFrame({'Name':['Ak','Bk','Ck','Dk'],'Grade':['A','E','F','C']})
pd.merge(df1,df2)

pd.merge(df1,df2, on = 'Name')
pd.merge(df1, df2, left_on=0,right_on='A')

birth.groupby('gender')[['births']].mean()
birth.groupby('state')[['births']].mean()
birth.groupby(['state','gender'])[['births']].mean()
birth.groupby('gender')[['births']].sum()

url = 'http://bit.ly/imdbratings'
movies= pd.read_csv(url, sep = ',')
movies.head(10)
movies.describe()['star_rating']
movies.describe()

movies["title"].head()

type(movies["title"])
movies.groupby("title")[['star_rating']].mean()

