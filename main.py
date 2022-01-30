import pandas as p
import csv
import plotly.express as pe
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


data = p.read_csv("data.csv")

scoreList = data["Score"].tolist()
acceptedList = data["Accepted"].tolist()

plot = pe.scatter(x = scoreList , y= acceptedList)
plot.show()

#-------------------------------- using algorithm solving y=mx+c ------------------------------------

scoreArray = np.array(scoreList)
acceptedArray = np.array(acceptedList)

m , c  = np.polyfit(scoreArray , acceptedArray , 1)

y = []

for x in scoreArray:
    yValue = m*x+c
    y.append(yValue)

plot = pe.scatter(x = scoreList , y= acceptedList)


plot.update_layout(shapes = [
    dict(
        type = 'line',
        x0 = min(scoreArray),x1 = max(scoreArray),
        y0 = min(y),y1 = max(y)
    )
])

plot.show()

X = np.reshape(scoreList , (len(scoreList)) , 1)                      
Y = np.reshape(acceptedList , (len(acceptedList)) , 1)


lr = LogisticRegression()
lr.fit(X,Y)

plt.figure()
plt.scatter(X.ravel() , Y , color = "black" , zorder = 20)


def model(x):
    return 1/(1+np.exp(-x))

Xtest = np.linspace(0,100,200)

chances = model(Xtest*lr.coef_ + lr.intercept_).ravel()

plt.plot(Xtest , chances , color="red" , linewidth=3)

plt.axhline(y=0 , color='k' , linestyle = '-' )

plt.axhline(y=1 , color='k' , linestyle = '-' )

plt.axhline(y=0.5 , color='b' , linestyle = '--' )


plt.axhline(x=Xtest[165] , color='b' , linestyle = '--')

plt.ylabel('y')

plt.xlabel('x') 

plt.xlim(75,85)

#plt.show()

#-------------------------- code to check about acceptance ----------------------
userScore = float(input("Enter your Marks --> "))

chances = model( userScore * lr.coef_ + lr.intercept_).ravel()[0]

if (chances <= 0.01):
    print("You will not get accepted")

elif (chances >= 1):
    print("You will get accepted") 

elif (chances <= 0.5):
    print("You might not get accepted")

else:
    print("You might get accepted")