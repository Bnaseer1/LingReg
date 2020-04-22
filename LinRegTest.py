import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

values =\
       [
     [0, 25],
     [1, 36],
     [2, 35],
     [3, 54],
     [4, 72],
     [5, 83],
     ]

X = np.array(values)[:,0].reshape(-1,1)
y = np.array(values)[:,1].reshape(-1,1)


predict_x = [6]
predict_x = np.array(predict_x).reshape(-1,1)

reg = LinearRegression()
reg.fit(X,y)
predict_y = reg.predict(predict_x)
m = reg.coef_
c = reg.intercept_
print("Predicted y:\n", predict_y)

plt.title('Predict next number in sequence')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X,y,color='red')
plt.scatter(6 ,predict_y, color='blue')
new_y=[m*i+c for i in np.append(X, predict_x)]
new_y=np.array(new_y).reshape(-1,1)
plt.plot(np.append(X, predict_x),new_y, color='green')
plt.show()
