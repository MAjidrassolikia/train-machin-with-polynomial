# با سلام استاد گرامی :
#در حالت پیشرفض پلی نومیال درجه 3 در نظر گرفته شده برای مشاهده نتیجه سایر درجه ها میبایست انها 
#را از حالت کامند خارج کرد در هر بار اجرا فقط یک درجه قابل نمایش است

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class LRTemplate(object):

    def __init__(self, in_shape, w=None, lr = 0.1):
        self.in_shape = in_shape
        self.w = w if w is not None else np.zeros((in_shape, 1))
        self.lr = lr

    def cost(self, x, y):
        y_hat = np.dot(x, self.w)
        error = y - y_hat
        cost = np.mean(error ** 2)
        return cost

    def train(self, x, y):
        y_hat = np.dot(x, self.w)
        error = y - y_hat
        gradient = np.dot(x.T, error) / len(y)
        self.w += self.lr * gradient
        cost = self.cost(x, y)
        return cost

    def predict(self, x):
        return np.dot(x, self.w)

    def params(self):
        return self.w


def validate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    val_error = np.mean((y_pred - y_val) ** 2)
    return val_error


def poly(x, degree):
    n, m = x.shape
    res = np.empty((n, m * degree + 1))
    res[:, 0] = np.ones(n)
    print(res.shape)
    for i in range(1, degree + 1):
        for j in range(m):
            res[:, (i - 1) * m + j] = np.power(x[:, j], i)
    return res


df = pd.read_csv('HW_data.csv')
shuffled_df = shuffle(df, random_state=42)

n = len(df)

train_size = int(n * 0.7)
val_size = int(n * 0.15)

x_plot = df['x'].values
y_plot = df['y'].values
x_plot_train = x_plot[:train_size].reshape(-1,1)
x_plot_val = x_plot[train_size:train_size+val_size].reshape(-1,1)
x_plot_test = x_plot[train_size + val_size:].reshape(-1,1)
x_plot = np.c_[np.ones_like(x_plot), x_plot]
y_plot_train = np.c_[y_plot]
X_train_poly2_plot = poly(x_plot_train, 2)
x_plot_val_poly2 = poly(x_plot_val, 2)
x_plot_test_poly2 = poly(x_plot_test, 2)
X_train_poly3_plot = poly(x_plot_train, 3)
x_plot_val_poly3 = poly(x_plot_val, 3)
x_plot_test_poly3 = poly(x_plot_test, 3)
X_train_poly7_plot = poly(x_plot_train, 7)
x_plot_val_poly7 = poly(x_plot_val, 7)
x_plot_test_poly7 = poly(x_plot_test, 7)


X = shuffled_df['x'].values
y = shuffled_df['y'].values

X_train = X[:train_size].reshape(-1, 1)
y_train = y[:train_size].reshape(-1, 1)
X_train_poly3 = poly(X_train, 3)
X_train_poly2 = poly(X_train, 2)
X_train_poly7 = poly(X_train, 7)
X_train = np.c_[np.ones_like(X_train), X_train]
y_train = np.c_[y_train]


X_val = X[train_size:train_size + val_size].reshape(-1, 1)
y_val = y[train_size:train_size + val_size].reshape(-1, 1)
X_val_poly3 = poly(X_val, 3)
X_val_poly2 = poly(X_val, 2)
X_val_poly7 = poly(X_val, 7)
X_val = np.c_[np.ones_like(X_val), X_val]
y_val = np.c_[y_val]

X_test = X[train_size + val_size:].reshape(-1, 1)
y_test = y[train_size + val_size:].reshape(-1, 1)
X_test_poly3 = poly(X_test, 3)
X_test_poly2 = poly(X_test, 2)
X_test_poly7 = poly(X_test, 7)
X_test = np.c_[np.ones_like(X_test), X_test]
y_test = np.c_[y_test]


# w_poly2 = np.zeros((3, 1), dtype=np.float64)
w_poly3 = np.zeros((4, 1), dtype=np.float64)
# w_poly7 = np.zeros((8, 1), dtype=np.float64)

# model_poly2 = LRTemplate(in_shape=2, w=w_poly2, lr=0.01)
model_poly3 = LRTemplate(in_shape=4, w=w_poly3, lr=0.01)
# model_poly7 = LRTemplate(in_shape=8, w=w_poly7, lr=0.000001)


for i in range(100000):
    # train_cost = model_poly2.train(X_train_poly2, y_train)
    train_cost = model_poly3.train(X_train_poly3, y_train)
    # train_cost = model_poly7.train(X_train_poly7, y_train)
    if i % 100 == 0:
        # print('Epoch:', i, 'Training cost (poly2):', train_cost_)
        print('Epoch:', i, 'Training cost (poly3):', train_cost)
        # print('Epoch:', i, 'Training cost (poly7):', train_cost)
    if train_cost < 0.1:
        break


# y_pred_poly2_train = model_poly2.predict(X_train_poly2_plot)
# y_pred_poly2_val = model_poly2.predict(x_plot_val_poly2)
# y_pred_poly2_test = model_poly2.predict(x_plot_test_poly2)

y_pred_poly3_train = model_poly3.predict(X_train_poly3_plot)
y_pred_poly3_val = model_poly3.predict(x_plot_val_poly3)
y_pred_poly3_test = model_poly3.predict(x_plot_test_poly3)

# y_pred_poly7_train = model_poly7.predict(X_train_poly7_plot)
# y_pred_poly7_val = model_poly7.predict(x_plot_test_poly7)
# y_pred_poly7_test = model_poly7.predict(x_plot_test_poly7)


# val_error_poly2 = validate(model_poly2, X_val_poly2, y_val)
val_error_poly3 = validate(model_poly3, X_val_poly3, y_val)
# val_error_poly7 = validate(model_poly7, X_val_poly7, y_val)

# print('Validation error (poly2):', val_error_poly2)
print('Validation error (poly3):', val_error_poly3)
# print('Validation error (poly7):', val_error_poly7)


plt.scatter(X_train[:, 1], y_train, c='blue', label='Training data')
# plt.plot(x_plot_train[:, 0], y_pred_poly2_train, color='red', label='Degree 2')
plt.plot(x_plot_train[:, 0], y_pred_poly3_train, color='green', label='Degree 3')
# plt.plot(x_plot_train[:, 0], y_pred_poly7_train, color='orange', label='Degree 7')

plt.scatter(X_val[:, 1], y_val, c='black', label='Validation data')
# plt.plot(x_plot_val[:, 0], y_pred_poly2_val, color='red')
plt.plot(x_plot_val[:, 0], y_pred_poly3_val, color='green')
# plt.plot(x_plot_val[:, 0], y_pred_poly7_val, color='blue')
plt.scatter(X_test[:, 1], y_test, c='red', label='test data')
# plt.plot(x_plot_train[:, 0],y_pred_poly2_test,color = 'green')
plt.plot(x_plot_test[:, 0], y_pred_poly3_test, color='green')
# plt.plot(x_plot_test[:,0],y_pred_poly7_test,color ='green')


plt.legend()
plt.show()
