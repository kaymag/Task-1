from operator import inv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm 

N= 600**2
#N=700**2
#N=800**2

#Модуль формирования системы уравнений МКЭ
a, b = -1, 1
h= 1/N
x_i = np.linspace(a, b, N)
alpha1 = 0
alpha2 = -1
beta1 = 1
beta2 = 1
alpha = 0
beta = 0

K_1 = np.zeros(N)
K_2 = np.zeros(N)
K_3 = np.zeros(N)
F = np.zeros(N)
for i in range (N):
    a_left = (2 + h*(i-0.5))/(3 + h*(i-0.5))
    a_right = (2 + h*(i+0.5))/(3 + h*(i+0.5))
    c_left = 1 + np.sin(h*(i-0.5))
    c_right = 1 + np.sin(h*(i+0.5))
    f_left = 1 - h*(i-0.5)
    f_right = 1 - h*(i+0.5)
    K_1[i] = -a_left/h + c_left*h/6
    K_2[i] = (a_left + a_right)/h + (c_left + c_right)*h/3
    K_3[i] = -a_right/h + c_right*h/6
    F[i] = (f_left + f_right)*h/2

#Решение системы методом прогонки
S = np.zeros(N)
T = np.zeros(N)
S[0] = alpha2/(h*alpha1 + alpha2)
T[0] = alpha*h/(h*alpha1 + alpha2)
for i in range (1,N):
    S[i] = -K_3[i-1]/(K_2[i-1] + K_1[i]*S[i])
    T[i] = (F[i] - K_1[i]*T[i])/(K_2[i] + K_1[i]*S[i])

U_fem = np.zeros(N)
U_fem[N-1] = (beta2*T[N-1] + h*beta)/(h*beta1 + beta2 - beta*S[N-1])
for i in range(N-2, -1, -1):
    U_fem[i] = S[i] * U_fem[i+1] + T[i]


plt.plot(x_i, U_fem, label = 'Numerical solution FEM')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Numerical Solution of PDE')
plt.legend()
plt.show()
