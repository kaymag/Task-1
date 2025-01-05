from operator import inv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm 

N= 1000
#N=700**2
#N=800**2

#Модуль формирования системы уравнений МКЭ
a, b = -1, 1
h= (b-a)/N
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



#Решение системы методом Якоби
U_jak_new = np.zeros(N)

err = np.inf
iter = 0

#Нахождение сигма
u_k = np.zeros(N)
u_k1 = np.zeros(N)
err = np.inf

K = np.zeros((N-1, N-1))
K[0, 0] = K_2[0]
K[0, 1] = K_3[0]
for i in range(1, N-2):
    K[i][i-1] = K_1[i]
    K[i][i] = K_2[i]
    K[i][i+1] = K_3[i]
K[-1][-2] = K_1[-1]
K[-1][-1] = K_2[-1]

D = np.diag(np.diag(K))
D_inv = np.linalg.inv(D)

val, vec = np.linalg.eig(np.dot(D_inv, K))
lambda_min = val[0]
lambda_max = val[-1]
sigma = 2/(lambda_max + lambda_min)

#Нахождение Решение
while err > sigma:
    u_k1[0] = u_k[0] - sigma/K_2[0]*(K_2[0]*u_k[0] + K_3[0]*u_k[1] - F[0])

    for i in range(1, N-2):
        u_k1[i] = u_k[i] - sigma/K_2[i]*(K_1[i]*u_k[i-1] + K_2[i]*u_k[i] + K_3[i]*u_k[i+1] - F[i])

    u_k1[N-2] = u_k[N-2] - sigma/K_2[N-2]*(K_1[N-2]*u_k[N-3] + K_2[N-2]*u_k[N-2] - F[N-2])
    err = norm(u_k - u_k1)
    u_k = u_k1.copy()

for i in range(N):
    U_jak_new[i] = u_k[i-1]

plt.plot(x_i, U_jak_new, label = 'Numerical solution Jakoby')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Numerical Solution of PDE')
plt.legend()
plt.show()