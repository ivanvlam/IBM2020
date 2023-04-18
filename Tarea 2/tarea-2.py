#!/usr/bin/env python
# coding: utf-8

# # Tarea 2 - Introducción a la Biomecánica
# ### Pregunta 2

# Importación de librerías

import numpy as np
import matplotlib.pyplot as plt

# Definición de parámetros

global L
global h
global R
global mu
global lambda_

L = 10
h = 1
R = 10
mu = 1
lambda_ = 10

# Creación de la matriz

B_x = np.linspace(0, L, 500)
B_y = np.linspace(-h / 2, h / 2, 100)

BB_x, BB_y = np.meshgrid(B_x, B_y)

# Ploteo sin deformación

fig = plt.figure(figsize=(8,4))
plt.fill(BB_x, BB_y, color='#ABABAB')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.plot()

# Función de deformación

def phi(x, y):
    xx = (y + R) * np.sin(np.pi * x / L)
    yy = (y - (y + R) * (1 - np.cos(np.pi * x / L)))

    return xx, yy

# Creación de malla deformada

BB_XX, BB_YY = phi(BB_x,BB_y)

# Ploteo con deformación

fig = plt.figure(figsize=(6, 6))
plt.plot(BB_x, BB_y, color='#ABABAB')
plt.plot(BB_XX, BB_YY, color='#2E2E2E', linewidth=2)
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.plot()

# Definición de tensor gradiente de deformación

F = np.zeros((*BB_x.shape, 2, 2))
F.shape

# Instanciación de tensores P y sigma

P = np.zeros(BB_x.shape)
P_11 = np.zeros(BB_x.shape)
P_12 = np.zeros(BB_x.shape)
P_21 = np.zeros(BB_x.shape)
P_22 = np.zeros(BB_x.shape)

s = np.zeros(BB_x.shape)
s_11 = np.zeros(BB_x.shape)
s_12 = np.zeros(BB_x.shape)
s_21 = np.zeros(BB_x.shape)
s_22 = np.zeros(BB_x.shape)

T_norm = np.zeros(BB_x.shape)

t_norm = np.zeros(BB_x.shape)

# Inverse transpose matrix

def inv_transp(A):
    return np.linalg.inv(A.transpose())

# Vector tangente

vec_N = np.array([1, 0])

# Cálculo de tensores de deformación

for i in range(len(BB_x[:, 0])):
    for j in range(len(BB_y[0, :])):
        local_x = BB_x[i, j]
        local_y = BB_y[i, j]

        # Gradiente de deformación
        local_F = np.array([
            [np.pi / L * (local_y + R) * np.cos(np.pi * local_x / L), \
              np.sin(np.pi * local_x / L)],
            [-np.pi / L * (local_y + R) * np.sin(np.pi * local_x / L), \
              np.cos(np.pi * local_x / L)]
        ])

        # Transpuesta de F
        local_F_t = local_F.transpose()

        # Inversa de la transpuesta de F
        local_F_it = inv_transp(local_F)

        # Determinante de F
        J = abs(np.pi / L * (local_y + R))
        
        # Logaritmo de J
        try:
            log_J = np.log(J)
        except:
            log_J = 1
        
        # Tensor de Piola Kirchoff
        local_P = mu * (local_F - local_F_it) + lambda_ * log_J * local_F_it

        P_11[i, j] = local_P[0, 0]
        P_12[i, j] = local_P[0, 1]
        P_21[i, j] = local_P[1, 0]
        P_22[i, j] = local_P[1, 1]

        # Tensor de Cauchy
        local_s = local_P @ local_F_t / J

        s_11[i, j] = local_s[0, 0]
        s_12[i, j] = local_s[0, 1]
        s_21[i, j] = local_s[1, 0]
        s_22[i, j] = local_s[1, 1]

        # Tracción material
        local_T = local_P @ vec_N

        T_norm_local = np.linalg.norm(local_T)
        T_norm[i, j] = T_norm_local

        # Tracción espacial
        vec_n = np.array([np.sin(np.pi * local_x / L), np.cos(np.pi * local_x / L)])
        local_t = local_s @ vec_n

        t_norm_local = np.linalg.norm(local_t)
        t_norm[i, j] = t_norm_local

# Plot P_11

plt.figure(figsize=(8, 4))
fig = plt.pcolor(BB_x, BB_y, P_11, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$P_{11}$')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.show()

# Plot P_12

plt.figure(figsize=(8, 4))
fig = plt.pcolor(BB_x, BB_y, P_12, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$P_{12}$')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.show()

# Plot P_22

plt.figure(figsize=(8, 4))
fig = plt.pcolor(BB_x, BB_y, P_22, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$P_{22}$')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.show()

# Plot s_11

plt.figure(figsize=(6, 6))
fig = plt.pcolor(BB_XX, BB_YY, s_11, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$\sigma_{11}$')
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.show()

# Plot s_12

plt.figure(figsize=(6, 6))
fig = plt.pcolor(BB_XX, BB_YY, s_12, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$\sigma_{12}$')
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.show()

# Plot s_22

plt.figure(figsize=(6, 6))
fig = plt.pcolor(BB_XX, BB_YY, s_22, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$\sigma_{22}$')
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.show()

# Plot ||T||

plt.figure(figsize=(8, 4))
fig = plt.pcolor(BB_x, BB_y, T_norm, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$||T||$')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.show()

# Creación de malla para tracción material

B_x_r = np.linspace(0, L, 20)
B_y_r = np.linspace(-h / 2, h / 2, 3)

# Creación de malla para tracción espacial

BB_x_r, BB_y_r = np.meshgrid(B_x_r, B_y_r)
BB_XX_r, BB_YY_r = phi(BB_x_r,BB_y_r)

# Creación de vectores de tracción material y espacial

T = np.zeros((2, *BB_x_r.shape))
t = np.zeros((2, *BB_x_r.shape))

# Cálculo de tensores de deformación

for i in range(len(BB_x_r[:, 0])):
    for j in range(len(BB_y_r[0, :])):
        local_x = BB_x_r[i, j]
        local_y = BB_y_r[i, j]

        # Gradiente de deformación
        local_F = np.array([
            [np.pi / L * (local_y + R) * np.cos(np.pi * local_x / L), \
              np.sin(np.pi * local_x / L)],
            [-np.pi / L * (local_y + R) * np.sin(np.pi * local_x / L), \
              np.cos(np.pi * local_x / L)]
        ])

        # Inversa de la transpuesta de F
        local_F_it = inv_transp(local_F)

        # Determinante de F
        J = abs(np.pi / L * (local_y + R))
        
        # Logaritmo de J
        try:
            log_J = np.log(J)
        except:
            log_J = 1
        
        # Tensor de Piola Kirchoff
        local_P = mu * (local_F - local_F_it) + lambda_ * log_J * local_F_it

        # Tracción material
        local_T = local_P @ vec_N

        T_norm_local = np.linalg.norm(local_T)
        T[0, i, j], T[1, i, j] = local_T[0] / T_norm_local, local_T[1] / T_norm_local

# Plot tracción material

plt.figure(figsize=(8, 4))
plt.quiver(BB_x_r, BB_y_r, *T, color = 'k')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.show()

# Plot tracción material sobre configuración no deformada

plt.figure(figsize=(8, 4))
fig = plt.pcolor(BB_x, BB_y, T_norm, cmap='autumn_r', shading='auto')
plt.quiver(BB_x_r, BB_y_r, *T, color = 'k')
plt.colorbar(fig, label = r'$||T||$')
plt.axis('equal')
plt.xlim(-0.1, 10.1)
plt.show()

# Plot ||t||

plt.figure(figsize=(6, 6))
fig = plt.pcolor(BB_XX, BB_YY, t_norm, cmap='autumn_r', shading='auto')
plt.colorbar(fig, label = r'$||t||$')
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.show()

# Cálculo de tensores de deformación

for i in range(len(BB_x_r[:, 0])):
    for j in range(len(BB_y_r[0, :])):
        local_x = BB_x_r[i, j]
        local_y = BB_y_r[i, j]

        # Gradiente de deformación
        local_F = np.array([
            [np.pi / L * (local_y + R) * np.cos(np.pi * local_x / L), \
              np.sin(np.pi * local_x / L)],
            [-np.pi / L * (local_y + R) * np.sin(np.pi * local_x / L), \
              np.cos(np.pi * local_x / L)]
        ])

        # Transpuesta de F
        local_F_t = local_F.transpose()

        # Inversa de la transpuesta de F
        local_F_it = inv_transp(local_F)

        # Determinante de F
        J = abs(np.pi / L * (local_y + R))
        
        # Logaritmo de J
        try:
            log_J = np.log(J)
        except:
            log_J = 1
        
        # Tensor de Piola Kirchoff
        local_P = mu * (local_F - local_F_it) + lambda_ * log_J * local_F_it

        # Tensor de Cauchy
        local_s = local_P @ local_F_t / J

        # Tracción espacial
        vec_n = np.array([np.sin(np.pi * local_x / L), np.cos(np.pi * local_x / L)])
        local_t = local_s @ vec_n

        t_norm_local = np.linalg.norm(local_t)
        t_norm[i, j] = t_norm_local
        t[0, i, j], t[1, i, j] = local_t[0] / t_norm_local, local_t[1] / t_norm_local

# Plot tracción espacial

plt.figure(figsize=(6, 6))
plt.quiver(BB_XX_r, BB_YY_r, *t, color = 'k')
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.show()

# Plot tracción espacial sobre configuración deformada

plt.figure(figsize=(6, 6))
fig = plt.pcolor(BB_XX, BB_YY, t_norm, cmap='autumn_r', shading='auto')
plt.quiver(BB_XX_r, BB_YY_r, *t, color = 'k')
plt.colorbar(fig, label = r'$||t||$')
plt.axis('equal')
plt.ylim(-21.5, 1.5)
plt.show()


# #### Hecho con :heart: por Iván Vergara Lam
