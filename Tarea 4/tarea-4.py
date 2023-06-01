#!/usr/bin/env python
# coding: utf-8

# # Tarea 4 - Introducción a la Biomecánica
# ### Pregunta 1

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from scipy import integrate
from decimal import Decimal
import scipy.optimize as sp

# Paths
harmonic_path = join('data', 'harmonic.csv')
quasistatic_path = join('data', 'quasistatic.csv')

# Data windows
cycle_h = {             # Data windows for the harmonic experiment
    '0': 3935,          # Start of the experiment
    'f': 4136,          # End of the experiment
    '1': 3956           # End of the first oscillation
}
cycle_q = {             # Data windows for the quasistatic experiment
    '0': 3870,          # Start of the experiment
    'f': 6571           # End of the experiment
}

# Constants
g = 9.81                # Gravity
A = 7 * 10 ** (-6)      # Area of the tissue
L_0 = 2.2               # Initial length of the tissue

# Graphs
naranjo = '#F59A23'     
azul = '#010589'
rojo = '#E40C2B'
verde = '#00D300'
fig_size = (8, 5)

# Import data
harmonic = pd.read_csv(harmonic_path, skiprows=[0, 1, 2, 4])
quasistatic = pd.read_csv(quasistatic_path, skiprows=[0, 1, 2, 4])
quasistatic.head()

# Remove spaces from column names
quasistatic.rename(columns=lambda x: x.strip(), inplace=True)
harmonic.rename(columns=lambda x: x.strip(), inplace=True)

# Check data
quasistatic.loc[:, 'Elapsed Time']

# Data columns
time_q = quasistatic.loc[:, 'Elapsed Time']
disp_q = quasistatic.loc[:, 'Disp']
load_q = quasistatic.loc[:, 'Load 3']
load_q = load_q - load_q[cycle_q['0']]          # Normalized load

str_q = (load_q * g / 1000) / A                 # Stress
def_q = (disp_q - disp_q[cycle_q['0']]) / L_0   # Normalized deformation

time_h = harmonic.loc[:, 'Elapsed Time']
disp_h = harmonic.loc[:, 'Disp']
load_h = harmonic.loc[:, 'Load 3']
load_h = load_h - load_h[cycle_h['0']]          # Normalized load

str_h = (load_h * g / 1000) / A                 # Stress
def_h = (disp_h - disp_h[cycle_h['0']]) / L_0   # Normalized deformation

# Plot quasistatic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_q,
    disp_q,
    color=naranjo,
    linewidth=3
)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Displacement [mm]', fontsize=12)
# plt.title('Quasistatic (Displacement vs. Time)', fontsize=14)
plt.show()

# Plot quasistatic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_q[cycle_q['0']:cycle_q['f']],
    disp_q[cycle_q['0']:cycle_q['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Displacement [mm]', fontsize=12)
# plt.title('Quasistatic (Displacement vs. Time)', fontsize=14)
plt.show()

# Plot quasistatic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    disp_q,
    load_q,
    color=naranjo,
    linewidth=3
)
# End of the experiment
'''
plt.plot(
    [disp_q[cycle_q['f']] for _ in range(cycle_q['0'], cycle_q['f'])],
    load_q[cycle_q['0']:cycle_q['f']],
    color=naranjo,
    linewidth=3
)
'''
plt.xlabel('Displacement [mm]', fontsize=12)
plt.ylabel('Load [g]', fontsize=12)
# plt.title('Quasistatic (Load vs. Displacement)', fontsize=14)
plt.show()

# Plot quasistatic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    disp_q[cycle_q['0']:cycle_q['f']],
    load_q[cycle_q['0']:cycle_q['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Displacement [mm]', fontsize=12)
plt.ylabel('Load [g]', fontsize=12)
# plt.title('Quasistatic (Load vs. Displacement)', fontsize=14)
plt.show()

# Plot quasistatic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    def_q,
    str_q,
    color=naranjo,
    linewidth=3
)
plt.xlabel(r'Deformación $\epsilon$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
# plt.title('Quasistatic (Load vs. Displacement)', fontsize=14)
plt.show()

# Plot quasistatic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    def_q[cycle_q['0']:cycle_q['f']],
    str_q[cycle_q['0']:cycle_q['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel(r'Deformación $\epsilon$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
# plt.title('Quasistatic (Load vs. Displacement)', fontsize=14)
plt.show()


# #### Caso Armónico

# Plot harmonic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_h,
    disp_h,
    color=naranjo,
    linewidth=3
)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Displacement [mm]', fontsize=12)
# plt.title('Harmonic (Displacement vs. Time)', fontsize=14)
plt.show()

# Plot harmonic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_h[cycle_h['0']:cycle_h['f']],
    disp_h[cycle_h['0']:cycle_h['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Time [s]', fontsize=12)
plt.ylabel('Displacement [mm]', fontsize=12)
# plt.title('Harmonic (Displacement vs. Time)', fontsize=14)
plt.show()

# Plot harmonic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    disp_h,
    load_h,
    color=naranjo,
    linewidth=3
)
plt.xlabel('Displacement [mm]', fontsize=12)
plt.ylabel('Load [g]', fontsize=12)
# plt.title('Harmonic (Load vs. Displacement)', fontsize=14)
plt.show()

# Plot harmonic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    disp_h[cycle_h['0']:cycle_h['f']],
    load_h[cycle_h['0']:cycle_h['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Displacement [mm]', fontsize=12)
plt.ylabel('Load [g]', fontsize=12)
# plt.title('Harmonic (Load vs. Displacement)', fontsize=14)
plt.show()

# Plot harmonic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    def_h,
    str_h,
    color=naranjo,
    linewidth=3
)
plt.xlabel(r'Deformación $\epsilon$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()

# Plot harmonic data
fig = plt.figure(figsize=fig_size)
plt.plot(
    def_h[cycle_h['0']:cycle_h['f']],
    str_h[cycle_h['0']:cycle_h['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel(r'Deformación $\epsilon$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()


# ### Pregunta 2

# Stretch as a function of deformation
stretch_q =  def_q + 1

# #### Sólido Neohookeano incompresible

# Neo-Hookean model
def neo_hookean(c, lam):
    return c * (lam ** 2 - 1 / lam)

# Neo-Hookean model error
c_0 = 286

def error_nh(c, x, y):
    return abs(y - neo_hookean(c, x))

# Neo-Hookean model fit
nh_c_fit, nh_cov = sp.leastsq(
    func=error_nh,
    x0=c_0,
    args=(
        stretch_q[cycle_q['0']:cycle_q['f']],
        str_q[cycle_q['0']:cycle_q['f']]
    )
)
mu_nh = nh_c_fit
mu_nh

# Plot Neo-Hookean model fit for quasistatic data
fig = plt.figure(figsize=fig_size)
plt.scatter(
    stretch_q[cycle_q['0']:cycle_q['f']:180],
    str_q[cycle_q['0']:cycle_q['f']:180],
    color=naranjo,
    linewidth=3,
    label='Data'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    neo_hookean(nh_c_fit, stretch_q[cycle_q['0']:cycle_q['f']]),
    color=azul,
    linewidth=3,
    label='Modelo Neohookeano'
)
plt.legend()
plt.xlabel(r'Estiramiento $\lambda$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()


# #### Sólido de Voight

# Plot deformation as a function of time
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_q[cycle_q['0']:cycle_q['f']],
    def_q[cycle_q['0']:cycle_q['f']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel(r'Deformación $\epsilon$', fontsize=12)
plt.show()

# Slope of the deformation vs. time curve
global m
m = (def_q[cycle_q['f']] - def_q[cycle_q['0']]) / \
    (time_q[cycle_q['f']] - time_q[cycle_q['0']]) * 1e2
    #(cycle_q['f'] - cycle_q['0'])
print('%.4E' % Decimal(m))

# Voight model
def voight(c, lam):
    mu, E_v = c
    return mu * m + (lam - 1) * E_v

# Voight model error
c_1 = [1, 1]

def error_v(c, x, y):
    return abs(y - voight(c, x))

# Voight model fit
v_c_fit, v_cov_c = sp.leastsq(
    func=error_v,
    x0=c_1,
    args=(
        stretch_q[cycle_q['0']:cycle_q['f']],
        str_q[cycle_q['0']:cycle_q['f']]
    )
)
mu_v, E_v = v_c_fit
mu_v, E_v

# Plot Voight model fit for quasistatic data
fig = plt.figure(figsize=fig_size)
plt.scatter(
    stretch_q[cycle_q['0']:cycle_q['f']:180],
    str_q[cycle_q['0']:cycle_q['f']:180],
    color=naranjo,
    linewidth=3,
    label='Data'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    voight(v_c_fit, stretch_q[cycle_q['0']:cycle_q['f']]),
    color=azul,
    linewidth=3,
    label='Sólido de Voight'
)
plt.legend()
plt.xlabel(r'Estiramiento $\lambda$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()


# #### Sólido lineal estándar

# Linear Standard solid model
def lineal_standard(c, lam):
    mu, E_1, E_2 = c
    tau_e = mu / E_2
    tau_s = mu * (E_1 + E_2) / (E_1 * E_2)
    return E_1 * ((lam - 1) + m * (tau_s - tau_e) + \
                    np.exp(-(lam - 1) / (tau_e * m)) * \
                    m * (tau_e - tau_s))

# Linear Standard solid model error
c_2 = [1e6, 2.1e6, 3.1e6]       # Best fit parameters
# c_2 = [1, 2.1, 3.1]             # Best fit graph

def error_sls(c, x, y):
    return abs(y - lineal_standard(c, x))

# Linear Standard solid model fit
sls_c_fit, sls_cov_c = sp.leastsq(
    func=error_sls,
    x0=c_2,
    args=(
        stretch_q[cycle_q['0']:cycle_q['f']],
        str_q[cycle_q['0']:cycle_q['f']],
    )
)
mu_sls, E_1, E_2 = sls_c_fit
tau_e = mu_sls / E_2
tau_s = mu_sls * (E_1 + E_2) / (E_1 * E_2)
print(mu_sls, tau_e, tau_s, E_1)

# Plot Linear Standard solid model fit for quasistatic data
fig = plt.figure(figsize=fig_size)
plt.scatter(
    stretch_q[cycle_q['0']:cycle_q['f']:180],
    str_q[cycle_q['0']:cycle_q['f']:180],
    color=naranjo,
    linewidth=3,
    label='Data'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    lineal_standard(sls_c_fit, stretch_q[cycle_q['0']:cycle_q['f']]),
    color=azul,
    linewidth=3,
    label='Sólido lineal estándar'
)
plt.legend()
plt.xlabel(r'Estiramiento $\lambda$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()

# Linear Standard solid model 
# Using the stress as a function of time
'''
def lineal_standard(c, t):
    mu, E_1, E_2 = c
    tau_e = mu / E_2
    tau_s = mu * (E_1 + E_2) / (E_1 * E_2)
    lambda_2 = stretch_q[round(t * 100)]
    return (lambda_2 - 1) * E_1 * (1 - (1 - tau_s / tau_e) * np.exp(-t / tau_e))

c_2 = [2, 2, 2]

def error_sls(c, x, y):
    return abs(y - lineal_standard(c, x))

sls_c_fit, sls_cov_c = sp.leastsq(
    func=error_sls,
    x0=c_2,
    args=(
        time_q[cycle_q['0']:cycle_q['f']],
        str_q[cycle_q['0']:cycle_q['f']],
    )
)
mu_sls, E_1, E_2 = sls_c_fit
tau_e = mu_sls / E_2
tau_s = mu_sls * (E_1 + E_2) / (E_1 * E_2)
print(mu_sls, tau_e, tau_s)

fig = plt.figure(figsize=fig_size)
plt.scatter(
    stretch_q[cycle_q['0']:cycle_q['f']:180],
    str_q[cycle_q['0']:cycle_q['f']:180],
    color=naranjo,
    linewidth=3,
    label='Data'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    lineal_standard(sls_c_fit, time_q[cycle_q['0']:cycle_q['f']]),
    color=azul,
    linewidth=3,
    label='Sólido lineal estándar'
)
plt.legend()
plt.xlabel(r'Estiramiento $\lambda$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()
'''
pass


# #### Comparación de modelos

# Plot all models for quasistatic data
fig = plt.figure(figsize=fig_size)
plt.scatter(
    stretch_q[cycle_q['0']:cycle_q['f']:180],
    str_q[cycle_q['0']:cycle_q['f']:180],
    color=naranjo,
    linewidth=3,
    label='Data'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    voight(v_c_fit, stretch_q[cycle_q['0']:cycle_q['f']]),
    color=verde,
    linewidth=3,
    label='Sólido de Voight'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    lineal_standard(sls_c_fit, stretch_q[cycle_q['0']:cycle_q['f']]),
    color=rojo,
    linewidth=3,
    label='Sólido lineal estándar'
)
plt.plot(
    stretch_q[cycle_q['0']:cycle_q['f']],
    neo_hookean(nh_c_fit, stretch_q[cycle_q['0']:cycle_q['f']]),
    color=azul,
    linewidth=3,
    label='Neohookeano incompresible'
)
plt.legend()
plt.xlabel(r'Estiramiento $\lambda$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()


# ### Pregunta 3

# Plot first oscillation stress vs. deformation
fig = plt.figure(figsize=fig_size)
plt.plot(
    def_h[cycle_h['0']:cycle_h['1']],
    str_h[cycle_h['0']:cycle_h['1']],
    color=naranjo,
    linewidth=3
)
plt.xlabel(r'Deformación $\epsilon$', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()

# Integrate using Simpson's rule
# https://en.wikipedia.org/wiki/Simpson%27s_rule
# https://stackoverflow.com/questions/17602076/how-do-i-integrate-two-1-d-data-arrays-in-python

e_loss = integrate.simpson(
      str_h[cycle_h['0']:cycle_h['1']],
      def_h[cycle_h['0']:cycle_h['1']],
      axis=-1,
      even='avg'
   )
print(f'Energía disipada:', '%.4E' % Decimal(e_loss))

# Plot first oscillation deformation vs. time
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_h[cycle_h['0']:cycle_h['1']],
    def_h[cycle_h['0']:cycle_h['1']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel(r'Deformación $\epsilon$', fontsize=12)
plt.show()

# Define constants
# sigma_0: stress amplitude
# epsilon_0: deformation amplitude
# omega: angular frequency

global sigma_0, epsilon_0, omega

sigma_0 = max(str_h[cycle_h['0']:cycle_h['f']]) / 2
epsilon_0 = max(def_h[cycle_h['0']:cycle_h['1']]) / 2
omega = 10 * np.pi

# Energy dissipated using the Voight model
epsilon_0 ** 2 * np.pi * mu_v * omega

# Energy dissipated using the linear standard model
epsilon_0 ** 2 * np.pi * E_1 * omega * \
    (tau_s - tau_e) / (1 + omega ** 2 * tau_e ** 2)


# ### Bonus

# Plot first oscillation stress vs. time
fig = plt.figure(figsize=fig_size)
plt.plot(
    time_h[cycle_h['0']:cycle_h['1']],
    str_h[cycle_h['0']:cycle_h['1']],
    color=naranjo,
    linewidth=3
)
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()

# Harmonic deformation model
def harmonic_model(c, t):
    delta = c
    return sigma_0 * (1 + np.cos(omega * t + delta))

# Harmonic deformation model error
c_h = 1

def error_h(c, x, y):
    return abs(y - harmonic_model(c, x))

# Harmonic deformation model fit
h_c_fit, h_cov = sp.leastsq(
    func=error_h,
    x0=c_h,
    args=(
        time_h[cycle_h['0']:cycle_h['f']],
        str_h[cycle_h['0']:cycle_h['f']]
    )
)
delta_h = h_c_fit

# Plot harmonic model fit for harmonic data
fig = plt.figure(figsize=fig_size)
plt.scatter(
    time_h[cycle_h['0']:cycle_h['f']],
    str_h[cycle_h['0']:cycle_h['f']],
    color=naranjo,
    linewidth=3,
    label='Data'
)
plt.plot(
    time_h[cycle_h['0']:cycle_h['f']],
    harmonic_model(h_c_fit, time_h[cycle_h['0']:cycle_h['f']]),
    color=azul,
    linewidth=3,
    label='Modelo Armónico'
)
plt.legend()
plt.xlabel('Tiempo [s]', fontsize=12)
plt.ylabel('Tensión [Pa]', fontsize=12)
plt.show()

delta_h


# #### Hecho con :heart: por Iván Vergara Lam
