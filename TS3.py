# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:31:05 2026

@author: Casandra
"""
'''En esta tarea semanal analizaremos un fenómeno muy particular que se da al calcular la DFT, el efecto de desparramo espectral.  

Luego, haremos el siguiente experimento:

Senoidal de frecuencia f0=k0∗fS/N=k0.Δf
potencia normalizada, es decir energía (o varianza) unitaria
Se pide:

a) Sea k0
 

N4
 
N4+0.25
 
N4+0.5
 
Notar que a cada senoidal se le agrega una pequeña desintonía respecto a  Δf
. Graficar las tres densidades espectrales de potencia (PDS's) y discutir cuál
 es el efecto de dicha desintonía en el espectro visualizado.

b) Verificar la potencia unitaria de cada PSD, puede usar la identidad de Parseval. En base a la teoría estudiada. Discuta la razón por la cual una señal senoidal tiene un espectro tan diferente respecto a otra de muy pocos Hertz de diferencia. 

c) Repetir el experimento mediante la técnica de zero padding. Dicha técnica consiste en agregar ceros al final de la señal para aumentar Δf
 de forma ficticia. Probar agregando un vector de 9*N ceros al final. Discuta los resultados obtenidos.'''
import numpy as np
import matplotlib.pyplot as plt

#%%Definiciones

N = 1000 # cantidad de muestras
fs = 1000 # frecuencia de nyquist
ff = N / 4 # frecuencia de la funcion seno que cae en el bin ubicado en N / 4


#%%funciones

def function_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = vmax * np.sin(2 * np.pi * tt * ff * (nn / fs) + ph) + dc #funcion seno
    return(tt, xx)

#%%Script
#%%Funciones seno con las potencias normalizadas
A = np.sqrt(2) #Amplitud del seno para que su potencia = 1
#K = k0
tt, sen = function_sen(vmax= A, ff = ff, nn = N, fs = fs)

SEN = np.fft.fft(sen) * 2 / N #multiplico por dos porque tomaré valores hasta nyquist

SEN_dB = 20 * np.log10(np.abs(SEN))
SEN_dB = SEN_dB - np.max(SEN_dB)

#K = k0 + .25

ff = (ff + 0.25)
tt, sen25 = function_sen(vmax= A, ff = ff, nn = N, fs = fs)

SEN25 = np.fft.fft(sen25) * 2 / N


SEN25_dB = 20 * np.log10(np.abs(SEN25))
SEN25_dB = SEN25_dB - np.max(SEN_dB)


#K = k0 + .5
ff = (ff + 0.5)
tt, sen5 = function_sen(vmax= A, ff = ff, nn = N, fs = fs)

SEN5 = np.fft.fft(sen5) * 2/ N

SEN5_dB = 20 * np.log10(np.abs(SEN5))
SEN5_dB = SEN5_dB - np.max(SEN_dB)


#%% Parseval para corroborar que los senos tienen potencias normalizadas

Pot = np.sum(0.5 * np.abs(SEN)**2)
Pot25 = np.sum(0.5 * np.abs(SEN25)**2)
Pot5 = np.sum(0.5 * np.abs(SEN5)**2)

#%% Zero padding
zeros = np.zeros(9 * N)

w = np.concatenate((sen, zeros))
w25 = np.concatenate((sen25, zeros))
w5 = np.concatenate((sen5, zeros))

W = np.fft.fft(w) * 2/ (10 * N)
W25 = np.fft.fft(w25) * 2 / (10 * N)
W5 = np.fft.fft(w5) * 2 / (10 * N)


W_dB = 20 * np.log10(np.abs(W[:(10 * N) // 2]))
W25_dB = 20 * np.log10(np.abs(W25[:(10 * N) // 2]))
W5_dB = 20 * np.log10(np.abs(W5[:(10 * N) // 2]))

W_dB = W_dB - np.max(W_dB)
W25_dB = W25_dB - np.max(W_dB)
W5_dB = W5_dB - np.max(W_dB)

frec_nyquist_W = np.arange(0 , N // 2, N / (10 * fs))
#%% Gráficos sin zero padding

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(
    (np.abs(SEN))[:N // 2] ** 2,
    ":.",
    linestyle = ":",
    label='k = N/4'
)

plt.plot(
    np.abs(SEN25)[:N // 2] ** 2,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.25'
)

plt.plot(
    np.abs(SEN5)[:N // 2] ** 2,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.5'
)

plt.title('Espectro de potencia sin zero padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia')
plt.xlim(200, 300)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(
    SEN_dB,
    ":.",
    linestyle = ":",
    label='k = N/4'
)

plt.plot(
    SEN25_dB,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.25'
)

plt.plot(
    SEN5_dB,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.5'
)

plt.title('Espectro de amplitud en dB sin zero padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.xlim(200, 300)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%% Gráficos con zero padding

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(
    frec_nyquist_W,
    np.abs(W)[:(10 * N) // 2] ** 2,
    ":.",
    linestyle = ":",
    label='k = N/4'
)

plt.plot(
    frec_nyquist_W,
    np.abs(W25)[:(10 * N) // 2] ** 2,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.25'
)

plt.plot(
    frec_nyquist_W,
    np.abs(W5)[:(10 * N) // 2] ** 2,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.5'
)

plt.title('Espectro de potencia con zero padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia')
plt.xlim(200, 300)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)

plt.plot(
    frec_nyquist_W,
    W_dB,
    ":.",
    linestyle = ":",
    label='k = N/4'
)

plt.plot(
    frec_nyquist_W,
    W25_dB,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.25'
)

plt.plot(
    frec_nyquist_W,
    W5_dB,
    ":.",
    linestyle = ":",
    label='k = N/4 + 0.5'
)

plt.title('Espectro de amplitud en dB con zero padding')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.xlim(200, 300)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()