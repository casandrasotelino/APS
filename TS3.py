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

N = 1000
fs = 1000
k0 = N/4
ff = k0 * fs / N

#%%funciones

def function_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 1000, fs = 1000):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = vmax * np.sin(2 * np.pi * tt * ff * nn / fs + ph) + dc #funcion seno
    return(tt, xx)

#%%Script

A = np.sqrt(2) #Amplitud del seno para que su potencia = 1
#K = k0
tt, sen = function_sen(vmax= A, ff = ff, nn = N, fs = fs)

SEN = np.fft.fft(sen) / N

#frecuencias hasta Nyquist
freqs = np.fft.fftfreq(N, 1/fs)
corte = N // 2
frec_nyquist = freqs[:corte]


mod_SEN = np.abs(SEN) ** 2
mod_SEN_dB = 10 * np.log10(mod_SEN)

#K = k0 + .25
ff = (k0 + 0.25) * fs / N
tt, sen25 = function_sen(vmax= A, ff = ff, nn = N, fs = fs)

SEN25 = np.fft.fft(sen25) / N




mod_SEN25 = np.abs(SEN25) ** 2 #esta es la energia
mod_SEN25_dB = 10 * np.log10(mod_SEN25)

#K = k0 + .5
ff = (k0 + 0.5) * fs / N
tt, sen5 = function_sen(vmax= A, ff = ff, nn = N, fs = fs)

SEN5 = np.fft.fft(sen5) / N




mod_SEN5 = np.abs(SEN5) ** 2 #esta es la energia
mod_SEN5_dB = 20 * np.log10(mod_SEN5)

Pot = np.sum(np.abs(SEN)**2)
Pot25 = np.sum(np.abs(SEN25)**2)
Pot5 = np.sum(np.abs(SEN5)**2)

plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(frec_nyquist, mod_SEN_dB[:corte])
plt.xlabel("frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.title("módulo de FFT en dB (w)")
plt.grid(True)  

plt.subplot(2, 1, 2)
plt.plot(frec_nyquist,  np.abs(SEN[:corte]), ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("Transformada de Fourier de la función")
'''
'''
zeros = np.zeros(9 * N)
W = np.concatenate(SEN, zeros)**2
W25 = np.concatenate(SEN25, zeros)**2
W5 = np.concatenate(SEN5, zeros)**2

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(frec_nyquist,  W[:corte], ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("K = N/4 + .25")

plt.subplot(2, 1, 2)
plt.plot(frec_nyquist,  W25[:corte], ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("K = N/4")
plt.subplot(2, 1, 3)
plt.plot(frec_nyquist,  W5[:corte], ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("K = N/4")
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(frec_nyquist,  W[:corte], ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("K = N/4 + .25")

plt.subplot(2, 1, 2)
plt.plot(frec_nyquist,  W25[:corte], ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("K = N/4")
plt.subplot(2, 1, 3)
plt.plot(frec_nyquist,  W5[:corte], ":.")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [W]")
plt.grid(True)  
plt.title("K = N/4")