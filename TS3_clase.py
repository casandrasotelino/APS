# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:50:45 2026

@author: Casandra
"""
# -*- coding: utf-8 -*-

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

## CLASE 03-09 - ts3 con iri <3
#%% LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import kstest


#%% 1. FUNCIÓN SENOIDAL
def funcion_sen(A = 1, DC = 0, ff = 1, phi = 0, nn = 1000, fs = 1000):
    tt = np.arange(nn) / fs
    xx = A * np.sin(2 * np.pi * ff * tt + phi) + DC
    return tt, xx
 
#%% 2. PARÁMETROS GENERALES
Ps = 1 # potencia de la senoidal (normalizada a 1W)
Amplitud = np.sqrt(2*Ps)
N = 1000
fs = 1000
k0 = N/4 # desp sumarle 0.25 y 0.5
ff = k0 * (fs/N)
 
#%% 3. SEÑAL SENOIDAL
tt, xx = funcion_sen( A = Amplitud, ff = ff, nn = N, fs = fs)
 
fft_seno = np.fft.fft(xx)/N #/N para normalizarlo

POT = np.abs(fft_seno) ** 2 
POTdB = 10 * np.log10(POT) 
POTdB = POTdB -  np.max(POTdB) #Normalizo en 0

#%% 4. FRECUENCIAS HASTA NYQUIST
frecuencias = np.fft.fftfreq(N, 1/fs)
corte = N // 2
frec_nyquist = frecuencias[:corte]



## k1 = N/4 + 0.25
k1 = (N/4) + 0.25
f1 = k1 * (fs/N)
tt, x1 = funcion_sen( A = Amplitud, ff = f1, nn = N, fs = fs)
fft_seno1 = np.fft.fft(x1)/N #Normalizada
POT1 = np.abs(fft_seno1) ** 2
POT1dB = 10 * np.log10(POT1) 
POT1dB = POT1dB -  np.max(POT1dB) #Normalizo en 0

## k = N/4 + 0.5
k2 = (N/4) + 0.5
f2 = k2 * (fs/N)
tt, x2 = funcion_sen(A = Amplitud, ff = f2 ,nn = N, fs = fs)
fft_seno2 = np.fft.fft(x2)/N
POT2 = np.abs(fft_seno2) ** 2
POT2dB = 10 * np.log10(POT2) 
POT2dB = POT2dB -  np.max(POT2dB) #Normalizo en 0

#%% CALCULAMOS PARSEVAL:
# Potencia calculada en el tiempo:
pot = np.mean(xx**2)
pot1 = np.mean(x1**2)
pot2 = np.mean(x2**2)

# Potencia calculada desde la FFT (Parseval):
potencia_frecuencia = np.sum(np.abs(fft_seno)**2)
potencia_frecuencia1 = np.sum(np.abs(fft_seno1)**2)
potencia_frecuencia2 = np.sum(np.abs(fft_seno2)**2)

print(f"Potencia en el tiempo 0: {pot}")
print(f"Potencia en el tiempo 1: {pot1}")
print(f"Potencia en el tiempo 2: {pot2}")


print(f"Potencia en frecuencia 0 (Parseval): {potencia_frecuencia}")
print(f"Potencia en frecuencia 1 (Parseval): {potencia_frecuencia1}")
print(f"Potencia en frecuencia 2 (Parseval): {potencia_frecuencia2}")

zeros=np.zeros(9*N)
xx_zeros=np.concatenate((xx,zeros))
x1_zeros=np.concatenate((x1,zeros))
x2_zeros=np.concatenate((x2,zeros))

N_pad = N + 9*N  # 10*N muestras

frecuencias_pad = np.fft.fftfreq(N_pad, 1/fs)
corte_pad = N_pad // 2
frec_nyquist_pad = frecuencias_pad[:corte_pad]

fft_xxz = (1/N) * np.fft.fft(xx_zeros)
fft_x1z = (1/N) * np.fft.fft(x1_zeros)
fft_x2z = (1/N) * np.fft.fft(x2_zeros)

POT_pad = (np.abs(fft_xxz[:corte_pad]))**2
POT1_pad = (np.abs(fft_x1z[:corte_pad]))**2
POT2_pad = (np.abs(fft_x2z[:corte_pad]))**2

POT_pad_dB = 10 * np.log10(POT_pad)
POT_pad_dB = POT_pad_dB - np.max(POT_pad_dB)
POT1_pad_dB = 10 * np.log10(POT1_pad)
POT1_pad_dB = POT_pad_dB - np.max(POT_pad_dB)
POT2_pad_dB = 10 * np.log10(POT2_pad)
PO2_pad_dB = POT2_pad_dB - np.max(POT_pad_dB)

#%% GRAFICOS
plt.figure(1, figsize=(10,8))

## k = N/4
plt.subplot(2,1,1)
plt.title("ESPECTRO DE POTENCIA")
plt.grid(True)  
plt.plot(frec_nyquist, (POT[:corte]), ":.", label='k = N/4', linestyle = "", color = "green")
plt.plot(frec_nyquist, (POT1[:corte]), ":.", label='k = N/4 + 0.25', linestyle = "", color = "red")
plt.plot(frec_nyquist, (POT2[:corte]), ":.", label='k = N/4 + 0.5', linestyle = "", color = "blue")
plt.legend(fontsize=8)

plt.subplot(2,1,2)
plt.title("ESPECTRO DE POTENCIA EN dB")
plt.plot(frec_nyquist, POTdB[: corte] , ":.", label ='k = N/4', linestyle = "", color = "green")
plt.plot(frec_nyquist, POT1dB[: corte] , ":.", label ='k = N/4 + 0.25', linestyle = "", color = "red")
plt.plot(frec_nyquist, POT2dB[: corte] , ":.", label ='k = N/4 + 0.5', linestyle = "", color = "blue")
plt.grid(True) 
plt.tight_layout()
plt.legend(fontsize=8)
'''
# K = N/4 + 0.25
plt.figure(2)
plt.subplot(2,1,1)
plt.title("ESPECTRO DE POTENCIA")
plt.grid(True)  
plt.legend(fontsize=8)

plt.subplot(2,1,2)
plt.title("ESPECTRO DE POTENCIA EN dB")
plt.grid(True)  
plt.tight_layout()
plt.legend(fontsize=8)

# K = N/4 + 0.5
plt.figure(3)
plt.subplot(2,1,1)
plt.title("ESPECTRO DE POTENCIA")
plt.grid(True)  
plt.subplot(2,1,2)
plt.title("ESPECTRO DE POTENCIA EN dB")
plt.grid(True)  
plt.tight_layout()
plt.legend(fontsize=8)
'''
#%%Graficos con zero padding
plt.figure(figsize=(10,6))
plt.title("Espectro de potencia con zero padding")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [Watt]")

plt.plot(frec_nyquist_pad, POT_pad, ".", label='k = N/4')
plt.plot(frec_nyquist_pad, POT1_pad, ".", label='k = N/4+0.25')
plt.plot(frec_nyquist_pad, POT2_pad, ".", label='k = N/4+0.5')

plt.legend(fontsize=8)
plt.grid(True)
plt.xlim(200, 300)  # zoom cerca del pico para ver bien el desparramo
plt.tight_layout()

plt.figure(figsize=(10,6))
plt.title("Espectro de potencia en dB con zero padding")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Potencia [dB]")
plt.plot(frec_nyquist_pad, POT_pad_dB,":v", label='k = N/4')
plt.plot(frec_nyquist_pad, POT1_pad_dB, ".", label='k = N/4+0.25')
plt.plot(frec_nyquist_pad, POT2_pad_dB, ".", label='k = N/4+0.5')
plt.xlim(200, 300)  # zoom cerca del pico para ver bien el desparramo
plt.legend(fontsize=8)
plt.grid(True)
plt.xlim(200, 300)  # zoom cerca del pico para ver bien el desparramo
plt.tight_layout()
