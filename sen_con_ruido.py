# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:10:53 2026

@author: Casandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

#%% definiciones
#le atribuyo una amplitud tal que la potencia de la funcion seno sea 0
fs = 1000
N = 1000

#%%funciones

def function_sen_nq(potx = 1, dc_sen = 0, ff = 1, ph = 0, nn = 1000,
                    fs = 1000, SNRdB = 10, dc_noise = 0):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = np.pow(potx, 0.5) * np.sin(2 * np.pi * tt * ff * nn / fs + ph) + dc_sen #funcion seno
    des_est = np.sqrt(pow(10, ((-SNRdB/10)+ np.log10(potx))))
    nq = np.random.normal(0, des_est, nn) +dc_noise #np.random.normal(loc, scale, size) → genera datos normales con media loc y desviación estándar scale.
    f_suma = xx + nq
    return(tt, f_suma)

def function_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 1000, fs = 1000):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = vmax * np.sin(2 * np.pi * tt * ff * nn / fs + ph) + dc #funcion seno
    return(tt, xx)

#%%script
'''
tt, f_con_ruido = function_sen_nq(potx= 1, SNRdB=60)
#var = np.var(xx) #debe dar vmax^2/2
#print("var = ", var)
plt.plot(tt, f_con_ruido)
#para un SNRdB = 20 = -10*log(Pnq) si Px = 1, lo otro en carpeta (20/8)
'''
#%%20/8; FFT
#x(k = 0) = DC * N

#tt, xx = function_sen(ff = 1)
tt, f_sen_ruido = function_sen_nq(ff = 4, SNRdB= 40, ph= 0)
# Calcular la FFT

f_sen_ruido = f_sen_ruido[:N//2]


#Y = np.fft.fft(xx)
Y = (1/N) * np.fft.fft(2 * f_sen_ruido) #la parte imaginaria(incluye fase) representa las deltas
#multiplico * 1/N para normalizar y *2 para que la potencia sea 1 de la mitad de f_sen_conruido
# Calcular las frecuencias asociadas
#freqs = np.fft.fftfreq(N, 1/fs) + N/2
#Y_acot = Y[:N//2]
tt_acot = tt[:N//2]
#xx_acot = xx[:N//2]


DC = Y[0]/N
fase_Y = np.angle(Y)
mod_Y = np.abs(Y)
Y_dB = 20 * np.log10(mod_Y)

plt.figure(1)
#plt.plot(xx_acot)
#plt.plot(f_sen_ruido)
plt.subplot(2,1,1)
plt.plot(Y_dB)
plt.subplot(2,1,2)
plt.plot(fase_Y)

#%%cuantización
B = 8 #2^B bits
Vfs = 1.65 #uso 1.65 pq ax= np.sqrt(2) q es aprox 1.5, le sumo un poco x el error
q = 2 * Vfs /pow(2, B) #niveles de cuantización
f_sen_ruido_D = np.round(f_sen_ruido/q) * q
nq = (f_sen_ruido_D - f_sen_ruido) #/ q #normalizo en q para que el valor este enre +-0.5 no lo normalizo asi queda en volts ;)

plt.figure(2)
plt.plot(f_sen_ruido_D, ':x')
plt.plot(f_sen_ruido, ':v')

#%% analisis del ruido
#demuestro que ruido es incorrelado; la energía debe ser una delta en n = 0

Pnq = np.correlate(nq, nq, mode='full') / (N/2)
plt.figure(3)
plt.plot(Pnq)
i = (N/2)-1

#print("varianza: ", np.var(nq), "valor de la energía en el delta: ", Pnq[i])

#copio de chat:
#%% Test de Kolmogorov-Smirnov

resultado = kstest(nq, 'uniform', args=(-q/2, 0))

print("Estadístico KS =", resultado.statistic)
print("p-value =", resultado.pvalue)


#%% Decisión estadística

alpha = 0.05

if resultado.pvalue > alpha:
    print("No se rechaza H0.")
    print("El ruido es compatible con una distribución uniforme.")
else:
    print("Se rechaza H0.")
    print("El ruido NO es compatible con una distribución uniforme.")