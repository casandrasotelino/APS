# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:10:53 2026

@author: Casandra
"""

import numpy as np
import matplotlib.pyplot as plt

#%% definiciones
#le atribuyo una amplitud tal que la potencia de la funcion seno sea 0
fs = 1000
N = 1000

#%%funciones

def function_sen_nq(potx = 1, dc = 0, ff = 1, ph = 0, nn = 1000,
                    fs = 1000, SNRdB = 10):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = np.pow(potx, 0.5) * np.sin(2 * np.pi * tt * ff * nn / fs + ph) + dc #funcion seno
    des_est = np.sqrt(pow(10, ((-SNRdB/10)+ np.log10(potx))))
    nq = np.random.normal(0, des_est, nn) #np.random.normal(loc, scale, size) → genera datos normales con media loc y desviación estándar scale.
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

#tt, xx = function_sen(ff = 250)
tt, f_sen_ruido = function_sen_nq(ff = 4, SNRdB= 10000, ph= 1)
# Calcular la FFT

Y = np.fft.fft(f_sen_ruido) #la parte imaginaria(incluye fase) representa las deltas
# Calcular las frecuencias asociadas
#freqs = np.fft.fftfreq(N, 1/fs) + N/2
DC = Y[0]/N
im_Y = np.imag(Y)

# Magnitud normalizada
magnitude = np.abs(Y) / N

plt.plot(im_Y)


