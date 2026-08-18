# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:26:42 2026

@author: Casandra
"""

import numpy as np
import matplotlib.pyplot as plt

#%% definiciones
fs = 1000 #Hz
N = 1000 #muestras
# definir ff tq ff < N/2 (nyquist), cuando m paso
#ff = 3 #frecuencia de la sinoidal
#ff = 500 #Hz
#ff = 999 #Hz
ff = 1001 #Hz
''''
todo lo que este afuera del rango de |ff|<fs quta la periodicidad del expectro,
pero también interpola.
 Limita la energía.
'''
#ff = 2001 #Hz
phi = np.pi /3 #para que los valores tomasdos no sean las raices y no acumulen tanto error
vmax = 1.5
dc = 4 #valor medio // offset



#%%definicion de funciones

def function_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = vmax * np.sin(2 * np.pi * tt * ff * N / fs + phi) + dc #funcion seno
    return(tt, xx)

def DFT_sen(w, w0 = ff * N / fs):
    a = 0
    delta_neg = w -w0
    delta_pos = w + w0
    if (delta_neg == 0):
        a = 1
    elif delta_pos == 0:
        a = 1
    return a

#%% comienzo de mi script


tt, xx = function_sen(vmax, 0, ff, phi , N, fs);
plt.plot(tt, xx)
w = tt
delta = tt
for i in w:
    w[i] = 1/tt[i]
    delta[i] = DFT_sen(w = w[i])

plt.plot(w, delta)
#plt.show()