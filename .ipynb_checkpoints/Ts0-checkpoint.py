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
ff = 3 #frecuencia de la sinoidal
#ff = 100
#ff = 500 #Hz
#ff = 999 #Hz
#ff = 1001 #Hz
#ff = 2001 #Hz
phi = 0
vmax = 1.5
dc = 4 #valor medio // offset


#%%definicion de funciones

def function_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = N, fs = fs):
    tt = np.arange(0, nn/fs, 1/fs) #vector tiempo (segundos)
    xx = vmax * np.sin(2 * np.pi * tt * ff * N / fs + phi) + dc #funcion seno
    return(tt, xx)

#%% comienzo de mi script


tt, xx = function_sen(vmax, 0, ff, 0 , N, fs);
plt.plot(tt, xx)

#plt.show()