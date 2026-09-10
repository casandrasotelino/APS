# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:50:45 2026

@author: Casandra
En esta tarea continuaremos con el capítulo 14 de Holton. DSP Principles and App.

Comenzaremos con la generación de la siguiente señal:

x(k)=a0⋅sen(Ω1⋅n)+na(n)
siendo

a0=2
Ω1=Ω0+fr⋅2πN
Ω0=π2


siendo la variable aleatoria definida por la siguiente distribución de probabilidad

fr∼U(−2,2)


na∼N(0,σ2)
"""
#%%Librerias
import numpy as np
import scipy.signal.windows as sig
import matplotlib.pyplot as plt

#%%definiciones
a0 = np.sqrt(2) #para que la potencia del seno sea 1 Watt
O0 = np.pi / 2 #defasaje del seno
N = 1000 #cantidad de muestras
fs = 1000 #frecuencia de sampleo [Hz]
R = 200
tt = np.arange(0, N/fs, 1/fs) #vector tiempo (segundos)
tt = tt.reshape((N , 1)) #vector tiempo en columna
tt = np.tile(tt, (1, R))

fr = np.random.uniform(low = -2, high = 2, size = R) #frecuencia del seno
fr_mat = np.tile(fr, (N, 1))
potna = 1 #Watt


#na_SNR10 = np.random.normal(0, np.sqrt(10 ** (-1)), tt) #ruido analógico que atribuye 10 snrdb a la funcion x
na_SNR3 = np.random.normal(0, np.sqrt(10 ** (-3 / 10)), N * R) #ruido analógico que atribuye 3 snrdb a la funcion x
na_SNR3 = na_SNR3.reshape((N, R))
#O1 =  fr * 2 * np.pi * N / fs 


#%%Script
#x_SNR10 = a0 * np.sen(O1 * tt) + na_SNR10

xx = np.sqrt(2) * np.sin((fr + N / 4) * 2 * np.pi * N / fs * tt )  #funcion seno de potencia = 1 W
xx_R = xx + na_SNR3 #funcion seno con ruido de SNRdB = 3


XX = np.fft.fft(xx,axis = 0) / N

mod_XX = np.abs(XX)
mod_XX_ = mod_XX[:N // 2]

XXR = np.fft.fft(xx_R,axis = 0) / N

mod_XXR = np.abs(XXR)
mod_XXR_ = mod_XXR[:N // 2]
#plt.plot(xx[:, 0])
plt.plot(mod_XX_ , ":.", linestyle = ":")
#blackmanharris(M[, sym, xp, device])M numero de muestras
w = sig.blackmanharris(N)
