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
ph = N // 4 #tal que el defasaje del especro este en N/4 (representa n = pi/2)
#es importante que ph sea un int porque en el codigo se usa como indice de arrays
#va a ser el flanco de los estimadores, la potencia se debería consentrar en este bin
fs = 1000 #frecuencia de sampleo [Hz]
R = 200
tt = np.arange(0, N/fs, 1/fs) #vector tiempo (segundos)
tt = tt.reshape((N , 1)) #vector tiempo en columna
#tt = np.tile(tt, (1, R))

fr = np.random.uniform(low = -2, high = 2, size = R) #frecuencia del seno
#fr_mat = np.tile(fr, (N, 1))
potna = 1 #Watt


#na_SNR10 = np.random.normal(0, np.sqrt(10 ** (-1)), tt) #ruido analógico que atribuye 10 snrdb a la funcion x
na_SNR3 = np.random.normal(0, np.sqrt(10 ** (-3 / 10)), N * R) #ruido analógico que atribuye 3 snrdb a la funcion x
na_SNR3 = na_SNR3.reshape((N, R))
#O1 =  fr * 2 * np.pi * N / fs 


#%%Script
#x_SNR10 = a0 * np.sen(O1 * tt) + na_SNR10

xx = np.sqrt(2) * np.sin((fr + ph) * 2 * np.pi * N / fs * tt )  #funcion seno de potencia = 1 W
xx_R = xx + na_SNR3 #funcion seno con ruido de SNRdB = 3


XX = np.fft.fft(xx,axis = 0) / N

mod_XX = np.abs(XX)
mod_XX_ = mod_XX[:N // 2] * 2 #para normalizar la potencia en 1

XXR = np.fft.fft(xx_R,axis = 0) / N

mod_XXR = np.abs(XXR)
mod_XXR_ = mod_XXR[:N // 2] * 2 
#plt.plot(xx[:, 0])

#plt.plot(mod_XX_ , ":.", linestyle = ":") 



#blackmanharris(M[, sym, xp, device])M numero de muestras
w_rect=sig.get_window('boxcar', N)
w_rect = w_rect.reshape((N , 1)) #vector tiempo en columna

w_flat = sig.get_window('flattop', N)
w_flat = w_flat.reshape((N , 1)) #vector tiempo en columna

w_bmh = sig.get_window('blackmanharris', N)
w_bmh = w_bmh.reshape((N , 1)) #vector tiempo en columna

w_tri = sig.get_window('triang', N)
w_tri = w_tri.reshape((N , 1)) #vector tiempo en columna


xw0 = xx * w_rect
xw1 = xx * w_flat
xw2 = xx * w_bmh
xw3 = xx * w_tri

XW0 = np.fft.fft(xw0, axis = 0) / N 
XW1 = np.fft.fft(xw1, axis = 0) / N
XW2 = np.fft.fft(xw2, axis = 0) / N 
XW3 = np.fft.fft(xw3, axis = 0) / N

XW0_mod = (np.abs(XW0))[:N // 2] * 2
XW1_mod = (np.abs(XW1))[:N // 2] * 2
XW2_mod = (np.abs(XW2))[:N // 2] * 2
XW3_mod = (np.abs(XW3))[:N // 2] * 2

XW0_dB = 20 * np.log10(XW0_mod) 
XW1_dB = 20 * np.log10(XW1_mod)
XW2_dB = 20 * np.log10(XW2_mod)
XW3_dB = 20 * np.log10(XW3_mod)

XW0_dB = XW0_dB - np.max(XW0_dB, axis = 0)
XW1_dB = XW1_dB - np.max(XW1_dB, axis = 0)
XW2_dB = XW2_dB - np.max(XW2_dB, axis = 0)
XW3_dB = XW3_dB - np.max(XW3_dB, axis = 0)

#plt.hist(mod_XX_[:  , N//4], bins= N // 2, color='#F2AB6D', rwidth=0.85) #plt.hist(datos, bins=8, color='#F2AB6D', rwidth=0.85)

#a1 = np.sum(mod_XX_[:  , N//4], axis = -1)
# Figure 2: Effect of windows

plt.figure(1)
plt.subplot(2, 1, 1)
for r in np.arange(0, R):
    lbl1 = "Rectangular" if r == 0 else ""
    lbl2 = "Flattop" if r == 0 else ""
    lbl3 = "Blackman-Harris" if r == 0 else ""
    lbl4 = "Triangular" if r == 0 else ""

    plt.plot(XW0_mod[:, r], color='tab:blue', linestyle='--', alpha=0.15, label=lbl1)
    plt.plot(XW1_mod[:, r], color='tab:orange', linestyle='--', alpha=0.15, label=lbl2)
    plt.plot(XW2_mod[:, r], color='tab:green', linestyle='--', alpha=0.15, label=lbl3)
    plt.plot(XW3_mod[:, r], color='tab:red', linestyle='--', alpha=0.15, label=lbl4)

#plt.xlim(247, 253)
plt.xlabel("Índice de frecuencia ($k$)")
plt.ylabel("$|X_w[k]|$")
plt.title("Efecto de las ventanas sobre 200 realizaciones (Lineal)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
for r in np.arange(0, R):
    lbl1 = "Rectangular" if r == 0 else ""
    lbl2 = "Flattop" if r == 0 else ""
    lbl3 = "Blackman-Harris" if r == 0 else ""
    lbl4 = "Triangular" if r == 0 else ""

    plt.plot(XW0_dB[:, r], color='tab:blue', linestyle='--', alpha=0.15, label=lbl1)
    plt.plot(XW1_dB[:, r], color='tab:orange', linestyle='--', alpha=0.15, label=lbl2)
    plt.plot(XW2_dB[:, r], color='tab:green', linestyle='--', alpha=0.15, label=lbl3)
    plt.plot(XW3_dB[:, r], color='tab:red', linestyle='--', alpha=0.15, label=lbl4)

#plt.xlim(247, 253)
#plt.ylim(0, -70)

plt.xlabel("Índice de frecuencia ($k$)")
plt.ylabel("$10 \log_{10}(|X_w[k]|)$")
plt.title("Efecto de las ventanas sobre 200 realizaciones (dB)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

#%% Estimadores

#obtengo todos los valores para las muestras en ph
array_w0 = XW0_mod[ph, :]
array_w1 = XW1_mod[ph, :]
array_w2 = XW2_mod[ph, :]
array_w3 = XW3_mod[ph, :]


estim_Amplitud = [np.mean(array_w0), np.mean(array_w1), np.mean(array_w2), 
                  np.mean(array_w3)]

'''
indices_max = []

for i in range(R):
    indice = np.argmax(XW0_mod[:, i])
    indices_max.append(indice)

media_indices = np.mean(indices_max)
'''
indices_max0 = np.argmax(XW0_mod, axis=0)
indices_max1 = np.argmax(XW1_mod, axis=0)
indices_max2 = np.argmax(XW2_mod, axis=0)
indices_max3 = np.argmax(XW3_mod, axis=0)
estim_freq = [np.mean(indices_max0), np.mean(indices_max1),
              np.mean(indices_max2), np.mean(indices_max3)]

sa = estim_Amplitud - a0
va = [np.var(array_w0), np.var(array_w1),
      np.var(array_w2), np.var(array_w3)]

sO = np.array(estim_freq) - 250
vO = [np.var(indices_max0), np.var(indices_max1),
      np.var(indices_max2), np.var(indices_max3)]

#calibro los valores del estimador de amplitud 
# sabiendo que el valor esperado real es la amplitud de la señal sinusoide
array_w0 = array_w0 + a0 - np.mean(array_w0)
array_w1 = array_w1 + a0 - np.mean(array_w1)
array_w2 = array_w2 + a0 - np.mean(array_w2)
array_w3 = array_w3 + a0 - np.mean(array_w3)



#%%Tablas
#amplitud
filas = ['Rectangular', 'Flat-top', 'Blackman', 'Triangular']
columnas = [r'$s_a$', r'$v_a$']

fig, ax = plt.subplots()

ax.axis('off')

datos = [
    [f'{sa[0]:.3g}', f'{va[0]:.3g}'],
    [f'{sa[1]:.3g}', f'{va[1]:.3g}'],
    [ f'{sa[2]:.3g}', f'{va[2]:.3g}'],
    [f'{sa[3]:.3g}', f'{va[3]:.3g}']
]

tabla = ax.table(
    cellText=datos,
    rowLabels=filas,
    colLabels=columnas,
    cellLoc='center',
    loc='center',
    colWidths=[0.12, 0.12, 0.12]
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.8)

plt.title('Estimación de Amplitud', fontsize=18, fontweight='bold')
plt.show()

#frecuencias
filas = ['Rectangular', 'Flat-top', 'Blackman', 'Triangular']
columnas = [r'$s_{\Omega}$', r'$v_{\Omega}$']

fig, ax = plt.subplots()

ax.axis('off')

datos = [
    [f'{sO[0]:.3g}', f'{vO[0]:.3g}'],
    [f'{sO[1]:.3g}', f'{vO[1]:.3g}'],
    [ f'{sO[2]:.3g}', f'{vO[2]:.3g}'],
    [f'{sO[3]:.3g}', f'{vO[3]:.3g}']
]

tabla = ax.table(
    cellText=datos,
    rowLabels=filas,
    colLabels=columnas,
    cellLoc='center',
    loc='center',
    colWidths=[0.12, 0.12, 0.12]
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1, 1.8)

plt.title('Estimación de Frecuencias', fontsize=18, fontweight='bold')
plt.show()
'''
#%%Histograma
plt.figure(3)
plt.title("Histograma con los valores calibrados")
plt.hist(array_w0, bins=20, alpha=0.5, label='Rectangular')
plt.hist(array_w1, bins=20, alpha=0.5, label='Flat-top')
plt.hist(array_w2, bins=20, alpha=0.5, label='Blackman')
plt.hist(array_w3, bins=20, alpha=0.5, label='Triangular')

plt.xlabel('Amplitud')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid()
plt.show()
'''