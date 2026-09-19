# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:18:49 2026

@author: Casandra
Se pide:

1) Realizar la estimación de la densidad espectral de potencia (PSD) de
    cada señal mediante alguno de los métodos vistos en clase (Periodograma 
    ventaneado, Welch, Blackman-Tukey).

2) Realice una estimación del ancho de banda de cada señal y presente los
    resultados en un tabla para facilitar la comparación.
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
import matplotlib.pyplot as plt



##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy') #Tengo 30000 muestras
#el período del ECG es de 810 s 
N = 30000
K = [3, 7, 8, 35]
fs = 1000

plt.figure()
for k in K:
    freq_ECG, PSD_ECG = sig.welch(ecg_one_lead, fs=fs, nperseg= N // k)
    
    plt.plot(freq_ECG, PSD_ECG, ":.", linestyle = ":",
              label=f'Long. promedio = {N // k} muestras (K={k})')

plt.title('Estimación de PSD - Método de Welch - ECG sin ruido')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad Espectral de Potencia [Watt/Hz]')
plt.legend(loc='best', fontsize=9)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xlim([0, fs/2])  # hasta Nyquist
plt.tight_layout()
plt.show()