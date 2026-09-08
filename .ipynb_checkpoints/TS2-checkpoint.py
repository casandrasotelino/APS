# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:06:39 2026
CUANTIZACION

@author: Casandra
"""
'''
En esta tarea semanal retomamos la consigna de la tarea anterior, donde simulamos el bloque de cuantización de un ADC de B bits en un rango de  ±VF
 Volts. Ahora vamos a completar la simulación del ADC incluyendo la capacidad de muestrear a fs Hertz.

Para ello se simulará el comportamiento del dispositivo al digitalizar una senoidal contaminada con un nivel predeterminado de ruido. Comenzaremos describiendo los parámetros a ajustar de la senoidal:

frecuencia f0
 arbitraria, por ejemplo f0=fS/N=Δf
 
energía normalizada, es decir energía (o varianza) unitaria
Con respecto a los parámetros de la secuencia de ruido, diremos que:

será de carácter aditivo, es decir la señal que entra al ADC será sR=s+n
. Siendo n
 la secuencia que simula la interferencia, y s
 la senoidal descrita anteriormente.
La potencia del ruido será Pn=kn.Pq
 W siendo el factor k una escala para la potencia del ruido de cuantización Pq=q212
.
finalmente, n
 será incorrelado y Gaussiano.
El ADC que deseamos simular trabajará a una frecuencia de muestreo fS=1000
 Hz y tendrá un rango analógico de ±VF=2
 Volts.

Se pide:
'''
#%% Librerías
import numpy as np
import matplotlib.pyplot as plt
#%% definiciones

fs = 1000 #frecuencia de sampleo
N = 1000 #número de muestras
B = [4, 8, 16]; i = int(0) #canidad de bits
k = [0.1, 1, 10]; j = int(2)
Vref = 1 #rango del ADC = 2 -> toma valores de -1 a +1
q = (2 * Vref) /(2 ** B[i]) #niveles de cuantización

#%%funciones

def function_sen_nq(A = np.sqrt(2), dc_sen = 0, ff = 1, ph = 0, nn = 1000,
                    fs = 1000, des_est = 1, dc_noise = 0):
    tt = np.arange(0, nn/fs, 1/fs)
    xx = A * np.sin(2 * np.pi * tt * ff + ph) + dc_sen #como A es = 1 entoces la energía se encuentra normalizada
    nq = np.random.normal(0, des_est, nn) + dc_noise
    f_suma = xx + nq
    return(tt, f_suma, xx)


#%%Script

Pq = q**2 / 12 #Potencia del ruido de cuantización
Pa = k[j] * Pq #Potencia del ruido analítico
tt, sR, s = function_sen_nq(ff = 1, des_est = np.sqrt(Pa), ph= 0)

sR_D = np.round(sR / q) * q #señal seno con ruido digitalizada

FFT_senD = np.fft.fft(sR_D) / N
FFT_senA = np.fft.fft(sR) / N

FAdB = 20 * np.log10(np.abs(FFT_senA))
FDdb = 20 * np.log10(np.abs(FFT_senD))

E_q = np.median(np.abs(FFT_senD)) #Media del espectro de potencia del sR_D
E_q = 20 * np.log10(E_q)
E_a = np.median(np.abs(FFT_senA)) #Media del espectro de potencia del sR
E_a = 20 * np.log10(E_a)
 
e_q = sR_D - sR #error de cuantizacion del ruido

FC = (1 **2 / 12)/(Vref ** 2) # factor de carga

#%%Gráficos
#%%Potencia (dB ruidos)
plt.figure(figsize=(12, 5))

plt.plot(FDdb, color = "cyan", label=r'$S_0 = Q_{v,s}[S]$ (ADC out)')
#plt.plot(S_dB, label=r'$S$ (analog)')
plt.plot(FAdB, color = "orange", label=r'$sR_D = S + \eta_Q$ (ADC in)')

# Niveles de ruido
plt.axhline(
    E_a,
    linestyle='--',
    label=r'$\eta = %.1f$ dB (piso analog)' % E_a,
    color = "red"
)

plt.axhline(
    E_q,
    linestyle=':',
    label=r'$\eta_Q = %.1f$ dB (piso digital)' % E_q,
)

# Ejes
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')

# Título
plt.title(
    rf"Señal muestreada por un ADC de {B[i]} bits -  $\pm V_R = 2.0$ V - $q = 0.125$ V"
)

plt.xlim(0, 500)
plt.grid(False)

plt.legend()
plt.tight_layout()

#%% senos(tt)
plt.figure(figsize=(12, 6))

# Linea 1: Señal cuantizada sR_D (ADC out)
plt.plot(tt, sR_D, label=r"$sQ = Q_{B,V_R}\{s_R\}$ (ADC out)", color="#1f77b4", linewidth=2)

# Linea 2: Señal con ruido sR (ADC in)
plt.plot(
    tt,
    sR,
    label=r"$s_R = s + n$  (ADC in)",
    color="green",
    linestyle=":",
    marker="o",
    markersize=3,
    markerfacecolor="none",
    markeredgecolor="green",
    alpha=0.8,
)

# Linea 3: Señal analogica pura s(tt)
plt.plot(tt, s, label=r"$s$ (analog)", color="orange", linestyle=":", linewidth=1.5)

# Titulos y etiquetas
plt.title(
    rf"Señal muestreada por un ADC de {B[i]} bits - $\pm V_R = {Vref:.1f}$ V - q = {q:.3f} V"
)
plt.xlabel("tiempo [segundos]")
plt.ylabel("Amplitud [V]")

# Leyenda y formato de reticula/ejes
plt.legend(loc="upper right")
plt.tight_layout()

#%% Graficacion del histograma
plt.figure(figsize=(12, 6))

# Histograma de los errores de cuantizacion entre -q/2 y q/2 (10 bins)
n, bins, patches = plt.hist(
    e_q, bins=10, range=(-q / 2, q / 2), edgecolor="none"
)

# Linea discontinua roja delimitando la distribucion teorica uniforme
plt.plot(
    [-q / 2, -q / 2, q / 2, q / 2],
    [0, 100, 100, 0],
    color="red",
    linestyle="--",
    linewidth=1.5,
)

# Titulo
plt.title(
    rf"Ruido de cuantización para {B[i]} bits - $\pm V_R =$ {Vref:.1f} V - q = {q:.3f} V"
)

plt.tight_layout()
plt.show()