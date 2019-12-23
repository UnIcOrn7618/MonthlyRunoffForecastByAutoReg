import matplotlib.pyplot as plt
plt.rcParams['font.size']=6
import pandas as pd
import numpy as np
from scipy.fftpack import fft

Fs=1000 #sampling frequency
T=1/Fs #sampling period(interval)
L=1500 #length of signal
t=np.arange(start=0,stop=L,step=1)*T
print(t)
s = [0.7*np.sin(2*np.pi*50*ts) for ts in t] # amplitude=0.7, frequency=50Hz
# freqs = t-0.5-1/T 
# print(freqs)
print(s)
plt.figure()
plt.subplot(2,1,1)
plt.plot(s)
plt.axhline(y=0)
plt.subplot(2,1,2)
plt.plot(abs(fft(s)))
plt.show()