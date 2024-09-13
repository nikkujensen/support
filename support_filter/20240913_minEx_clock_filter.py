# %% modules
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# %% load data
data = np.load("data.npz")
t = data['t'].squeeze()     # (s)       time vector
dt = np.diff(t)[0]          # (s)       sample period
fs = 1/dt                   # (Hz)      sampling rate
y = data['y'].squeeze()     # (null)    signal

# %% create filter and filter data
# define filter function
def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = sc.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = sc.signal.sosfiltfilt(sos, data)
    return filtered_data, sos

# filter the data
cutoff = 0.1                                    # (Hz)      cutoff frequency
poles  = 100                                    # (null)    filter order
yF, sosfilt = highpass(y, cutoff, fs, poles)

# %% compute power spectrum
fout1,psd  = sc.signal.welch(y.squeeze(),fs)
fout2,psdF = sc.signal.welch(yF.squeeze(),fs)

# %% plot
fig,ax = plt.subplots(3,1)
fig.tight_layout()

# time domain
axh = ax[0]
axh.plot(t, y)
axh.set(xlabel='Time (s)',
          ylabel='Amplitude')
axh.grid()

# filter characteristics
axh = ax[1]
w, h = sc.signal.sosfreqz(sosfilt, worN=8000,fs=fs)
db = 20 * np.log10(abs(h))
yUB = np.max(db[np.isfinite(db)])
yLB = np.min(db[np.isfinite(db)])
axh.plot(w, db)
axh.fill_betweenx([yLB,yUB],cutoff,fout1[-1],
                    color='g',
                    alpha=0.1)
axh.set(xlabel='Frequency (Hz)',
        ylabel='Filter Freq. Response (db)')
axh.grid()

# power spectrum
axh = ax[2]
yUB = np.max(np.fmax(psd,psdF))
yLB = np.min(np.fmin(psd,psdF))
axh.plot(fout1, psd)
axh.plot(fout2, psdF)
axh.fill_betweenx([yLB,yUB],cutoff,fout1[-1],
                    color='g',
                    alpha=0.1)
axh.set_yscale('log')
axh.set(xlabel='Frequency (Hz)',
          ylabel='Power spectrum')
axh.legend(["Original","Original Filt.","Passband"],loc="lower right")
axh.grid()

plt.savefig("output.png",format="png")
plt.show()