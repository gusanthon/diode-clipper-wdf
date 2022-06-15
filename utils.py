import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_test_wave(fs,f,amp,t,kind):
  N = int(t * fs)
  n = np.arange(0,N/fs,1/fs)
  if kind=='sin':
    x = np.sin(2 * np.pi * f * n) * amp
  elif kind == 'cos':
    x = np.cos(2 * np.pi * f * n) * amp
  elif kind == 'delta':
    x = np.zeros(N)
    x[0] = 1
  return x

def read_to_linear(path,bd,lin=0):
  fs,x = scipy.io.wavfile.read(path)
  if lin:
    return fs,x
  if bd == 16:
    x = x.astype(np.float32, order='C') / 32768.0  
  elif bd == 32:
    x = x.astype(np.float32, order='C') / 2147354889.0 
  return fs,x 

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int32(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

def plot_fft(audio,fs,title='output spectrum'):
  N = len(audio)
  w = scipy.signal.windows.hann((N))
  w_fft = scipy.fft.fft(w*audio)
  xf = scipy.fft.fftfreq(N, 1/fs)[:N//2]
  plt.plot(xf, 2.0/N * np.abs(w_fft[0:N//2]),label='fft hann window')
  plt.title(title)
  plt.legend()
  plt.show()

def freqz(x,fs):
    w,h = scipy.signal.freqz(x,1,4096)
    H = 20 * np.log10(np.abs(h))
    f = w / (2 * np.pi) * fs
    angles = np.unwrap(np.angle((h)))
    return f,H,angles

def plot_magnitude_response(f,H,label='magnitude',c='b',title=''):
    ax = plt.subplot(111)
    # magnitude response
    plt.plot(f,H,label=label,color=c)
    ax.semilogx(f, H)
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequency [hz]')
    plt.title(title + "Magnitude response")

def plot_phase_response(f,angles,mult_locater=(np.pi/2),denom=2,label='phase',c='b',title=''):
    ax = plt.subplot(111)
    plt.plot(f,angles,label=label,color=c)
    ax.semilogx(f, angles)
    plt.ylabel('Angle [radians]')
    plt.xlabel('Frequency [hz]')
    plt.title(title + "Phase response")
    ax.yaxis.set_major_locator(plt.MultipleLocator(mult_locater))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=denom)))

def plot_bode(x, fs):
    f,H,angles = freqz(x,fs)
    # Plot the magnitude response of a signal x.
    plot_magnitude_response(f,H)
    plt.show()
    # phase response
    plot_phase_response(f,angles)
    plt.show()

def ltspice_freqz(filename, out_label='V(vout)'):
    def imag_to_mag(z):
        # Returns the maginude of an imaginary number.
        a, b = map(float, z.split(','))
        return 20*np.log10(np.sqrt(a*a + b*b))

    def imag_to_phase(z):
        a,b = map(float,z.split(','))
        return np.arctan2(b,a)

    x = pd.read_csv(filename, delim_whitespace=True)
    x['H_dB'] = x[out_label].apply(imag_to_mag)
    x['Phase'] = x[out_label].apply(imag_to_phase)
    f = np.array(x['Freq.'])
    H_db = np.array(x['H_dB'])
    angles = np.array(x['Phase'])
    return f,H_db,angles

def plot_ltspice_bode(filename,mult_locater=(np.pi/2),denom=2):

    f,H_db,angles = ltspice_freqz(filename)
    plot_magnitude_response(f,H_db)
    plt.show()

    plot_phase_response(f,angles,mult_locater=mult_locater,denom=denom)
    plt.show()

def compare_vs_spice(x,fs,spicepath,mult_locater=(np.pi/2),denom=2,title=''):
    wdf_f, wdf_H, wdf_angles = freqz(x,fs)
    spice_f,spice_H,spice_angles = ltspice_freqz(spicepath)
    plot_magnitude_response(wdf_f,wdf_H,label='wdf')
    plot_magnitude_response(spice_f,spice_H,label='spice',c='orange',title=title)
    plt.legend()
    plt.show()

    plot_phase_response(wdf_f,wdf_angles,label='wdf')
    plot_phase_response(spice_f,spice_angles,label='spice',mult_locater=mult_locater,denom=denom,c='orange',title=title)
    plt.legend()
    plt.show()
