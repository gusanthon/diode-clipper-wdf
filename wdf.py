import numpy as np
import scipy.stats

##########################################################################################################################################
########### WDF ELEMENT CLASSES 
##########################################################################################################################################

class base_wdf():
    def __init__(self):
        self.a, self.b = 0, 0
        self.parent = None

    def connect_to_parent(self,p):
        self.parent = p

    def calc_impedance(self):
        pass

    def impedance_change(self):
        self.calc_impedance()
        if self.parent != None:
            self.parent.impedance_change()

    def wave_to_voltage(self):
        return (self.a + self.b) / 2.

    def propagate_reflected_wave(self):
        pass

    def accept_incident_wave(self,a):
        self.a = a

class root_wdf(base_wdf):
    def __init__(self):
        base_wdf.__init__(self)
        self.next = None

    def connect_to_parent(self, p):
        pass

##########################################################################################################################################

class Resistor(base_wdf):
    def __init__(self,R=1e-9):
        base_wdf.__init__(self)
        self.Rp = R
        self.calc_impedance()

    def set_resistance(self,new_R):
        if self.Rp == new_R:
            return
        self.Rp = new_R
        self.impedance_change()

    def calc_impedance(self):
        self.G = 1./self.Rp

    def propagate_reflected_wave(self):
        self.b = 0
        return self.b

##########################################################################################################################################

class Capacitor(base_wdf):
    def __init__(self,C,fs, tolerance = 0, sample_size = 20):
        base_wdf.__init__(self)
        self.fs = fs
        self.tolerance = tolerance
        data_normal = scipy.stats.norm.rvs(size = sample_size,loc = 0, scale = C * self.tolerance / 2)
        rand_samp = np.random.choice(data_normal)
        self.C = C + rand_samp
        self.z = 0
        self.calc_impedance()

    def prepare(self,new_fs):
        self.fs = new_fs
        self.impedance_change()
        self.reset()

    def set_capacitance(self,new_C):
        if self.C == new_C:
            return
        self.C = new_C
        self.impedance_change()

    def calc_impedance(self):
        self.Rp = 1./ (2 * self.C * self.fs)
        self.G = 1./self.Rp

    def accept_incident_wave(self,a):
        self.a = a
        self.z = self.a

    def propagate_reflected_wave(self):
        self.b = self.z
        return self.b

    def reset(self):
        self.z = 0

##########################################################################################################################################

class Inductor(base_wdf):
    def __init__(self,L,fs):
        self.fs = fs
        self.L = L
        self.z = 0
        self.calc_impedance()

    def prepare(self,new_fs):
        self.fs = new_fs
        self.impedance_change()
        self.reset()

    def set_inductance(self,new_L):
        if self.L == new_L:
            return
        self.L = new_L
        self.impedance_change()

    def calc_impedance(self):
        self.Rp = 2 * self.L * self.fs
        self.G = 1./self.Rp

    def accept_incident_wave(self,a):
        self.a = a
        self.z = self.a 

    def propagate_reflected_wave(self):
        self.b = -self.z 
        return self.b

    def reset(self):
        self.z = 0 

##########################################################################################################################################

class Parallel_adaptor(base_wdf):
    def __init__(self,p1,p2):
        base_wdf.__init__(self)
        self.p1 = p1
        self.p2 = p2
        self.b_temp = 0
        self.b_diff = 0
        self.p1_reflect = 1
        p1.connect_to_parent(self)
        p2.connect_to_parent(self)
        self.calc_impedance()

    def calc_impedance(self):
        self.G = self.p1.G + self.p2.G
        self.Rp = 1./self.G
        self.p1_reflect = self.p1.G / self.G

    def accept_incident_wave(self,a):
        b2 = a + self.b_temp
        self.p1.accept_incident_wave(self.b_diff + b2)
        self.p2.accept_incident_wave(b2)
        self.a = a

    def propagate_reflected_wave(self):
        self.p1.propagate_reflected_wave()
        self.p2.propagate_reflected_wave()
        self.b_diff = self.p2.b - self.p1.b 
        self.b_temp = -self.p1_reflect * self.b_diff
        self.b = self.p2.b + self.b_temp
        return self.b

##########################################################################################################################################

class Series_adaptor(base_wdf):
    def __init__(self,p1,p2):
        base_wdf.__init__(self)
        self.p1 = p1
        self.p2 = p2
        self.p1_reflect = 1
        p1.connect_to_parent(self)
        p2.connect_to_parent(self)
        self.Rp = self.p1.Rp + self.p2.Rp
        self.calc_impedance()
        
    def calc_impedance(self):
        self.G = 1./self.Rp
        self.p1_reflect = self.p1.Rp / self.Rp

    def accept_incident_wave(self,a):
        b1 = self.p1.b - self.p1_reflect * (a + self.p1.b + self.p2.b)
        self.p1.accept_incident_wave(b1)
        self.p2.accept_incident_wave(0 - (a + b1))
        self.a = a

    def propagate_reflected_wave(self):
        self.b = -(self.p1.propagate_reflected_wave() + self.p2.propagate_reflected_wave())
        return self.b

##########################################################################################################################################

class Polarity_inverter(base_wdf):
    def __init__(self,p1):
        base_wdf.__init__(self)
        p1.connect_to_parent(self)
        self.p1 = p1
        self.calc_impedance()

    def calc_impedance(self):
        self.Rp = self.p1.Rp 
        self.G = 1./self.Rp

    def accept_incident_wave(self,a):
        self.a = a
        self.p1.accept_incident_wave(-a)

    def propagate_reflected_wave(self):
        self.b = 0 - self.p1.propagate_reflected_wave()

##########################################################################################################################################

class Ideal_voltage_source(root_wdf):
    def __init__(self,next):
        root_wdf.__init__(self)
        self.next = next
        self.Vs = 0
        self.calc_impedance()

    def calc_impedance(self):
        pass

    def set_voltage(self,new_V):
        self.Vs = new_V

    def propagate_reflected_wave(self):
        self.b = 0 - self.a + 2 * self.Vs
        return self.b

##########################################################################################################################################

class Resistive_voltage_source(base_wdf):
    def __init__(self,Rval: float = None):
        base_wdf.__init__(self)
        self.Rval = Rval if Rval else 1e-9
        self.Vs = self.calc_impedance()

    def set_resistance(self,new_R):
        if self.Rval == new_R:
            return
        self.Rval = new_R
        self.impedance_change()

    def calc_impedance(self):
        self.Rp = self.Rval
        self.G = 1./self.Rp

    def set_voltage(self,new_V):
        self.Vs = new_V

    def propagate_reflected_wave(self):
        self.b = self.Vs 
        return self.b

##########################################################################################################################################

class Ideal_current_source(root_wdf):
    def __init__(self,next):
        root_wdf.__init__(self)
        self.next = next
        self.Is = 0
        self.next.connect_to_parent(self)
        self.calc_impedance()

    def calc_impedance(self):
        self.two_R = 2 * self.next.Rp 
        self.two_R_Is = self.two_R * self.Is

    def propagate_reflected_wave(self):
        self.b = self.two_R_Is + self.a
        return self.b

##########################################################################################################################################

class Resistive_current_source(base_wdf):
    def __init__(self,Rval: float = None):
        base_wdf.__init__(self)
        self.Is = 0
        self.Rval = Rval if Rval else 1e9
        self.calc_impedance()

    def set_resistance(self,new_R):
        if self.Rval == new_R:
            return
        self.Rval = new_R
        self.impedance_change()

    def calc_impedance(self):
        self.Rp = self.Rval
        self.G = 1./self.Rp

    def set_current_(self,new_I):
        self.Is = new_I

    def propagate_reflected_wave(self):
        self.b = self.Rp * self.Is
        return self.b

##########################################################################################################################################

class Diode(root_wdf):
    def __init__(self,next,Is,Vt=25.85e-3,n_diodes=1):
        root_wdf.__init__(self)
        self.next = next
        next.connect_to_parent(self)
        self.set_diode_params(Is,Vt,n_diodes)

    def set_diode_params(self,Is,Vt,n_diodes):
        self.Is = Is
        self.Vt = Vt * n_diodes
        self.one_over_Vt = 1./self.Vt
        self.calc_impedance()

    def calc_impedance(self):
        self.two_R_Is = 2 * self.next.Rp * self.Is
        self.R_Is_over_Vt = self.next.Rp * self.Is * self.one_over_Vt
        self.logR_Is_over_Vt = np.log(self.R_Is_over_Vt)
    
    def omega4(self,x):
        x1 = -3.341459552768620
        x2 = 8.0
        a = -1.314293149877800e-3
        b = 4.775931364975583e-2
        c = 3.631952663804445e-1
        d = 6.313183464296682e-1
        if x < x1:
            y = 0
        elif x < x2:
            y = d + x * (c + x * (b + x * a))
        else:
            y = x - np.log(x)
        return y - (y - np.exp(x - y) / (y + 1))

    def propagate_reflected_wave(self):
        self.b = self.a + self.two_R_Is - (2 * self.Vt) * self.omega4(self.logR_Is_over_Vt + self.a * self.one_over_Vt + self.R_Is_over_Vt)
        return self.b

class Diode_pair(Diode):
    def __init__(self,next,Is,Vt=25.85e-3,n_diodes=2):
        Diode.__init__(self,next,Is,Vt,n_diodes)

    def propagate_reflected_wave(self):
        lam = np.sign(self.a)
        lam_a_over_Vt = lam * self.a * self.one_over_Vt
        self.b = self.a - (2 * self.Vt) * lam * (self.omega4(self.logR_Is_over_Vt + lam_a_over_Vt) - self.omega4(self.logR_Is_over_Vt - lam_a_over_Vt))
        return self.b

##########################################################################################################################################
########### CIRCUIT CLASSES 
##########################################################################################################################################

class Diode_clipper():

    def __init__(self,sample_rate,cutoff=1000,input_gain_db=0,output_gain_db=0,n_diodes=2) -> None:
        self.fs = sample_rate
        self.cutoff = cutoff
        self.input_gain = 10**(input_gain_db/20)
        self.output_gain = 10**(output_gain_db/20)

        self.C = 47e-9
        self.R = 1./ (2 * np.pi * self.C * self.cutoff)

        self.R1 = Resistor(self.R)
        self.Vs = Resistive_voltage_source()

        self.S1 = Series_adaptor(self.Vs,self.R1)
        self.C1 = Capacitor(self.C,self.fs)

        self.P1 = Parallel_adaptor(self.S1,self.C1)
        self.Dp = Diode_pair(self.P1,2.52e-9,n_diodes=n_diodes)

    def process(self,sample):
        sample *= self.input_gain
        self.Vs.set_voltage(sample)
        self.Dp.accept_incident_wave(self.P1.propagate_reflected_wave())
        self.P1.accept_incident_wave(self.Dp.propagate_reflected_wave())
        return -(self.C1.wave_to_voltage() * self.output_gain)

    def __call__(self, *args: any, **kwds: any) -> any:
        return self.process(args[0])

    def set_cutoff(self,new_cutoff):
        self.cutoff = new_cutoff
        self.R = 1./ 2 * np.pi * self.cutoff * self.C
        self.R1.set_resistance(self.R)

    def set_input_gain(self,gain_db):
        self.input_gain = 10**(gain_db/20)

    def set_output_gain(self,gain_db):
        self.output_gain = 10**(gain_db/20)

##########################################################################################################################################

class Passive_LPF():

    def __init__(self,sample_rate,cutoff=1000) -> None:
        self.fs = sample_rate
        self.cutoff = cutoff

        self.Z = 800
        self.C = (1 / self.Z) / (2. * np.pi * cutoff)

        self.R1 = Resistor(10)
        self.R2 = Resistor(1e4)

        self.C1 = Capacitor(self.C,self.fs)
        self.C2 = Capacitor(self.C,self.fs)

        self.S1 = Series_adaptor(self.R2,self.C2)
        self.P1 = Parallel_adaptor(self.C1,self.S1)
        self.S2 = Series_adaptor(self.R1, self.P1)

        self.Vs = Ideal_voltage_source(self.P1)

    def process(self,sample):
        self.Vs.set_voltage(sample)
        self.Vs.accept_incident_wave(self.S2.propagate_reflected_wave())
        self.S2.accept_incident_wave(self.Vs.propagate_reflected_wave())
        return self.C2.wave_to_voltage()

    def __call__(self, *args: any, **kwds: any) -> any:
        return self.process(args[0])

    def set_cutoff(self,new_cutoff):
        self.cutoff = new_cutoff
        self.C = (1. /self.Z) / (2 * np.pi * self.cutoff)
        self.C1.set_capacitance(self.C)
        self.C2.set_capacitance(self.C)