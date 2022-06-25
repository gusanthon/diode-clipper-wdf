import scipy
import numpy as np

##############################################
#### BASE ONE PORT OBJECT
##############################################


class WDFOnePort(object):
    def __init__(self):
        # Incident- and reflected wave variables.
        self.a, self.b = 0, 0
        self.parent = None

    def connect_to_parent(self, p):
        self.parent = p

    def propagate_impedance_change(self):
        self.calc_impedance()
        if self.parent != None:
            self.parent.propagate_impedance_change()

    def wave_to_voltage(self):
        voltage = (self.a + self.b) / 2
        return voltage

    def wave_to_current(self):
        current = (1 / 2 * self.Rp) * (self.a + self.b)
        return current

    def __str__(self):
        return "wdf one port"

    def __repr__(self):
        return self.__str__()

    def calc_impedance(self):
        pass

    def set_incident_wave(self, a):
        pass

    def get_reflected_wave(self):
        pass


##############################################
#### RESISTOR
##############################################


class Resistor(WDFOnePort):
    def __init__(self, R):
        WDFOnePort.__init__(self)
        self.Rval = R  # Port resistence set to physical resistance
        self.calc_impedance()

    def calc_impedance(self):
        self.Rp = self.Rval

    def set_resistance(self, R):
        if self.Rval == R:
            return
        self.Rval = R
        self.propagate_impedance_change()

    def __str__(self):
        return str(self.Rp) + " Ohm Resistor"

    def set_incident_wave(self, a):
        self.a = a

    def get_reflected_wave(self):
        self.b = 0
        return self.b


##############################################
#### CAPACITOR
##############################################


class Capacitor(WDFOnePort):
    def __init__(self, C, fs, tolerance=0, sSize=10):
        WDFOnePort.__init__(self)
        # gaussian distribution of vals below and above given C according to tolerance
        # random sample is selected and added to given capacitance value
        self.tol = tolerance
        data_normal = scipy.stats.norm.rvs(size=sSize, loc=0, scale=C * self.tol / 2)
        rand_samp = np.random.choice(data_normal)
        self.C = C + rand_samp
        self.fs = fs
        self.z = 0
        self.calc_impedance()

    def set_capacitance(self, C):
        if self.C == C:
            return
        self.C = C
        self.propagate_impedance_change()

    def calc_impedance(self):
        self.Rp = 1 / (2 * self.fs * self.C)

    def reset(self):
        self.z = 0

    def __str__(self):
        return str(self.C) + " farad Capacitor"

    def get_reflected_wave(self):
        self.b = self.z
        return self.b

    def set_incident_wave(self, a):
        self.a = a
        self.z = self.a


##############################################
#### INDUCTOR
##############################################


class Inductor(WDFOnePort):
    def __init__(self, L, fs):
        WDFOnePort.__init__(self)
        self.fs = fs
        self.L = L
        self.z = 0
        self.calc_impedance()

    def calc_impedance(self):
        self.Rp = 2 * self.fs * self.L

    def set_inductance(self, L):
        self.L = L
        self.propagate_impedance_change()

    def __str__(self):
        return str(self.L) + " henry Inductor"

    def get_reflected_wave(self):
        self.b = -self.z
        return self.b

    def set_incident_wave(self, a):
        self.a = a
        self.z = self.a


##############################################
#### SHORT CIRCUIT, OPEN CIRCUIT, SWITCH
##############################################


class ShortCircuit(WDFOnePort):
    def __init__(self):
        WDFOnePort.__init__(self)

    def get_reflected_wave(self, a):
        self.a = a
        self.b = -a
        return self.b


class OpenCircuit(WDFOnePort):
    def __init__(self):
        WDFOnePort.__init__(self)

    def get_reflected_wave(self):
        self.b = self.a
        return self.b


class Switch(WDFOnePort):
    def __init__(self):
        WDFOnePort.__init__(self)
        __state = False

    def get_reflected_wave(self):
        if __state:  # Switch closed
            self.b = -self.a
        else:  # Switch open
            self.a = self.a
            self.b = self.a

        return self.b

    def change_state(self, state):
        __state = state


##############################################
#### IDEAL VOLTAGE/CURRENT SOURCES
##############################################


class IdealVoltageSource(WDFOnePort):
    def __init__(self, next):
        WDFOnePort.__init__(self)
        self.next = next
        self.next.connect_to_parent(self)
        self.Vs = 0
        self.Rp = 0

    def __str__(self):
        return "Ideal voltage source"

    def set_voltage(self, vs):
        self.Vs = vs

    def set_incident_wave(self, a):
        self.a = a

    def get_reflected_wave(self):
        self.b = 0 - self.a * 2 * self.Vs
        return self.b


class IdealCurrentSource(WDFOnePort):
    def __init__(self):
        WDFOnePort.__init__(self)

    def __str__(self):
        return "Ideal current source"

    def get_reflected_wave(self, i_s=0):
        self.b = self.a - 2 * R_p * i_s
        return self.b


##############################################
#### RESISTIVE VOLTAGE/CURRENT SOURCES
##############################################


class ResistiveVoltageSource(WDFOnePort):
    def __init__(self, Rs):
        WDFOnePort.__init__(self)
        self.Rval = Rs
        self.v_s = 0
        self.calc_impedance()

    def calc_impedance(self):
        self.Rp = self.Rval

    def __str__(self):
        return str(self.Rp) + " Ohm resistive voltage source"

    def set_voltage(self, v_s):
        self.v_s = v_s

    def set_incident_wave(self, a):
        self.a = a

    def get_reflected_wave(self):
        self.b = self.v_s
        return self.b


class ResistiveCurrentSource(WDFOnePort):
    def __init__(self, R):
        WDFOnePort.__init__(self)
        self.Rval = R

    def calc_impedance(self):
        self.Rp = self.Rval

    def __str__(self):
        return str(self.Rp) + " Ohm resistive current source"

    def get_reflected_wave(self, i_s=0):
        self.b = self.Rp * i_s
        return self.b


##############################################
#### DIODE PAIR
##############################################


class Diode_pair(WDFOnePort):
    def __init__(self, next, nDiodes=2, Is=2.52e-9, Vt=25.85e-3, q="best"):
        WDFOnePort.__init__(self)
        self.q = q
        self.next = next
        self.next.connect_to_parent(self)
        self.set_params(Is, Vt, nDiodes)
        self.calc_impedance()

    def set_params(self, Is, Vt, nDiodes):
        self.Is = Is
        self.Rp = self.next.Rp
        self.Vt = Vt * nDiodes
        self.oneOverVt = 1.0 / self.Vt

    def calc_impedance(self):
        self.R_Is = self.Rp * self.Is
        self.R_Is_overVT = self.R_Is * self.oneOverVt
        self.logR_Is_overVT = np.log(self.R_Is_overVT)

    def set_incident_wave(self, a):
        self.a = a

    def get_reflected_wave(self):
        def omega4(x):
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

        lam = np.sign(self.a)
        lam_a_overVT = lam * self.a / self.Vt
        if self.q == "best":
            self.b = (
                self.a
                - (2 * self.Vt) * lam * (omega4(self.logR_Is_overVT + lam_a_overVT))
                - omega4(self.logR_Is_overVT - lam_a_overVT)
            )
        elif self.q == "good":
            self.b = self.a + 2 * lam * (
                self.R_Is
                - self.Vt
                * omega4(
                    self.logR_Is_overVT
                    + lam * self.a * self.oneOverVt
                    + self.R_Is_overVT
                )
            )
        return self.b


##############################################
#### ADAPTOR BASE CLASS
##############################################


class adaptor(WDFOnePort):
    def __init__(self, p1, p2):
        WDFOnePort.__init__(self)
        self.p1_reflect = 1.0
        self.p1 = p1
        self.p2 = p2
        self.p1.connect_to_parent(self)
        self.p2.connect_to_parent(self)
        self.calc_impedance()


##############################################
#### SERIES ADAPTOR
##############################################


class Series(adaptor):
    def __init__(self, p1, p2):
        adaptor.__init__(self, p1, p2)

    def calc_impedance(self):
        self.Rp = self.p1.Rp + self.p2.Rp
        self.G = 1.0 / self.Rp
        self.p1_reflect = self.p1.Rp / self.Rp

    def set_incident_wave(self, a):
        b1 = self.p1.b - self.p1_reflect * (a + self.p1.b + self.p2.b)
        self.p1.set_incident_wave(b1)
        self.p2.set_incident_wave(0 - (a + b1))
        self.a = a

    def get_reflected_wave(self):
        self.b = 0 - self.p1.get_reflected_wave() + self.p2.get_reflected_wave()
        return self.b


##############################################
#### PARALLEL ADAPTOR
##############################################


class Parallel(adaptor):
    def __init__(self, p1, p2):
        adaptor.__init__(self, p1, p2)
        self.b_temp = 0
        self.b_diff = 0

    def calc_impedance(self):
        self.G = (1.0 / self.p1.Rp) + (1.0 / self.p2.Rp)
        self.Rp = 1.0 / self.G
        self.p1_reflect = (1.0 / self.p1.Rp) / self.G

    def set_incident_wave(self, a):
        b2 = a + self.b_temp
        self.p1.set_incident_wave(self.b_diff + b2)
        self.p2.set_incident_wave(b2)
        self.a = a

    def get_reflected_wave(self):
        self.p1.get_reflected_wave()
        self.p2.get_reflected_wave()
        self.b_diff = self.p2.b - self.p1.b
        self.b_temp = 0 - self.p1_reflect * self.b_diff
        self.b = self.p2.b + self.b_temp
        return self.b
