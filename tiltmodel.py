#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

class Particle(object):
    def __init__(self,aspect_ratio,short_radius):
        self.alpha = aspect_ratio
        self.R_o = short_radius

    @property
    def geometric_aspect(self):
        eps = np.sqrt(1-self.alpha**-2)
        return 0.5 + ((0.5*self.alpha)/eps)*np.arcsin(eps)

    @property
    def surface_area(self):
        return 4*np.pi*self.R_o**2*self.geometric_aspect

class Interface(object):
    def __init__(self,x,y,surface_tension):
        self.x = x
        self.y = y
        self.area = 2*self.x*self.y
        self.gamma = surface_tension
        
class AreaData(object):
    def __init__(self,path,particle,interface):
        self.file = path
        self.data = np.genfromtxt(self.file)
        self.particle = particle
        self.interface = interface

    @property
    def areas(self):
        return self.data[:,1]
    
    @property
    def _angles(self):
        return self.data[:,0]

    def tilt_angles(self,rad=False):
        if rad:
            return np.deg2rad(self._angles)
        else:
            return self._angles

    @property
    def M(self):
        return np.sin(self.tilt_angles(rad=True))

    @property
    def area_removed(self):
        return self.interface.area - self.areas

    @property
    def norm(self):
        return self.interface.gamma*self.particle.surface_area 

    @property
    def area_removed_norm(self):
        return self.area_removed/self.norm

    def poly_area_model(self,order):
        p_coeffs = np.polyfit(self.tilt_angles(rad=True),self.area_removed_norm,order)
        return np.poly1d(p_coeffs)

    def magnetisation_poly_area_model(self,order):
        p_coeffs = np.polyfit(self.M,self.area_removed_norm,order)
        return np.poly1d(p_coeffs)

    def return_fit_params(self,fit_function):
        x = self.tilt_angles(rad=True)
        y = self.area_removed_norm
        fit_params,covmat = optimize.curve_fit(fit_function,x,y,maxfev=200000)
        return fit_params

    @staticmethod
    def sin_func(theta,Ro,Rp,c):
        return c + np.pi*Rp*Ro**2/np.sqrt(Ro**2*np.cos(theta)**2 + Rp**2*np.sin(theta)**2)

    def sin_area(self,Ro,Rp,c):
        def wrapper(theta):
            return AreaData.sin_func(theta,Ro,Rp,c)
        return wrapper
        
    @property
    def sin_area_model(self):
        (Ro,Rp,c) = self.return_fit_params(AreaData.sin_func)
        return self.sin_area(Ro,Rp,c)

    def empirical_free_energy(self,area_model_function,unnormalised_b_field):
        B_norm = unnormalised_b_field/self.norm
        def wrapper(theta):
            return -area_model_function(theta) - B_norm*np.sin(theta)
        return wrapper

    def relative_empirical_free_energy(self,area_model_function,unnormalised_b_field,rel_angle):
        B_norm = unnormalised_b_field/self.norm
        free_energy_func = self.empirical_free_energy(area_model_function,unnormalised_b_field)
        def wrapper(theta):
            return free_energy_func(theta) - free_energy_func(rel_angle)
        return wrapper

    def _invert_closure(self,fn):
        def wrapper(theta):
            return -fn(theta)
        return wrapper

    def minimize_free_energy(self,free_energy_func,guess):
        F_theta = free_energy_func
        F_min = optimize.minimize(F_theta,guess)
        min_theta = F_min.x[0]
        return min_theta

    def fit_area_derivative(self,area_model,sample_dist,order):
        theta = np.arange(0,np.pi/2,sample_dist)
        y = np.gradient(area_model(theta),sample_dist)
        p_coeffs = np.polyfit(theta,y,order)
        return np.poly1d(p_coeffs)

    def free_energy_deriv(self,area_model_deriv,unnormalised_b_field):
        def B(theta):
            return -self.interface.gamma*area_model_deriv(theta) - unnormalised_b_field/self.norm*np.cos(theta)
        return B

    def minimise_free_energy_deriv(self,free_energy_deriv,guess):
        min = optimize.fsolve(free_energy_deriv,guess,full_output=True)

        if min[2] == 1:
            min_angle = min[0][0]
        else:
            min_angle = None
        return min_angle


        
class AnalyticalTiltModel(object):

    def __init__(self,particle,interface):
        self.p = particle
        self.I = interface

    def normalised_free_energy(self,unnormalised_b_field):
        def closure(theta):
            pref_term  = self.p.alpha/(4*self.p.geometric_aspect)
            sqrt_term  = 1/np.sqrt(np.cos(theta)**2 + self.p.alpha**2*np.sin(theta)**2)
            field_term = unnormalised_b_field*np.cos(np.pi/2 - theta)/(self.p.surface_area*self.I.gamma)
            return -(pref_term*sqrt_term + field_term)
        return closure

    def relative_normalised_free_energy(self,unnormalised_b_field,rel_angle):
        free_energy_func = self.normalised_free_energy(unnormalised_b_field)
        def wrapper(theta):
            return free_energy_func(theta) - free_energy_func(rel_angle)
        return wrapper

    def normalised_free_energy_magnetisation(self,unnormalised_b_field):
        def closure(M):
            pref_term  = self.p.alpha/(4*self.p.geometric_aspect)
            sqrt_term  = 1/np.sqrt(1 + M**2*(self.p.alpha**2 - 1))
            field_term = unnormalised_b_field*M/(self.p.surface_area*self.I.gamma)
            return -(pref_term*sqrt_term + field_term)
        return closure
        
    def relative_normalised_free_energy_magnetisation(self,unnormalised_b_field,rel_mag):
        free_energy_func = self.normalised_free_energy_magnetisation(unnormalised_b_field)
        def wrapper(M):
            return free_energy_func(M) - free_energy_func(rel_mag)
        return wrapper

    def minimize_free_energy(self,free_energy_func,guess):
        F_min = optimize.minimize(free_energy_func,guess)
        min_M = F_min.x[0]
        return min_M
        
    @property
    def norm(self):
        return self.p.surface_area*self.I.gamma

if __name__=="__main__":

    mins = []
    p = Particle(2.0,1.0)
    I = Interface(1,1,1)

    print p.surface_area

    M = np.arange(0,1.01,0.01)
    print M

    tiltmodel = AnalyticalTiltModel(p,I)

    B = [0,1,2,3,4,5,6,7]
    B = np.arange(0,7,0.1)

    for b_field in B:
        free_energy = tiltmodel.relative_normalised_free_energy_magnetisation(b_field,1.0)
        mins.append(tiltmodel.minimize_free_energy(free_energy,0.0))

#         plt.plot(M,free_energy(M),'--')
#     plt.grid()
#     plt.show()

    mins = np.array(mins)
    mins[np.where(mins > 1)] = 1
    
    plt.plot(B/tiltmodel.norm,mins,'ro')
    plt.show()
