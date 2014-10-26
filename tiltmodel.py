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
    def __init__(self,path):
        self.file = path
        self.data = np.genfromtxt(self.file)

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

    def area_removed(self,interface):
        return interface.area - self.areas

    def area_removed_norm(self,particle,interface):
        norm =  particle.surface_area*interface.gamma
        return self.area_removed(interface)/norm

    def poly_area(self,order):
        p_coeffs = np.polyfit(self.tilt_angles(rad=True),self.normalised_areas,order)
        return np.poly1d(p_coeffs)
        

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

    def relative_normalised_free_energy(self,unnormalised_b_field,angle):
        free_energy_func = self.normalised_free_energy(unnormalised_b_field)
        def wrapper(theta):
            return free_energy_func(theta) - free_energy_func(angle)
        return wrapper

if __name__=="__main__":

    p = Particle(3.0,1.0)
    I = Interface(40,20,1)

    print p.surface_area

    theta = np.arange(np.rad2deg(0),np.deg2rad(90),0.01)

    tiltmodel = AnalyticalTiltModel(p,1)

    b_field = 0
#     for b_field in [0,1,2,3,4,5,6,7]:
#         free_energy = tiltmodel.relative_normalised_free_energy(b_field,np.pi/2)
#         plt.plot(theta,free_energy(theta),'--')
#     plt.show()

    p2 = Particle(2.0,5.0)
    

    d = AreaData('1p.txt')
    
    A_rm = d.area_removed_norm(p,I)

    plt.plot(theta,A_rm(theta),'ro')
    plt.show()

