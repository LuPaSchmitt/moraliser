# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:34:09 2020

@author: Lukas
"""
import numpy as np
from scipy.special import sph_harm, genlaguerre
from scipy.special import factorial
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp

Z   = 1  # Anzahl der Elementarladungen
a_0 = 1. # Bohrscher Atomradius

def R(n,l, r):
    """Radialfunktion"""
    Z_n = 2*Z/(n*a_0)
    root = np.sqrt(Z_n**3 * factorial(n-l-1)/(2*n*factorial(n+l)))
    rho = Z_n * r
    L = np.polyval(genlaguerre(n-l-1, 2*l+1), rho)
    return root * np.exp(-0.5*rho) * rho**l * L

def psi(n,l,m, r,theta,phi):
    R_nl = R(n,l, r)
    Y_lm = sph_harm(m,l,phi,theta) 
    return R_nl * Y_lm

def plot(n,l,m, x,z, r,theta,phi):
    plt.figure(figsize=(7,7))
    plt.title(r"$\Psi_{%i%i%i}$" % (n,l,m))
    s_nlm = np.abs(psi(n,l,m, r,theta,phi))
    plt.pcolormesh(x, z, s_nlm, cmap=plt.get_cmap("CMRmap"))
    plt.gca().set_aspect("equal") 
    plt.xlim(x.min(),x.max())
    plt.ylim(z.min(),z.max())
    
def wf(n,l,m):
        plot_range = (5*n + 5*l) * a_0
        x_1d = np.linspace(-plot_range,plot_range,1000)
        y = 0
        z_1d = np.linspace(-plot_range,plot_range,1000)
        x,z = np.meshgrid(x_1d,z_1d)
        r     = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(np.sqrt(x**2+y**2), z )
        phi   = np.arctan2(y, x)
        plot(n,l,m, x,z, r,theta,phi)
        plt.show()
        
def linkomb(n1,l1,m1,n2,l2,m2):
    plot_range = (5*n1 + 5*l1) * a_0
    x_1d = np.linspace(-plot_range,plot_range,1000)
    y = 0
    z_1d = np.linspace(-plot_range,plot_range,1000)
    x,z = np.meshgrid(x_1d,z_1d)
    r     = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z )
    phi   = np.arctan2(y, x)
    
    plt.figure(figsize=(7,7))
    plt.title(r"$\Psi_{%i%i%i} + \Psi_{%i%i%i}$" % (n1,l1,m1,n2,l2,m2))
    s_nlm = np.abs(1/np.sqrt(2)*(psi(n1,l1,m1, r,theta,phi)+psi(n2,l2,m2, r,theta,phi)))
    plt.pcolormesh(x, z, s_nlm, cmap=plt.get_cmap("CMRmap"))
    plt.gca().set_aspect("equal") 
    plt.xlim(x.min(),x.max())
    plt.ylim(z.min(),z.max())
    
def pA(n,l,m,c):
    plot_range = (5*n + 5*l) * a_0
    x=0
    y=0
    z = np.linspace(-plot_range,plot_range,1000)
    r= np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z )
    phi   = np.arctan2(y, x)
    plt.figure(figsize=(7,7))
    plt.title(r"$\Psi_{%i%i%i}$" % (n,l,m))
    s_nlm = psi(n,l,m, r,theta,phi)
    if(c):
        s_nlm=np.abs(s_nlm)
    
    plt.plot(z,s_nlm)
    
    



