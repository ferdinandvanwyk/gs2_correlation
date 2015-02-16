# Standard
import os
import sys
import gc #garbage collector
import configparser
import logging
import operator #enumerate list
import multiprocessing

# Third Party
import numpy as np
from scipy.io import netcdf
import scipy.interpolate as interp
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
import seaborn as sns
sns.set_context('talk')

def field_to_real_space(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nt,nx,ny],dtype=float)
    for it in range(nt):
        field_real_space[it,:,:] = np.fft.irfft2(field[it,:,:])
        field_real_space[it,:,:] = np.roll(field_real_space[it,:,:],
                                                int(nx/2), axis=0)

    return field_real_space*nx*ny*rho_star/2

# Normalization parameters
# Outer scale in m
amin = 0.58044
# Thermal Velocity of reference species in m/s
vth = 1.4587e+05
# Larmor radius of reference species in m
rhoref = 6.0791e-03
# Expansion parameter
rho_star = rhoref/amin
# Angle between magnetic field lines and the horizontal in radians
pitch_angle = 0.6001
# Major radius at the outboard midplane
rmaj = 1.32574
# Reference density (m^-3)
nref = 1.3180e+19
# Reference temperature (eV)
tref = 2.2054e+02

# Read NetCDF
in_file = sys.argv[1]
ncfile = netcdf.netcdf_file(in_file, 'r')
kx = ncfile.variables['kx'][:]
ky = ncfile.variables['ky'][:]
t = ncfile.variables['t'][:]
phi = np.array(ncfile.variables['phi_igomega_by_mode'][:])
phi = np.swapaxes(phi, 1, 2)
phi = phi[:,:,:,0] + 1j*phi[:,:,:,1] 
dens = np.array(ncfile.variables['ntot_igomega_by_mode'][:,0,:,:,:])
dens = np.swapaxes(dens, 1, 2)
dens = dens[:,:,:,0] + 1j*dens[:,:,:,1] 
temp = np.array(ncfile.variables['tperp_igomega_by_mode'][:,0,:,:,:])
temp = np.swapaxes(temp, 1, 2)
temp = temp[:,:,:,0] + 1j*temp[:,:,:,1] 

# Calculate sizes and real arrays
if 'analysis' not in os.listdir():
    os.system("mkdir analysis")
if 'misc' not in os.listdir('analysis'):
    os.system("mkdir analysis/misc")
nt = len(t)
nkx = len(kx)
nky = len(ky)
nx = nkx
ny = 2*(nky - 1)
t = t*amin/vth
x = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref
y = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref \
                     *np.tan(pitch_angle)

# ZF Analysis

# Need to multiply by nx since ifft contains 1/nx implicitly but 
# spectral->real for GS2 variables require no factor. Finally, zf_vel is in 
# units of (1/kxfac vth) since technically: zf_vel = kxfac*IFT[(kx*phi_imag)] 
# however kxfac calculation is nontrivial.
v_zf = np.empty([nt,nx],dtype=float)
for it in range(nt):
    v_zf[it,:] = np.fft.ifft(phi[it,:,0]*kx).imag*nx

contours = np.around(np.linspace(np.min(v_zf), np.max(v_zf), 
                                 30),2)
plt.clf()
plt.contourf(x, t*1e6, v_zf, cmap='coolwarm', levels=contours)
plt.title('$v_{ZF}(x, t))$')
plt.colorbar()
plt.xlabel(r'$ x (m)$')
plt.ylabel(r'$t (\mu s)$')
plt.savefig('analysis/misc/zf_vs_x_t.pdf')

# Mean ZF vs x
plt.clf()
plt.plot(x, np.mean(v_zf, axis=0))
plt.title('$v_{ZFi, mean}(x))$')
plt.xlabel(r'$ x (m)$')
plt.ylabel(r'$v_{ZF} (v_{th,i}/kxfac)$')
plt.savefig('analysis/misc/zf_mean_vs_x.pdf')

# Local Heat Flux 
v_exb = field_to_real_space(-ky*phi)

dens = field_to_real_space(dens)
temp = field_to_real_space(temp)

q = ((nref*temp*tref + tref*dens*nref)*v_exb).real

plt.clf()
plt.contourf(np.transpose(dens[0,:,:]), cmap='coolwarm', levels=np.linspace(np.min(dens[0,:,:]), np.max(dens[0,:,:]), 30))
plt.colorbar()
plt.show()

plt.clf()
plt.contourf(np.transpose(q[0,:,:]), cmap='coolwarm', levels=np.linspace(np.min(q[0,:,:]), np.max(q[0,:,:]), 30))
plt.colorbar()
plt.show()













