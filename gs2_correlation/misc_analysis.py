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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_context('talk')

#local
import plot_style

def field_to_real_space(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nt,nx,ny],dtype=float)
    field_real_space = np.fft.irfft2(field, axes=[1,2])
    field_real_space = np.roll(field_real_space, int(nx/2), axis=1)

    return field_real_space*nx*ny

def plot_heat_flux(it, plot_lim):
    """
    Plots real space field and saves as a file indexed by time index.

    Parameters
    ----------
    it : int
        Time index to plot and save.
    """
    print('Saving frame %d of %d'%(it,nt))

    contours = np.around(np.linspace(-plot_lim,plot_lim,41),5)
    cbar_ticks = np.around(np.linspace(-plot_lim,plot_lim,5),7) 

    plt.clf()
    ax = plt.subplot(111)
    im = ax.contourf(x[int(len(x)/3):int(2*len(x)/3)], y, np.transpose(q[it,:,:]),levels=contours, cmap='seismic')
    plt.xlabel(r'$x (m)$')
    plt.ylabel(r'$y (m)$')
    plt.title(r'Time = %04d $\mu s$'%(int(np.round((t[it]-t[0])*1e6))))
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plot_style.ticks_bottom_left(ax)
    plt.colorbar(im, cax=cax, label=r'$Q_{ion} (Q_{gB})$', 
                 ticks=cbar_ticks, format='%.2f')
    plt.savefig("analysis/misc/film_frames/q_vs_x_and_y_%04d.png"%it, dpi=110)

def field_to_real_space_no_time(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nx,ny,nth])
    field_real_space = np.fft.irfft2(field, axes=[0,1])
    #field_real_space = np.roll(field_real_space, int(nx/2), axis=0)

    return field_real_space

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
# Reference temperature (kT)
tref = 2.2054e+02 / 8.6173324e-5 * 1.38e-23

# Read NetCDF
in_file = sys.argv[1]
ncfile = netcdf.netcdf_file(in_file, 'r')
kx = np.array(ncfile.variables['kx'][:])
ky = np.array(ncfile.variables['ky'][:])
th = np.array(ncfile.variables['theta'][:])
t = np.array(ncfile.variables['t'][:])
gradpar = np.array(ncfile.variables['gradpar'][:])
grho = np.array(ncfile.variables['grho'][:])
bmag = np.array(ncfile.variables['bmag'][:])
dth = np.append(np.diff(th), 0)

phi = np.array(ncfile.variables['phi_igomega_by_mode'][:])
phi = np.swapaxes(phi, 1, 2)
phi = phi[:,:,:,0] + 1j*phi[:,:,:,1] 

ntot = np.array(ncfile.variables['ntot_igomega_by_mode'][:,0,:,:,:])
ntot = np.swapaxes(ntot, 1, 2)
ntot = ntot[:,:,:,0] + 1j*ntot[:,:,:,1] 

t_perp = np.array(ncfile.variables['tperp_igomega_by_mode'][:,0,:,:,:])
t_perp = np.swapaxes(t_perp, 1, 2)
t_perp = t_perp[:,:,:,0] + 1j*t_perp[:,:,:,1] 

t_par = np.array(ncfile.variables['tpar_igomega_by_mode'][:,0,:,:,:])
t_par = np.swapaxes(t_par, 1, 2)
t_par = t_par[:,:,:,0] + 1j*t_par[:,:,:,1] 

q_nc = np.array(ncfile.variables['es_heat_flux'][:])
part_nc = np.array(ncfile.variables['es_part_flux'][:])
q_perp = np.array(ncfile.variables['es_heat_flux_perp'][:])
q_par = np.array(ncfile.variables['es_heat_flux_par'][:])
q_by_ky = np.array(ncfile.variables['total_es_heat_flux_by_ky'][:])

t_perp_final = np.array(ncfile.variables['tperp'][0,:,:,:,:])
t_perp_final = np.swapaxes(t_perp_final, 0, 1)
t_perp_final = t_perp_final[:,:,:,0] + 1j*t_perp_final[:,:,:,1] 

t_par_final = np.array(ncfile.variables['tpar'][0,:,:,:,:])
t_par_final = np.swapaxes(t_par_final, 0, 1)
t_par_final = t_par_final[:,:,:,0] + 1j*t_par_final[:,:,:,1] 

ntot_final = np.array(ncfile.variables['ntot'][0,:,:,:,:])
ntot_final = np.swapaxes(ntot_final, 0, 1)
ntot_final = ntot_final[:,:,:,0] + 1j*ntot_final[:,:,:,1] 

phi_final = np.array(ncfile.variables['phi'][:,:,:,:])
phi_final = np.swapaxes(phi_final, 0, 1)
phi_final = phi_final[:,:,:,0] + 1j*phi_final[:,:,:,1] 

# Calculate sizes and real arrays
if 'analysis' not in os.listdir():
    os.system("mkdir analysis")
if 'misc' not in os.listdir('analysis'):
    os.system("mkdir analysis/misc")
nt = len(t)
nkx = len(kx)
nky = len(ky)
nth = len(th)
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
kxfac = 1.9574
v_zf = np.empty([nt,nx],dtype=float)
for it in range(nt):
    v_zf[it,:] = 0.5*kxfac*np.fft.ifft(phi[it,:,0]*kx).imag*nx

contours = np.around(np.linspace(np.min(v_zf), np.max(v_zf), 
                                 30),2)
plt.clf()
plt.contourf(x, t*1e6, v_zf, cmap='coolwarm', levels=contours)
plt.title('$v_{ZF}(x, t)$')
plt.colorbar()
plt.xlabel(r'$ x (m)$')
plt.ylabel(r'$t (\mu s)$')
plt.savefig('analysis/misc/zf_vs_x_t.pdf')

# Mean ZF vs x
plt.clf()
plt.plot(x, np.mean(v_zf, axis=0))
plt.plot(x, np.mean(v_zf, axis=0) + (x - x[int(nx/2)])/rhoref*0.16)
plt.title('$v_{ZF, mean}(x)$')
plt.xlabel(r'$ x (m)$')
plt.ylabel(r'$v_{ZF} (arb. units)$')
plt.savefig('analysis/misc/zf_mean_vs_x.pdf')

##################################
# Repeat GS2 calculation exactly #
##################################

wgt = np.sum(dth*grho/bmag/gradpar)
dnorm = dth/bmag/gradpar

part_gs2 = np.empty([nkx, nky])
for ikx in range(nkx):
    for iky in range(nky):
        part_gs2[ikx, iky] = np.sum((ntot_final[ikx,iky,:]* \
                                np.conj(phi_final[ikx,iky,:])*ky[iky]*dnorm).imag)/wgt
part_gs2 *= 0.5
part_gs2[:,1:] /= 2

print('part calc = ', np.sum(part_gs2))
print('part_nc = ', part_nc[-1,0], '\n')

##################################
# Repeat GS2 calculation exactly #
##################################

q_gs2 = np.empty([nkx, nky])
for ikx in range(nkx):
    for iky in range(nky):
        q_gs2[ikx, iky] = np.sum(((t_perp_final[ikx,iky,:] + 
                          t_par_final[ikx,iky,:]/2 + 3/2*ntot_final[ikx,iky,:])* \
                                np.conj(phi_final[ikx,iky,:])*ky[iky]*dnorm).imag)/wgt
q_gs2 *= 0.5
q_gs2[:,1:] /= 2

print('Q calc = ', np.sum(q_gs2))
print('q_final_gs2 = ', q_nc[-1,0], q_perp[-1,0], q_par[-1,0]/2, 
        q_perp[-1,0]+q_par[-1,0]/2, '\n')

####################################
# Now do calculation in real space #
####################################

#########################
# Final time step first #
#########################
wgt = np.sum(dth*grho/bmag/gradpar)
dnorm = dth/bmag/gradpar

phi_final[:,1:,:] = phi_final[:,1:,:]/2
ntot_final[:,1:,:] = ntot_final[:,1:,:]/2
t_perp_final[:,1:,:] = t_perp_final[:,1:,:]/2
t_par_final[:,1:,:] = t_par_final[:,1:,:]/2

v_exb_final_real = field_to_real_space_no_time(1j*ky[np.newaxis,:,np.newaxis]*phi_final)*nx*ny
ntot_final_real = field_to_real_space_no_time(ntot_final)*nx*ny
t_perp_final_real = field_to_real_space_no_time(t_perp_final)*nx*ny
t_par_final_real = field_to_real_space_no_time(t_par_final)*nx*ny

q_final = (t_perp_final_real + t_par_final_real/2 + 3/2*ntot_final_real)*v_exb_final_real
part_final = ntot_final_real*v_exb_final_real

part_k_final = np.fft.rfft2(part_final, axes=[0,1])/nx/ny
q_k_final = np.fft.rfft2(q_final, axes=[0,1])/nx/ny

q_k_fsa = np.sum(dnorm * q_k_final, axis=2) / wgt
part_k_fsa = np.sum(dnorm * part_k_final, axis=2) / wgt

print('part_final(kx=0, ky=0) = ', part_k_fsa[0,0].real/2)
print('q_final(kx=0, ky=0) = ', q_k_fsa[0,0].real/2, '\n')

#############################
# Outboard midplane in time #
#############################

# Fourier correction
phi[:,:,1:] = phi[:,:,1:]/2
ntot[:,:,1:] = ntot[:,:,1:]/2
t_perp[:,:,1:] = t_perp[:,:,1:]/2
t_par[:,:,1:] = t_par[:,:,1:]/2

# Convert to real space
v_exb = field_to_real_space(1j*ky*phi)
ntot = field_to_real_space(ntot)
t_perp = field_to_real_space(t_perp)
t_par = field_to_real_space(t_par)

q = ((t_perp + t_par/2 + 3/2*ntot)*v_exb).real/2

q_k = np.fft.rfft2(q, axes=[1,2])/nx/ny
print('q_k(0,0) = ', q_k[-1,0,0])
# Only look at central third
q = q[:,int(85/3):int(2*85/3),:]/nx/ny
print(np.min(q), np.max(q))

# Make film

if 'film_frames' not in os.listdir('analysis/misc/'):
    os.system('mkdir analysis/misc/film_frames')
os.system("rm analysis/misc/film_frames/*.png")

for it in range(nt):
    plot_heat_flux(it, 0.08)

os.system("avconv -threads 2 -y -f image2 -r 30 -i 'analysis/misc/film_frames/q_vs_x_and_y_%04d.png' -q 1 analysis/misc/q_vs_x_and_y.mp4")












