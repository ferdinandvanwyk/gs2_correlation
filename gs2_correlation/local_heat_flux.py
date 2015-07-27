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

def field_to_real_space_no_time(field):
    """
    Converts field from (kx, ky) to (x, y) and saves as new array attribute.
    """

    field_real_space = np.empty([nx,ny,nth])
    field_real_space = np.fft.irfft2(field, axes=[0,1])
    #field_real_space = np.roll(field_real_space, int(nx/2), axis=0)

    return field_real_space

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

dens = np.array(ncfile.variables['ntot_igomega_by_mode'][:,0,:,:,:])
dens = np.swapaxes(dens, 1, 2)
dens = dens[:,:,:,0] + 1j*dens[:,:,:,1] 

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
print('part_nc = ', part_nc[-1,0])

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
        q_perp[-1,0]+q_par[-1,0]/2)

ncfile.close()
sys.exit()

# Full Final time step check
t_perp_final[1:,:,:] = t_perp_final[1:,:,:]/2
t_par_final[1:,:,:] = t_par_final[1:,:,:]/2

v_exb_final_real = field_to_real_space_no_time(1j*ky[np.newaxis,:,np.newaxis]*phi_final)*nx*ny
ntot_final_real = field_to_real_space_no_time(ntot_final)*nx*ny
#t_perp_final = field_to_real_space_no_time(t_perp_final)
#t_par_final = field_to_real_space_no_time(t_par_final)

#q_final = (t_perp_final + t_par_final + 2*ntot_final)*v_exb_final
q_final = ntot_final_real*v_exb_final_real

q_k_final = np.fft.rfft2(q_final, axes=[0,1])/nx/ny
#print(np.max(np.abs(q_final - np.fft.irfft2(q_k_final, axes=[0,1])*nx*ny)))
q_k_fsa = np.sum((dth/(bmag*np.abs(gradpar))) * q_k_final, axis=2) / \
            np.sum(dth*grho/(bmag*np.abs(gradpar)))

#print('q_final(kx=0, ky=0) = ', q_k_fsa[0,0])

# Fourier correction
phi[:,1:,:] = phi[:,1:,:]/2
dens[:,1:,:] = dens[:,1:,:]/2
t_perp[:,1:,:] = t_perp[:,1:,:]/2
t_par[:,1:,:] = t_par[:,1:,:]/2

# Convert to real space
v_exb = field_to_real_space(1j*ky*phi)
dens = field_to_real_space(dens)*rho_star
t_perp = field_to_real_space(t_perp)
t_par = field_to_real_space(t_par)

q = rhoref**2 * vth * (nref*tref) / amin**3 * ((dens*np.sqrt(t_perp**2 + t_par**2))*v_exb/2).real

# return to Q_gB
q = q / (nref * tref * vth * rhoref**2/amin**2)
q_k = np.fft.ifft2(q, axes=[1,2])
print('q(kx=0, ky=0) = ', q_k[-1,0,0])
# Only look at central third
q = q[:,int(85/3):int(2*85/3),:]
print(np.min(q), np.max(q))

# Make film

if 'film_frames' not in os.listdir('analysis/misc/'):
    os.system('mkdir analysis/misc/film_frames')
os.system("rm analysis/misc/film_frames/*.png")

for it in range(nt):
    plot_heat_flux(it, 30)

os.system("avconv -threads 2 -y -f image2 -r 10 -i 'analysis/misc/film_frames/q_vs_x_and_y_%04d.png' analysis/misc/q_vs_x_and_y.mp4")

