import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Normalizations
rhoref = 6.0791e-03

# Read order is [lx, ly, kx, ky]
perp = abs(np.genfromtxt('analysis/perp_fit.csv', delimiter=','))
print perp.shape

plt.plot(perp[:])
plt.xlabel('Time Window')
plt.ylabel('Value (rho or rho^-1)')
plt.yscale('log')
plt.show()

print 'lx = ', np.mean(np.mean(perp[:,0], axis=0))*rhoref
print 'std(lx) = ', np.std(perp[:,0], axis=0)*rhoref

print 'ly = ', np.mean(np.mean(perp[:,1], axis=0))*rhoref
print 'std(ly) = ', np.std(perp[:,1])*rhoref

print 'kx = ', np.mean(np.mean(perp[:,2], axis=0))/rhoref
print 'std(kx) = ', np.std(perp[:,2])/rhoref

print 'ky = ', np.mean(np.mean(perp[:,3], axis=0))/rhoref
print 'std(ky) = ', np.std(perp[:,3])/rhoref

print 'theta = ', np.arctan(np.mean(np.mean(perp[:,2]/perp[:,3], axis=0)))
print 'std(theta) = ', np.std(np.arctan(perp[:,2]/perp[:,3]))

