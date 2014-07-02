import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Normalizations
rhoref = 6.0791e-03

# Read order is [lx, ly, kx, ky]
perp = abs(np.genfromtxt('analysis/perp_fit.csv', delimiter=','))
#dt, corr_fn = pkl.load(open('analysis/corr_fn.pkl', 'rb'))
print perp.shape

plt.plot(perp[:])
plt.show()

print 'lx = ', np.mean(np.mean(perp[:,0], axis=0))*rhoref
print 'std(lx) = ', np.std(perp[:,0], axis=0)*rhoref

print 'ly = ', np.mean(np.mean(perp[:,1], axis=0))*rhoref
print 'std(ly) = ', np.std(perp[:,1])*rhoref

print 'kx = ', np.mean(np.mean(perp[:,2], axis=0))
print 'std(kx) = ', np.std(perp[:,2])

print 'ky = ', np.mean(np.mean(perp[:,3], axis=0))
print 'std(ky) = ', np.std(perp[:,3])

print 'theta = ', np.arctan(np.mean(np.mean(perp[:,2]/perp[:,3], axis=0)))

