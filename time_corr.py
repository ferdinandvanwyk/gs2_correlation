import numpy as np
import matplotlib.pyplot as plt

tau = np.genfromtxt('analysis/time_fit.csv', delimiter=',')
print tau.shape
tau = np.array(tau)

#print 'Average = ', np.mean(tau_new)
#print 'Standard Deviation = ', np.std(tau_new)

plt.contourf(np.log(tau))
plt.xlabel('Minor Radius')
plt.ylabel("Time window")
plt.colorbar()
plt.show()
