import numpy as np
import matplotlib.pyplot as plt

tau = np.genfromtxt('analysis/time_fit.csv', delimiter=',')
print tau.shape
tau = np.array(tau)

#print 'Average = ', np.mean(tau_new)
#print 'Standard Deviation = ', np.std(tau_new)

plt.plot(np.transpose(tau))
plt.yscale('log')
#plt.colorbar()
plt.show()
