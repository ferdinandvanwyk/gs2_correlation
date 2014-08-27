import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

tau = np.genfromtxt('analysis/time_fit.csv', delimiter=',')
print tau.shape
tau = np.array(tau)

plt.plot(np.transpose(tau[:]))
plt.xlabel('Minor Radius')
plt.ylabel("Time Correlation $(\mu s)$")
plt.show()

print 'Average = ', np.mean(np.mean(tau, axis=0))
print 'Standard Deviation = ', np.std(np.mean(tau, axis=0))
