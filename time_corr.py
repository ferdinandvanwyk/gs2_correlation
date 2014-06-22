import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

tau = np.genfromtxt('analysis/time_fit.csv', delimiter=',')
#dt, corr_fn = pkl.load(open('analysis/corr_fn.pkl', 'rb'))
print tau.shape
#print corr_fn.shape
tau = np.array(tau)
#corr_fn = np.array(corr_fn)

plt.plot(np.transpose(tau))
plt.yscale('log')
plt.xlabel('Minor Radius')
plt.ylabel("Time window")
##plt.colorbar()
plt.show()

print 'Average = ', np.mean(np.mean(tau, axis=0))
print 'Standard Deviation = ', np.std(np.mean(tau, axis=0))

#plt.plot(dt,corr_fn[:,18,30:35])
#plt.hold(True)
#plt.plot(dt[375:575], np.exp(-dt[375:575]/tau[18]))
#plt.show()
#plt.plot(dt[275:475], corr_fn[275:475,64,30:35])
#plt.show()
