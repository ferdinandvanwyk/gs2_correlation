import numpy as np
import matplotlib.pyplot as plt

tau = np.genfromtxt('analysis/time_fitting.csv', delimiter=',')

tau_new = []
for i in range(len(tau)):
  if tau[i] > 1:
    tau_new.append(tau[i])

tau_new = np.array(tau_new)

print 'Average = ', np.mean(tau_new)
print 'Standard Deviation = ', np.std(tau_new)

p1, = plt.plot(tau)
plt.hold(True)
p2, = plt.plot(tau_new)
plt.legend([p1, p2], ['Original Tau', 'New Tau'])
plt.show()
