#This file contains functions which generate films
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#Make film of a 2D function
def film_2d(xpts, ypts, field, nt, field_name):
  files = []
  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(111)
  os.system("mkdir film_frames")
  for it in range(nt):
        ax.cla()
        ax.contourf(xpts, ypts, np.transpose(field[it,:,:]), levels=np.linspace(-0.4,1,11))
        plt.xlabel(r'$\Delta x (\rho_i)$')
        plt.ylabel(r'$\Delta y (\rho_i)$')
        fname = "film_frames/"+field_name+"_%04d.png"%it
        print 'Saving frame = ', fname
        fig.savefig(fname)
        files.append(fname)

  print 'Making movie animation.mp4'
  os.system("ffmpeg -threads 2 -y -f image2 -r 40 -i 'film_frames/"+field_name+"_%04d.png' "+field_name+".mp4")
