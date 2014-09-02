#This file contains functions which generate films
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#Make film of a 2D function
def film_2d(xpts, ypts, field, nt, field_name):
  files = []
  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(111)
  os.system("mkdir analysis/film_frames")
  for it in range(nt):
        ax.cla()
        ax.contourf(xpts, ypts, np.transpose(field[it,:,:]), levels=np.linspace(-0.4,1,11))
        plt.xlabel(r'$\Delta x (\rho_i)$')
        plt.ylabel(r'$\Delta y (\rho_i)$')
        fname = "analysis/film_frames/"+field_name+"_%04d.jpg"%it
        print 'Saving frame = ', fname
        fig.savefig(fname)
        files.append(fname)

  print 'Making movie animation.mp4'
  os.system("ffmpeg -threads 2 -y -f image2 -r 40 -i 'analysis/film_frames/"+field_name+"_%04d.jpg' analysis/"+field_name+".mp4")

#Make film of a 2D function
def real_space_film_2d(xpts, ypts, field, field_name):

  #Need max and min values of field to set film levels
  field_max = np.max(field)
  field_min = np.min(field)

  files = []
  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(111)
  os.system("mkdir analysis/film_frames")
  for it in range(field.shape[0]):
        ax.cla()
        ax.contourf(xpts, ypts, np.transpose(field[it,:,:]), levels=np.linspace(field_min,field_max,30), cmap=plt.cm.afmhot)
        plt.xlabel(r'$x (m)$')
        plt.ylabel(r'$y (m)$')
        fname = "analysis/film_frames/"+field_name+"_%04d.png"%it
        print 'Saving frame = ', fname
        fig.savefig(fname)
        files.append(fname)

  print 'Making movie animation.mp4'
  os.system("ffmpeg -threads 2 -y -f image2 -r 40 -i 'analysis/film_frames/"+field_name+"_%04d.png' analysis/"+field_name+".mp4")
