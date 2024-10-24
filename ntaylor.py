""" NTaylor: Normalized Taylor Diagrams

This module allows to generate normalized Taylor diagrams

References:
Taylor, K. E. (2001). Summarizing multiple aspects of model performance in a single diagram. Journal of Geophysical Research: Atmospheres, 106, 7183–7192. doi: 10.1029/2000JD900719
Kärnä, T., & Baptista, A. M. (2016). Evaluation of a long-term hindcast simulation for the Columbia River estuary. Ocean Modelling, 99, 1–14. doi: 10.1016/j.ocemod.2015.12.007

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



################################################################################
# get statistics ###############################################################
################################################################################

def get_stat(x0, x):

  """ Calculate mean, bias, root mean square error, centered root mean square error, standard deviation and correlation coefficient of x (test dataset) in comparison with x0 (reference dataset)

  Required parameters:
  x0 (NumPy array of shape (n)): reference dataset
  x (NumPy array of shape (n) or (m, n)): test dataset(s)

  Returns:
  NumPy array of shape (6): statistics of the reference dataset
  NumPy array of shape (m, 6): statistics of the test dataset (m = 1 if x of shape (n))

  """

  # number of data
  n = x0.shape[0]

  # mean of the reference
  s0 = np.mean(x0)

  # standard deviation of the reference
  sigma0 = np.sqrt(np.sum((x0 - s0) ** 2.) / n)

  # all statistics
  stat0 = np.array([s0, 0., 0., 0., sigma0, 1.])

  # reshape x if only 1 dimension
  if x.ndim == 1:
    x = np.reshape(x, (1, x.shape[0]))

  # initialization
  stat = np.zeros((x.shape[0], 6))

  # for each dataset
  for i in range(x.shape[0]):

    # dataset #i
    xi = x[i, :]

    # mean
    s = np.mean(xi)

    # bias
    bias = s - s0

    # root mean square error
    rmse = np.sqrt(np.sum((xi - x0) ** 2.) / n)

    # centered root mean square error
    crmse = np.sqrt(np.sum(((xi - s) - (x0 - s0)) ** 2.) / n)

    # standard deviation
    sigma = np.sqrt(np.sum((xi - s) ** 2.) / n)

    # correlation coefficient
    r = np.sum((xi - s) * (x0 - s0)) / (n * sigma * sigma0)

    # all statistics
    stat[i, 0] = s
    stat[i, 1] = bias
    stat[i, 2] = rmse
    stat[i, 3] = crmse
    stat[i, 4] = sigma
    stat[i, 5] = r

  return stat0, stat



################################################################################
# get normalized statistics ####################################################
################################################################################

def get_statn(x0, x):

  """ Calculate mean, bias, root mean square error, normalized centered root mean square error, normalized standard deviation and correlation coefficient of x (test dataset), in comparison with x0 (reference dataset)

  Required parameters:
  x0 (NumPy array of shape (n)): reference dataset
  x (NumPy array of shape (n) or (m, n)): test dataset(s)

  Returns:
  NumPy array of shape (m, 6): normalized statistics of the test dataset (m = 1 if x of shape (n))

  """

  # all statistics
  [stat0, stat] = get_stat(x0, x)

  # normalized statistics
  statn = stat
  statn[:, 3] /= stat0[4]
  statn[:, 4] /= stat0[4]

  return statn



################################################################################
# normalized diagram ###########################################################
################################################################################

def diagn(ax, statn, prop, sigma_lim = 1.5, sigma_color = 'k',
          crmse_color = 'C8', r_color = 'C9', no_sigma_axis = False,
          no_crmse_axis = False, r_label = 'correlation coefficient',
          target_color = 'k'):

  """ Create a normalized Taylor diagram

  Required parameters:
  ax (figure axis)
  statn (NumPy array of shape (m, 6)): normalized statistics from get_statn()
  prop: (NumPy array of shape (m, 3) and type object):
    --> prop[i, :] = [marker, markerfacecolor, markeredgecolor] of dataset i

  # Optional parameters
  sigma_lim (float, default = 1.5): maximum displayed value of the normalized standard deviation
  sigma_color (color, default is 'k'): color for the normalized standard deviation on the diagram
  crmse_color (color, default is 'C8'): color for the normalized centered root mean square error on the diagram
  r_color (color, default is 'C9'): color of correlation coefficient on the diagram
  no_sigma_axis (logical, default is False): no tick and label for the normalized standard deviation if True
  no_crmse_axis (logical, default is False): no tick and label for the normalized centered root mean square error if True
  r_label (string, default is 'correlation coefficient'): label for the correlation coefficient

  Returns:
  figure axis

  """

  # axes linewidth
  linewidth = mpl.rcParams['axes.linewidth']

  # initialize figure
  ax.axis('square')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlim([0., 1.02 * sigma_lim])
  ax.set_ylim([0., 1.02 * sigma_lim])

  # standard deviation
  for rho in np.arange(.25, sigma_lim + .125, .25):
    theta = np.radians(np.arange(0., 180., 1.))
    x = rho * np.cos(theta);
    y = rho * np.sin(theta);
    if rho == 1.:
      ax.plot(x, y, color = sigma_color, linewidth = linewidth)
    else:
      ax.plot(x, y, '--', color = sigma_color, linewidth = linewidth)
  ax.set_yticks(np.arange(0., sigma_lim + .125, .25))
  ax.spines['left'].set_color(sigma_color)
  if no_sigma_axis:
    yticks = ax.get_yticks()
    yticklabels = []
    for i in range(len(yticks)):
      yticklabels.append('')
    ax.set_yticklabels(yticklabels)
  else:
    ax.set_ylabel('normalized standard deviation')

  # centered root mean square error
  for rho in np.arange(0.25, 1.125, 0.25):
    # full circles
    theta = np.arange(0., 1., .1/180.) * np.pi
    x = rho * np.cos(theta) + 1.
    y = rho * np.sin(theta)
    # remove what is out of sigma_lim circle
    y = y[x <= (1. + sigma_lim ** 2. - rho ** 2) * 0.5]
    x = x[x <= (1. + sigma_lim ** 2. - rho ** 2) * 0.5]
    ax.plot(x, y, '--', color = crmse_color, linewidth = linewidth)
  ax.set_xticks(np.arange(0., 1.625, .25))
  ax.set_xticklabels(['1.00', '0.75', '0.50', '0.25', '0.00', '0.25', '0.50'])
  ax.spines['bottom'].set_color(crmse_color)
  ax.tick_params(axis = 'x', colors = crmse_color)
  ax.xaxis.label.set_color(crmse_color)
  if no_crmse_axis:
    xticks = ax.get_xticks()
    xticklabels = []
    for i in range(len(xticks)):
      xticklabels.append('')
    ax.set_xticklabels(xticklabels)
  else:
    ax.set_xlabel('normalized centered root mean square error')

  # correlation coefficient lines
  for theta in [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]:
    x = 1.02 * sigma_lim * theta
    y = 1.02 * sigma_lim * np.sin(np.arccos(theta))
    ax.plot([0., x], [0., y], ':', color = r_color, linewidth = linewidth)
    ax.text(x / 1.02 * 1.03, y / 1.02 * 1.03, str(theta), \
            verticalalignment = 'center', horizontalalignment = 'left', \
            rotation = np.degrees(np.arccos(theta)), rotation_mode = 'anchor', \
            color = r_color)
  x = 1.15 * sigma_lim * np.cos(np.pi / 4.)
  y = 1.15 * sigma_lim * np.sin(np.pi / 4.)
  ax.text(x, y, r_label, verticalalignment = 'center', \
          horizontalalignment = 'center', rotation = -45., color = r_color)

  # target point
  ax.plot(1., 0., 'o', color = target_color)

  # data points
  rho = statn[:, 4]
  theta = np.arccos(statn[:, 5])
  x = rho * np.cos(theta)
  y = rho * np.sin(theta)
  for i in range(len(x)):
    if rho[i] < 1.5:
      ax.plot(x[i], y[i], marker = prop[i, 0], \
                          markerfacecolor = prop[i, 1], \
                          markeredgecolor = prop[i, 2])

  return ax