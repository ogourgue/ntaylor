import numpy as np
import matplotlib.pyplot as plt


def get_stat(x0, x):

  # number of data
  n = x0.shape[0]

  # mean of the reference
  s0 = np.mean(x0)

  # standard deviation of the reference
  sigma0 = np.sqrt(np.sum((x0 - s0) ** 2.) / n)

  # all statistics
  stat0 = [s0, 0., 0., 0., sigma0, 1.]

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


def get_statn(x0, x):

  # all statistics
  [stat0, stat] = get_stat(x0, x)

  # normalized statistics
  statn = stat
  statn[:, 3] /= stat0[4]
  statn[:, 4] /= stat0[4]

  return statn


def diagn(statn, prop, legend, title = '', no_crmse_axis = False, \
          no_std_axis = False):

  # maximum standard deviation to plot
  sigma_lim = 1.5

  # default colors
  sigma_col = 'k'
  crmse_col = 'C0'
  r_col = 'C2'

  # create figure
  fig = plt.figure()
  ax = plt.gca()
  plt.axis('square')
  plt.xlim([0., 1.01 * sigma_lim])
  plt.ylim([0., 1.01 * sigma_lim])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  # standard deviation
  for rho in np.arange(.25, sigma_lim + .125, .25):
    theta = np.radians(np.arange(0., 180., 1.))
    x = rho * np.cos(theta);
    y = rho * np.sin(theta);
    if rho == 1.:
      plt.plot(x, y, color = sigma_col, linewidth = 1.)
    else:
      plt.plot(x, y, '--', color = sigma_col, linewidth = .5)
  plt.yticks(np.arange(0., sigma_lim + .125, .25))
  ax.spines['left'].set_color(sigma_col)
  if no_std_axis:
    yticks = ax.get_yticks()
    yticklabels = []
    for i in range(len(yticks)):
      yticklabels.append('')
    ax.set_yticklabels(yticklabels)
  else:
    plt.ylabel('normalized standard deviation')

  # centered root mean square error
  for rho in np.arange(0.25, 1.125, 0.25):
    # full circles
    theta = np.arange(0., 1., .1/180.) * np.pi
    x = rho * np.cos(theta) + 1.
    y = rho * np.sin(theta)
    # remove what is out of sigma_lim circle
    y = y[x <= (1. + sigma_lim ** 2. - rho ** 2) * 0.5]
    x = x[x <= (1. + sigma_lim ** 2. - rho ** 2) * 0.5]
    plt.plot(x, y, '--', color = crmse_col, linewidth = .5)
  plt.xticks(np.arange(0., 1.125, .25), \
             ['1.00', '0.75', '0.50', '0.25', '0.00'])
  ax.spines['bottom'].set_color(crmse_col)
  ax.tick_params(axis = 'x', colors = crmse_col)
  ax.xaxis.label.set_color(crmse_col)
  if no_crmse_axis:
    xticks = ax.get_xticks()
    xticklabels = []
    for i in range(len(xticks)):
      xticklabels.append('')
    ax.set_xticklabels(xticklabels)
  else:
    plt.xlabel('normalized centered root mean square error')

  # correlation coefficient lines
  for theta in [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]:
    x = sigma_lim * theta
    y = sigma_lim * np.sin(np.arccos(theta))
    plt.plot([0., x], [0., y], ':', color = r_col, linewidth = .5)
    if theta < .95:
      coef = 1.04
    else:
      coef = 1.05
    ax.text(coef * x, coef * y, str(theta), verticalalignment = 'center', \
            horizontalalignment = 'center', \
            rotation = np.degrees(np.arccos(theta)), color = r_col)
  x = 1.1 * sigma_lim * np.cos(np.pi / 4.)
  y = 1.1 * sigma_lim * np.sin(np.pi / 4.)
  ax.text(x, y, 'correlation coefficient', verticalalignment = 'center', \
          horizontalalignment = 'center', \
          rotation = -45., color = r_col)

  # target point
  plt.plot(1., 0., 'ko')

  # for each data type
  for i in range(len(statn)):
    rho = statn[i][:, 4]
    theta = np.arccos(statn[i][:, 5])
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    for j in range(len(x)):
      if rho[j] < 1.5:
        plt.plot(x[j], y[j], marker = prop[i][j][0], \
                 markerfacecolor = prop[i][j][1], markeredgecolor = 'k')

  # legend
  for i in range(len(legend)):
    plt.plot(-1., -1., '.', marker = legend[i][0], markerfacecolor = legend[i][1], \
             markeredgecolor = legend[i][2], label = legend[i][3])
  if len(legend) > 0:
    if len(legend) < 12:
      plt.legend(bbox_to_anchor=(1.01, 1.), loc='upper left')
    else:
      plt.legend(bbox_to_anchor=(1.03, 1.), loc='upper left')

  # title
  if title is not '':
    plt.text(.75, 1.65, title, horizontalalignment = 'center')

  return fig