import numpy as np
import matplotlib.pyplot as plt


################################################################################

def get_stat(x0, x):

  """
  calculate different statistics (mean, bias, root mean square error, centered root mean square error, standard deviation, correlation coefficient) from x (test dataset), with regards to x0 (reference dataset)

  input:
  x0: array of shape (n)
  x: array of shape (m, n)

  output:
  stat0: reference statistic array of shape (6)
  stat: statistic array of shape (m, 6)
  """

  # number of data
  n = x0.shape[0]

  # mean of the reference
  s0 = np.mean(x0)

  # standard deviation of the reference
  sigma0 = np.sqrt(np.sum((x0 - s0) ** 2.) / n)

  # all statistics
  stat0 = np.array([s0, 0., 0., 0., sigma0, 1.])

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

def get_statn(x0, x):

  """
  calculate different statistics (mean, bias, root mean square error, normalized centered root mean square error, normalized standard deviation, correlation coefficient) from x (test dataset), with regards to x0 (reference dataset)

  input:
  x0: array of shape (n)
  x: array of shape (m, n)

  output:
  statn: normalized statistic array of shape (m, 6)
  """

  # all statistics
  [stat0, stat] = get_stat(x0, x)

  # normalized statistics
  statn = stat
  statn[:, 3] /= stat0[4]
  statn[:, 4] /= stat0[4]

  return statn


################################################################################

def diagn(ax, statn, prop, legend = None, title = '', sigma_lim = 1.5, \
          sigma_color = 'k', crmse_color = 'C0', r_color = 'C2', \
          no_sigma_axis = False, no_crmse_axis = False, \
          legend_horizontal_anchor = 1):

  """
  create a normalized taylor diagram

  input:
  ax: figure axes
  statn: normalized statistic array of shape (m, 6) from get_statn()
  prop: object array of shape (m, 3), with prop[i, :] = [marker, markerfacecolor, markeredgecolor] of data i
  legend: object array of shape (n, 3), with legend[i] = [marker, markerfacecolor, markeredgecolor, legend] of data i in the legend (for sake of flexibility, data in legend can be different from data in diagram)

  # input (not mandatory):
  title: title on top of the diagram
  sigma_lim: maximum value of normalized standard deviation (and hence normalized centered root mean square error) displayed on the diagram
  sigma_color: color for normalized standard deviation on the diagram
  crmse_color: color for normalized centered root mean square error on the diagram
  r_color: color of correlation coefficient on the diagram
  no_sigma_axis: if True, remove ticks and label of normalized standard deviation axis
  no_crmse_axis: if True, remove ticks and label of normalized centered root mean square error axis
  legend_horizontal_anchor: increase default value (1) to shift legend to the right

  output:
  ax: figure axes
  """

  # initialize figure
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
      plt.plot(x, y, color = sigma_color, linewidth = 1.)
    else:
      plt.plot(x, y, '--', color = sigma_color, linewidth = .5)
  plt.yticks(np.arange(0., sigma_lim + .125, .25))
  ax.spines['left'].set_color(sigma_color)
  if no_sigma_axis:
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
    plt.plot(x, y, '--', color = crmse_color, linewidth = .5)
  plt.xticks(np.arange(0., 1.125, .25), \
             ['1.00', '0.75', '0.50', '0.25', '0.00'])
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
    plt.xlabel('normalized centered root mean square error')

  # correlation coefficient lines
  for theta in [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]:
    x = sigma_lim * theta
    y = sigma_lim * np.sin(np.arccos(theta))
    plt.plot([0., x], [0., y], ':', color = r_color, linewidth = .5)
    if theta < .95:
      coef = 1.04
    else:
      coef = 1.05
    ax.text(coef * x, coef * y, str(theta), verticalalignment = 'center', \
            horizontalalignment = 'center', \
            rotation = np.degrees(np.arccos(theta)), color = r_color)
  x = 1.1 * sigma_lim * np.cos(np.pi / 4.)
  y = 1.1 * sigma_lim * np.sin(np.pi / 4.)
  ax.text(x, y, 'correlation coefficient', verticalalignment = 'center', \
          horizontalalignment = 'center', \
          rotation = -45., color = r_color)

  # target point
  plt.plot(1., 0., 'ko')

  # data points
  rho = statn[:, 4]
  theta = np.arccos(statn[:, 5])
  x = rho * np.cos(theta)
  y = rho * np.sin(theta)
  for i in range(len(x)):
    if rho[i] < 1.5:
      plt.plot(x[i], y[i], marker = prop[i, 0], \
                           markerfacecolor = prop[i, 1], \
                           markeredgecolor = prop[i, 2])

  # legend
  if legend is not None:
    for i in range(legend.shape[0]):
      plt.plot(-1., -1., '.', marker = legend[i, 0], \
                              markerfacecolor = legend[i, 1], \
                              markeredgecolor = legend[i, 2], \
                              label = legend[i][3])
    if legend.shape[0] > 0:
      plt.legend(bbox_to_anchor = (legend_horizontal_anchor, 1), \
                 loc = 'upper left')

  # title
  if title is not '':
    plt.text(.75, 1.65, title, horizontalalignment = 'center')

  return ax