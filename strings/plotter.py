import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl


def plot_contours(x, y,xlabel='', ylabel='',
                  input_x=None, input_y=None,
                  input_color=['red','blue','magenta'], fontsize=22,
                  title='', plot_samples=False, samples_color='gray',
                  contour_lw=1, savefile=None, plot_contours=True,
                  show_legend=False,fill_contours=True,
                  xmin=None,xmax=None, ymin=None, ymax=None,
                  input_label=None):
    """
    ...
    """
    np.atleast_1d(input_x)
    np.atleast_1d(input_y)
    
    
    n = 100

    points = np.array([x,y])
    posterior = kde(points)

    if xmin is None:
        xmin=x.min()

    if xmax is None:
        xmax=x.max()

    if ymin is None:
        ymin=y.min()

    if ymax is None:
        ymax=y.max()


    step_x = ( xmax - xmin ) / n
    step_y = ( ymax - ymin ) / n
    grid_pars = np.mgrid[0:n,0:n]
    par_x = grid_pars[0]*step_x + xmin
    par_y = grid_pars[1]*step_y + ymin
    grid_posterior = grid_pars[0]*0.


    for i in range(n):
        for j in range(n):
            grid_posterior[i][j] = posterior([par_x[i][j],par_y[i][j]])

    ix,iy = np.unravel_index(grid_posterior.argmax(), grid_posterior.shape)
    gridmaxx = par_x[ix,iy]
    gridmaxy = par_y[ix,iy]
    #print gridmaxx, gridmaxy

    pl.figure()
    ax = pl.gca()
    #xlabel = ax.get_xlabel()
    #ylabel = ax.get_ylabel()
    pl.title(title, fontsize=fontsize)
    fig = pl.gcf()
    xlabel = ax.set_xlabel(xlabel,fontsize=fontsize)
    ylabel = ax.set_ylabel(ylabel,fontsize=fontsize)
    #pl.xlabel(xlabel,fontsize=fontsize)
    #pl.ylabel(ylabel,fontsize=fontsize)
    if plot_samples:
        pl.plot(x,y,'o',ms=1, mfc=samplescolor, mec=samplescolor)
    for i,(inx, iny) in enumerate(zip(input_x,input_y)):
        if input_label is not None:
            label = input_label[i]
        else:
            label = 'string'
        pl.plot(inx,iny,'o',mew=0.1,ms=5,color=input_color[i],label=label,zorder=4)
    pl.plot(gridmaxx,gridmaxy,'x',mew=3,ms=5,color='k')
    if plot_contours:
        percentage_integral = np.array([0.95,0.68,0.])
        contours = 0.* percentage_integral		
        num_epsilon_steps = 1000.
        epsilon = grid_posterior.max()/num_epsilon_steps
        epsilon_marks = np.arange(num_epsilon_steps + 1)
        posterior_marks = grid_posterior.max() - epsilon_marks * epsilon
        posterior_norm = grid_posterior.sum()
        for j in np.arange(len(percentage_integral)):
            for i in epsilon_marks:
                posterior_integral = grid_posterior[np.where(grid_posterior>posterior_marks[i])].sum()/posterior_norm
                if posterior_integral > percentage_integral[j]:
                    break
            contours[j] = posterior_marks[i]
        contours[-1]=grid_posterior.max()
        pl.contour(par_x, par_y, grid_posterior, contours, linewidths=contour_lw,colors='k',zorder=3)
        if fill_contours:
            pl.contourf(par_x, par_y, grid_posterior, contours,colors=contour_colors,alpha=alpha,zorder=2)
        
    pl.xlim(xmin=xmin,xmax=xmax)
    pl.ylim(ymin=ymin,ymax=ymax)
    if show_legend: 
        pl.legend(prop={'size':20},numpoints=1,loc='upper left')
    if savefile is None:
        return par_x, par_y, grid_posterior, contours
    else:
        pl.savefig(savefile, bbox_extra_artists=[xlabel, ylabel], bbox_inches='tight')
        #pl.savefig(savefile)
    

