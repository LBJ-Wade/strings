import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
from scipy.interpolate import LSQUnivariateSpline as interpolate

from .statistic import PDF, PowerSpectrum, Moments

def plot_powerspectrum(cmb_mapset, string_mapset, which='gradgrad', residuals=True,
                beatdown=1., string_band=False, xmin=1000, xmax=5000, 
                ps_kwargs=None):
    
    cmb_maps = getattr(cmb_mapset, which)
    stringy_maps = getattr(string_mapset, which)

    if ps_kwargs is None:
        ps_kwargs = {}        
    power = PowerSpectrum(**ps_kwargs)
    
    if 'power' not in cmb_maps.statistics:
        cmb_maps.apply_statistic(power)
    if 'power' not in stringy_maps.statistics:
        stringy_maps.apply_statistic(power)

    nbins = len(cmb_maps.statistics['power'][0][0])
    Ncmb = cmb_maps.N
    Nstring = stringy_maps.N
    
    h_stringy = np.zeros((Nstring,nbins))
    h_cmb = np.zeros((Ncmb,nbins))

    for i in np.arange(Ncmb):
        xs = cmb_maps.statistics['power'][i][0]
        h_cmb[i,:] = cmb_maps.statistics['power'][i][1]

    h_mean = h_cmb.mean(axis=0)
    h_std = h_cmb.std(axis=0)

    for i in np.arange(Nstring):
        xs = stringy_maps.statistics['power'][i][0]
        h_stringy[i,:] = stringy_maps.statistics['power'][i][1]

    h_mean_string = h_stringy.mean(axis=0)
    h_std_string = h_stringy.std(axis=0)

    fig1 = plt.figure(figsize=(7,7))
    
    plt.plot(xs, h_mean, 'k', lw=1)
    plt.fill_between(xs, (h_mean - h_std), (h_mean + h_std), color='k', alpha=0.3)

    plt.plot(xs, h_mean_string, 'r', lw=1)
    plt.xlim(xmin=xmin, xmax=xmax)
    

    # PLOT RESIDUALS:
    fig2 = plt.figure(figsize=(7,7))

    plt.plot(xs, h_mean_string - h_mean, color='r')
    plt.xlim(xmin=xmin, xmax=xmax)
    
    plt.fill_between(xs, -h_std/beatdown, h_std/beatdown, color='k', alpha=0.3)
    plt.fill_between(xs, h_mean_string - h_mean - h_std_string/beatdown,
                     h_mean_string - h_mean + h_std_string/beatdown, color='r', alpha=0.3)

    plt.title('Gmu = {:.1e}'.format(stringy_maps.kwargs['Gmu']))

    return fig1, fig2

pdf_defaults={
    'binmin':{'gradgrad':0,
              'gradgrad_rotstick':3e8,
              'map':-500},
    'binmax':{'gradgrad':1e10,
              'gradgrad_rotstick':7e8,
              'map':500}}

pdf_xmax_defaults = {'gradgrad':0.2e10,
                     'gradgrad_rotstick':0.7e9,
                     'map':500}
    
def plot_pdf(cmb_mapset, string_mapset, which='gradgrad', residuals=True,
             beatdown=1., string_band=False, xmax=None,
             pdf_kwargs=None, fit_chisq=False,
             polyfit_order=30, ax=None,
             resid_only=False,
             title=None,
             pdf_only=False):
    
    cmb_maps = getattr(cmb_mapset, which)
    stringy_maps = getattr(string_mapset, which)

    if pdf_kwargs is None:
        pdf_kwargs = {}

    for prop in ['binmin', 'binmax']:
        if prop not in pdf_kwargs:
            pdf_kwargs[prop] = pdf_defaults[prop][which]

    if xmax is None:
        xmax = pdf_xmax_defaults[which]
            
            
    pdf = PDF(**pdf_kwargs)
    
    if 'pdf' not in cmb_maps.statistics:
        cmb_maps.apply_statistic(pdf)
    if 'pdf' not in stringy_maps.statistics:
        stringy_maps.apply_statistic(pdf)

    nbins = len(cmb_maps.statistics['pdf'][0][0])
    Ncmb = cmb_maps.N
    Nstring = stringy_maps.N
    
    h_stringy = np.zeros((Nstring,nbins))
    h_cmb = np.zeros((Ncmb,nbins))

    for i in np.arange(Ncmb):
        xs = cmb_maps.statistics['pdf'][i][0]
        h_cmb[i,:] = cmb_maps.statistics['pdf'][i][1]

    h_mean = h_cmb.mean(axis=0)
    #c = np.polyfit(xs, h_mean, polyfit_order)
    #h_mean = np.polyval(c, xs)
    h_std = h_cmb.std(axis=0)

    for i in np.arange(Nstring):
        xs = stringy_maps.statistics['pdf'][i][0]
        h_stringy[i,:] = stringy_maps.statistics['pdf'][i][1]

    h_mean_string = h_stringy.mean(axis=0)
    #c = np.polyfit(xs, h_mean_string, polyfit_order)
    #h_mean_string = np.polyval(c, xs)
    h_std_string = h_stringy.std(axis=0)

    ax_is_none = ax is None
    if not resid_only:
        if ax_is_none:
            fig1 = plt.figure(figsize=(7,7))
            ax = plt.gca()
        else:
            fig1 = plt.gcf()

        ax.plot(xs, h_mean, 'k', lw=1)
        ax.fill_between(xs, (h_mean - h_std), (h_mean + h_std), color='k', alpha=0.3)

        ax.plot(xs, h_mean_string, 'r', lw=1)
        ax.set_xlim(xmax=xmax)

        if title is None:
            title = 'Gmu = {:.1e}'.format(stringy_maps.kwargs['Gmu'])
        ax.set_title(title)

        plt.draw()

    # PLOT RESIDUALS:
    if not pdf_only:
        if ax_is_none:
            fig2 = plt.figure(figsize=(7,7))
            ax = plt.gca()
        else:
            fig2 = plt.gcf()
            

        ax.plot(xs, h_mean_string - h_mean, color='r')
        ax.set_xlim(xmax=xmax)

        if beatdown > 1:
            c = np.polyfit(xs, h_mean, polyfit_order)
            h_mean = np.polyval(c, xs)            
            c = np.polyfit(xs, h_mean_string, polyfit_order)
            h_mean_string = np.polyval(c, xs)
            
            
        ax.fill_between(xs, -h_std/beatdown, h_std/beatdown, color='k', alpha=0.3)
        ax.fill_between(xs, h_mean_string - h_mean - h_std_string/beatdown,
                            h_mean_string - h_mean + h_std_string/beatdown, color='r', alpha=0.3)
        if beatdown > 1:
            ax.plot(xs, -h_std, color='k', alpha=0.5)
            ax.plot(xs, h_std, color='k', alpha=0.5)
            ax.plot(xs, h_mean_string - h_mean - h_std_string, alpha=0.5, color='r')
            ax.plot(xs, h_mean_string - h_mean + h_std_string, alpha=0.5, color='r')
            
        if title is None:
            title = 'Gmu = {:.1e}'.format(stringy_maps.kwargs['Gmu'])
        ax.set_title(title)
        plt.draw()
        
    if resid_only:
        return fig2
    elif pdf_only:
        return fig1
    else:
        return fig1, fig2
        
    
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
    

