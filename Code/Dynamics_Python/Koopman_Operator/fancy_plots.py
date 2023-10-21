import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
plt.rc('text', usetex = True)
def fancy_plots_2():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2


def fancy_plots_4():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    return fig, ax1, ax2, ax3, ax4

def fancy_plots_3():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    return fig, ax1, ax2, ax3

def fancy_plots_1():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    return fig, ax1


  
def plot_states_angles(fig11, ax11, ax21, ax31, x, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$\phi$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[rad]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'${\theta}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[rad]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'${\psi}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    
def plot_states_angles_estimation(fig11, ax11, ax21, ax31, x, x_e, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    state_3_e_k, = ax31.plot(t[0:t.shape[0]], x_e[2, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="--")

    ax11.set_ylabel(r"$[rad]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$\phi$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[rad]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'${\theta}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[rad]$", rotation='vertical')
    ax31.legend([state_3_e, state_3_e_k],
            [r'${\psi}$', r'${\hat{\psi}}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    
def plot_states_position(fig11, ax11, ax21, ax31, x, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[m]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$x$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[m]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$y$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[m]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$z$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    
def plot_states_velocity_lineal(fig11, ax11, ax21, ax31, x, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$\mu_l$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$\mu_m$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$\mu_n$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    
def plot_states_velocity_lineal_estimation(fig11, ax11, ax21, ax31, x, x_e, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")

    state_1_e_k, = ax11.plot(t[0:t.shape[0]], x_e[7, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="--")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_e_k],
            [ r'$\mu_l$', r'$\hat{\mu_l}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$\mu_m$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$\mu_n$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    
def plot_states_velocity_angular(fig11, ax11, ax21, ax31, x, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$q$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$p$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax31.legend([state_3_e],
            [r'$r$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
    
def plot_states_velocity_angular_estimation(fig11, ax11, ax21, ax31, x, x_e, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], x[1, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[0, 0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="-")


    state_3_e, = ax31.plot(t[0:t.shape[0]], x[2, 0:t.shape[0]],
                    color='#3F8BB4', lw=1.0, ls="-")
                    
    state_3_e_k, = ax31.plot(t[0:t.shape[0]], x_e[5, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="-")

    ax11.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$q$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax21.legend([state_2_e],
            [r'$p$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xticklabels([])
    
    ax31.set_ylabel(r"$[rad/s]$", rotation='vertical')
    ax31.legend([state_3_e,state_3_e_k],
            [r'$r$', r'$\hat{r}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax31.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax31.axis([t[0], t[-1], x[2,:].min()-0.1, x[2,:].max()+0.1])
    ax31.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)


    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")

def plot_control_states(fig11, ax11, ax21, x, xp, u, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], xp[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="--")
    
    state_1_c, = ax11.plot(t[0:t.shape[0]], u[0, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[0,0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="--")
    
    state_2_c, = ax21.plot(t[0:t.shape[0]], u[1, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="-")



    ax11.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_c],
            [ r'$\mu_l$', r'$\mu_{ld}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[rad]$", rotation='vertical')
    ax21.legend([state_2_e, state_2_c],
            [r'$\alpha$', r'$\alpha_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
  
    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")

def plot_control_states_estimation(fig11, ax11, ax21, x, xp, x_e, t, name):
    ax11.set_xlim((t[0], t[-1]))
    ax21.set_xlim((t[0], t[-1]))
    ax11.set_xticklabels([])


    state_1_e, = ax11.plot(t[0:t.shape[0]], xp[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="--")
    
    state_1_c, = ax11.plot(t[0:t.shape[0]], x_e[7, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="-")


    state_2_e, = ax21.plot(t[0:t.shape[0]], x[10,0:t.shape[0]],
                    color='#3FB454', lw=1.0, ls="--")
    
    state_2_c, = ax21.plot(t[0:t.shape[0]], x_e[6, 0:t.shape[0]],
                    color='#3D4D55', lw=1.0, ls="-")



    ax11.set_ylabel(r"$[m/s]$", rotation='vertical')
    ax11.legend([state_1_e, state_1_c],
            [ r'$\mu_l$', r'$\hat{\mu_{l}}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    ## Figure 2
    #fig2, ax2 = fancy_plots()
    ax21.set_ylabel(r"$[rad]$", rotation='vertical')
    ax21.legend([state_2_e, state_2_c],
            [r'$\alpha$', r'$\hat{\alpha}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax21.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax21.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
  
    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")

def plot_error_estimation(fig11, ax11, x_e, t, name):
    ax11.set_xlim((t[0], t[-1]))


    state_1_e, = ax11.plot(t[0:t.shape[0]], x_e[0, 0:t.shape[0]],
                    color='#C43C29', lw=1.0, ls="-")
    
    ax11.set_ylabel(r"$[error]$", rotation='vertical')
    ax11.legend([state_1_e],
            [ r'$error$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax11.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)

    fig11.savefig(name + ".pdf")
    fig11.savefig(name + ".png")
