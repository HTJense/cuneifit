"""Cuneifit plotting functions.

These functions are designed to interface with either
`matplotlib <https://matplotlib.org/>`_ or
`getdist <https://getdist.readthedocs.io/>`_.
"""
import numpy as np
import jax.numpy as jnp


def plot_ellipse(ax, mean, cov, nstd=1., **kwargs):
    """Plot a confidence ellipse around a given mean with a given covariance
    matrix.

    Example:

    .. code-block:: python

        chain = pipeline.minimize(theta_start, max_steps=100,
                                  learning_rate=1e-2)
        theta_end = chain[-1]
        F = pipeline.fisher(theta_end)
        cov = jnp.linalg.inv(F)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        i = chain.parameters.index("param_1")
        j = chain.parameters.index("param_2")

        mean = np.array([theta_end[i], theta_end[j])
        subcov = np.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])

        cuneifit.plots.plot_ellipse(ax, mean, subcov, nstd=1., alpha=0.5,
                                    facecolor="blue")
        cuneifit.plots.plot_ellipse(ax, mean, subcov, nstd=2., alpha=0.3,
                                    facecolor="blue")


    Parameters
    ----------

    ax : matplotlib.pyplot.axes
        The subplot axes where this ellipse should be drawn.
    mean : jnp.ndarray
        The (x,y) coordinates around which the ellipse should be centred.
    cov : jnp.ndarray
        The covariance matrix between x and y.
    nstd : float
        The confidence contour to be draw, in Gaussian standard deviations
        (default = 1 sigma).
    **kwargs : dict
        Any parameters given to the ellipse (e.g. color, line widths, etc.).

    Return
    ------
    The ellipse drawn.
    """
    from scipy import stats
    from matplotlib import patches as mpt
    from matplotlib import transforms as mtr

    sx = np.sqrt(cov[0, 0])
    sy = np.sqrt(cov[1, 1])
    # Compute the x/y scale from the correlation coefficient.
    R = cov[0, 1] / (sx * sy)
    ex = np.sqrt(1. + R)
    ey = np.sqrt(1. - R)
    # Recompute the scale by transforming it from a Gaussian to a chi-square
    # distribution.
    nchi2 = np.sqrt(stats.chi2.ppf(2. * stats.norm.cdf(nstd) - 1., df=2))
    ellipse = mpt.Ellipse((0, 0), width=2.*ex, height=2.*ey, **kwargs)
    tf = mtr.Affine2D().rotate_deg(45).scale(nchi2*sx,
                                             nchi2*sy).translate(mean[0],
                                                                 mean[1])
    ellipse.set_transform(tf + ax.transData)
    ax.add_patch(ellipse)
    return ellipse


def plot_fisher_triangle(pipeline, theta, F=None, parameters=None, fig=None,
                         axes=None, line_args=None, contour_args=None,
                         marker_args=None, labels=None):
    """Make a full triangle plot for a pipeline fisher forecast.
    """
    if F is None:
        F = pipeline.fisher(theta)
    if line_args is None:
        line_args = {"lw": 1.2}
    if contour_args is None:
        contour_args = {"lw": 1.2}
    if marker_args is None:
        marker_args = {"lw": 0.8, "ls": "--", "color": "grey"}
    if labels is None:
        labels = {}

    cov = jnp.linalg.inv(F)
    err = jnp.sqrt(jnp.diag(cov))

    import matplotlib.pyplot as plt

    if parameters is None:
        parameters = pipeline.parameters

    if fig is None:
        fig = plt.figure(figsize=(2.5+1.2*len(parameters),
                                  2.0+1.2*len(parameters)))
    if axes is None:
        axes = fig.subplots(len(parameters), len(parameters),
                            gridspec_kw={"hspace": 0.0, "wspace": 0.0})

    for i, p in enumerate(parameters):
        for j, q in enumerate(parameters):
            ax = axes[j, i]
            if i > j:
                fig.delaxes(ax)
                continue
            elif i == j:
                mu = theta[i]
                sig = err[i]
                xmin, xmax = mu - 3.5 * sig, mu + 3.5 * sig
                x = np.linspace(xmin, xmax, 1000)
                y = np.exp(-np.power((x - mu) / sig, 2.) / 2.)
                ax.plot(x, y, **line_args)
                ax.set_ylim(0.0, None)
                ax.axvline(mu, **marker_args)

                for k in range(i):
                    axes[j, k].set_ylim(xmin, xmax)
                for k in range(i+1, len(parameters)):
                    axes[k, i].set_xlim(xmin, xmax)

                ax.set_yticks([])
            else:
                mu = np.array([theta[i], theta[j]])
                subcov = np.array([[cov[i, i], cov[i, j]],
                                   [cov[j, i], cov[j, j]]])
                plot_ellipse(ax, mu, subcov, nstd=1., alpha=0.5,
                             **contour_args)
                plot_ellipse(ax, mu, subcov, nstd=2., alpha=0.3,
                             **contour_args)

                ax.axvline(mu[0], **marker_args)
                ax.axhline(mu[1], **marker_args)

            if i == 0:
                ax.tick_params(axis='y', which='major', labelsize=8)
                ax.set_ylabel(labels.get(q, q), fontsize=12)
            else:
                ax.set_yticks([])

            if j == len(parameters)-1:
                ax.tick_params(axis='x', which='major', labelsize=8)
                ax.set_xlabel(labels.get(p, p), fontsize=12)
            else:
                ax.set_xticks([])

    return fig, axes


def getdist_triangle(plotter, pipeline, theta, parameters=None, line_args=None,
                     **kwargs):
    """Plot the parameter confidence contours directly on a getdist triangle
    plot.

    Example:

    .. code-block:: python

        from getdist import plots

        g = plots.get_subplot_plotter()
        g.triangle_plot([cobaya_chain], parameters, filled=True)

        cuneifit.plots.getdist_triangle(g, pipeline, theta, parameters,
                                        line_args={"lw": 1.2,
                                                   "color": "darkorange"},
                                        contour_args={"fc": "darkorange"})

    Parameters
    ----------

    plotter : getdist.GetDistPlotter
        The GetDist plot factory.
    pipeline : Pipeline
        A pipeline class for which we can compute a fisher matrix.
    theta : jnp.ndarray
        A parameter vector for the pipeline. The fisher matrix will be computed
        from :code:`pipeline.fisher(theta)`.
    parameters : Optional[list]
        A list of parameters to draw. If not given, we will use all parameters
        from the pipeline. This list must however, have the same length as the
        number of parameters drawn in getdist.
    line_args : dict
        A set of arguments passed to the :code:`plot` function for the 1D
        marginals.
    **kwargs : dict
        Keyword arguments that get passed on to
        :code:`getdist.plotter.add_2d_covariance`.
    """
    if parameters is None:
        parameters = pipeline.parameters
    if line_args is None:
        line_args = {}

    F = pipeline.fisher(theta)
    cov = jnp.linalg.inv(F)
    idx = jnp.array([pipeline.parameters.index(p) for p in parameters], int)
    cov = cov[:, idx][idx, :]
    mean = theta[idx]

    for i, p in enumerate(parameters):
        for j, q in enumerate(parameters):
            ax = plotter.subplots[j, i]
            if ax is None:
                continue

            if i == j:
                m = mean[i]
                s = np.sqrt(cov[i, i])
                x = np.linspace(*ax.get_xlim(), 100)
                y = np.exp(-0.5 * ((x - m) / s) ** 2)
                ax.plot(x, y, **line_args)
                continue

            x = mean[i]
            y = mean[j]
            subcov = np.array([[cov[i, i], cov[i, j]],
                               [cov[j, i], cov[j, j]]])
            plotter.add_2d_covariance([x, y], subcov, ax=ax, **kwargs)
