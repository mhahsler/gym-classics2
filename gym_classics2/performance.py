import matplotlib.pyplot as plt
import numpy as np

def simple_moving_average(data, window_size = 100):
    """Return a centered simple moving average padded with ``NaN`` values.

    Args:
        data: One-dimensional numeric sequence.
        window_size: Number of observations in the averaging window.

    Returns:
        NumPy array with the same length as ``data``.
    """
    weights = np.ones(window_size) / window_size
    sma = np.convolve(data, weights, mode='valid')
    sma = np.concatenate((np.full((window_size)//2, np.nan),sma,np.full(len(data)-len(sma)-(window_size)//2, np.nan)))  # Pad the beginning with NaN for alignment
    return sma


def cum_avg(data):
    """Return the cumulative average at every position in a numeric sequence.

    The value at position ``i`` is the mean of ``data[:i + 1]``.

    Args:
        data: One-dimensional numeric sequence.

    Returns:
        NumPy array containing the cumulative mean at each position. The array
        has the same length as ``data``.
    """
    return np.cumsum(data) / np.arange(1, len(data) + 1)

def plot_returns(returns, y_label = "Episode Return", title = "", window_size = 100,
                y_range = None, log_scale = False):
    """Plot episode returns, their moving average, and cumulative average.

    Displays the episode returns and both averages on one Matplotlib plot.
    The moving average is centered and padded with NaNs at its ends.

    Args:
        returns: One-dimensional sequence of returns, usually one value per
            episode.
        y_label: Label for the y-axis.
        title: Plot title.
        window_size: Number of episodes in the moving-average window.
        y_range: Optional ``(minimum, maximum)`` y-axis limits.
        log_scale: If ``True``, use a logarithmic y-axis. Values plotted on
            that axis must be positive.

    Returns:
        None. Displays the plot using Matplotlib.
    """
    x = range(len(returns))
    plt.plot(x, returns, label="Episode")
    plt.plot(x, simple_moving_average(returns, window_size), label="Moving Average (100)")
    plt.plot(x, cum_avg(returns), label="Cumulative Average")

    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    if y_range is not None:
        plt.ylim(y_range)
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.show()
    
def plot_episode_lengths(ep_lens, y_label = "Episode Length", title = "", window_size = 100,
                 y_range = None, log_scale = False):
    """Plot episode lengths, their moving average, and cumulative average.

    Displays the episode lengths and both averages on one Matplotlib plot.
    The moving average is centered and padded with NaNs at its ends.

    Args:
        ep_lens: One-dimensional sequence of episode lengths, usually one
            value per episode.
        y_label: Label for the y-axis.
        title: Plot title.
        window_size: Number of episodes in the moving-average window.
        y_range: Optional ``(minimum, maximum)`` y-axis limits.
        log_scale: If ``True``, use a logarithmic y-axis. Values plotted on
            that axis must be positive.

    Returns:
        None. Displays the plot using Matplotlib.
    """
    x = range(len(ep_lens))
    plt.plot(x, ep_lens, label="Episode Length")
    plt.plot(x, simple_moving_average(ep_lens, window_size), label="Moving Average (100)")
    plt.plot(x, cum_avg(ep_lens), label="Cumulative Average")

    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    if y_range is not None:
        plt.ylim(y_range)
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.show()
