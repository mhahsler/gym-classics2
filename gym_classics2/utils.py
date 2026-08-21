import numpy as np


def get_rng(rng=None):
    """Return *rng* as a NumPy random generator.

    ``rng`` may be a :class:`numpy.random.Generator`, an integer seed, or
    ``None``. Passing a generator lets callers share one reproducible random
    stream across an algorithm and all of its helpers.
    """
    return np.random.default_rng(rng)

def clip(x, low, high):
    """A scalar version of numpy.clip. Much faster because it avoids memory allocation."""
    return min(max(x, low), high)


# np.argmax does not break ties randomly
def random_argmax(x, axis=None, rng=None):
    """
    Argmax that breaks ties randomly. If axis is None, returns a single index.
    If axis is specified, returns an array of indices along that axis. ``rng``
    may be a NumPy generator or an integer seed.
    """
    rng = get_rng(rng)
    if axis is None:
        return rng.choice(np.where(x == np.max(x))[0])
    else:
        return np.apply_along_axis(lambda values: random_argmax(values, rng=rng), axis, x)
