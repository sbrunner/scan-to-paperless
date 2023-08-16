"""Utility function related to Jupyter notebook."""


def is_ipython() -> bool:
    """Return True if running in IPython (Jupyter)."""

    try:
        __IPYTHON__  # type: ignore[name-defined] # pylint: disable=pointless-statement
        return True
    except NameError:
        return False
