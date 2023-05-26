"""Initialize the trnpy package and define version."""
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'trnpy'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    try:
        from setuptools_scm import get_version
        __version__ = get_version(version_scheme='post-release',
                                  root='..', relative_to=__file__)
    except Exception:
        __version__ = 'unknown'
