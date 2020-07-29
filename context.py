import os
import sys

# Add the package root directory to PYTHONPATH
# and import a copy of our module
PKG_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PKG_ROOT)

# Set directories for data and matplotlibrc
RE_DATA = os.path.join(PKG_ROOT, 'data', 'Re_stress.dat')
