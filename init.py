import sys
from os import path

# temporary to turn this behavior on by default before it becomes permanent
import pandas as pd
pd.options.mode.copy_on_write = True

this_dir = path.dirname(path.abspath(__file__))
parent_dir = path.dirname(this_dir)

if not this_dir in sys.path:
    sys.path.append(this_dir)

if not parent_dir in sys.path:
    sys.path.append(parent_dir)