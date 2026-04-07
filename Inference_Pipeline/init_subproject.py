# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:45:25 2026

@author: cns-th-lab
"""

import sys
from os import path

this_dir = path.dirname(path.abspath(__file__))
parent_dir = path.dirname(this_dir)
par_parent_dir = path.dirname(parent_dir)

print(this_dir)
print(parent_dir)
print(par_parent_dir)
x = input("Is are these paths ok: ")
if x != "y":
    quit()

if not this_dir in sys.path:
    sys.path.append(this_dir)

if not parent_dir in sys.path:
    sys.path.append(parent_dir)

if not par_parent_dir in sys.path:
    sys.path.append(par_parent_dir)