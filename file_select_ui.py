#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 12:54:43 2025

@author: alex
"""

import tkinter as tk
from tkinter import filedialog

def InitializeTkinter():
    root = tk.Tk()
    root.withdraw()  # Hides the main window
    
def GetFile(dialogue_title):
    path = filedialog.askopenfilename(title=dialogue_title)
    if path:  # Check if a file was selected (user didn't cancel)
        print(f"Selected file path: {path}")
        return path
    else:
        return None
    
def GetDirectory(dialogue_title):
    path = filedialog.askdirectory(title=dialogue_title)
    if path:  # Check if a file was selected (user didn't cancel)
        print(f"Selected file path: {path}")
        return path
    else:
        return None