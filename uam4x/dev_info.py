import sys
DEV = True
if hasattr(sys, '_MEIPASS'):  # this indicates we are in the pyinstaller exe
    DEV = False