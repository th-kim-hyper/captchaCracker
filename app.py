import sys
# from tkinter import *

meipass = getattr(sys, '_MEIPASS', None)
if meipass:
	print(meipass)
else:
	print("Attribute _MEIPASS not found in sys module")
	
# root = Tk()
# root.mainloop()
