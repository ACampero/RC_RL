import curses

def main(win):
    win.nodelay(True)
    key=""
    win.clear()                
    win.addstr("Detected key:")
    while 1:          
        try:                 
           key = win.getkey()         
           win.clear()                
           win.addstr("Detected key:")
           win.addstr(str(key)) 
           if key == os.linesep:
              break           
        except Exception as e:
           # No input   
           pass         

def checkForKey(win):
	while True:
		try:
			win.nodelay(True)
			win.clear()
			key=win.getkey()
			win.clear()                
			if key=='a':
				win.addstr('hello')
			if key=='q':
				win.addstr('nope')
		except Exception as e:
			pass
curses.wrapper(checkForKey)