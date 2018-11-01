import curses
import time

# def report_progress(filename, progress):
#     """progress: 0-10"""
#     stdscr.addstr(0, 0, "Moving file: {0}".format(filename))
#     stdscr.addstr(1, 0, "Total progress: [{1:10}] {0}%".format(progress * 10, "#" * progress))
#     stdscr.refresh()

# if __name__ == "__main__":
#     stdscr = curses.initscr()
#     curses.noecho()
#     curses.cbreak()

#     try:
#         for i in range(10):
#             report_progress("file_{0}.txt".format(i), i+1)
#             time.sleep(0.5)
#     finally:
#         curses.echo()
#         curses.nocbreak()
#         curses.endwin()

# begin_x = 20; begin_y = 7
# height = 5; width = 40
# win = curses.newwin(height, width, begin_y, begin_x)
# curses.initscr()

# pad = curses.newpad(4, 4)

frames = ["0000\n0  0\n0 A0\n0000", "0000\n0 A0\n0  0\n0000", "0000\n0A 0\n0  0\n0000", "0000\n0  0\n0A 0\n0000"]
# for i in range(100):
#     box1.addstr('h')# box1.addstr(frames[i%4])
#     bpx1.refresh()
# frames = ['he', 'hi', 'ho', 'no']
screen = curses.initscr()
screen.immedok(True)

try:
    screen.border(0)

    box1 = curses.newwin(20, 20, 5, 5)
    # box1.box()    

    for i in range(10):
        box1.immedok(True)
        box1.addstr(0,0, frames[i%4])
        time.sleep(.5)


    screen.getch()

finally:
    curses.endwin()

# try:
#     screen.border(0)

#     box1 = curses.newwin(20, 20, 5, 5)
#     box1.box()    
#     for i in range(100):
#         box1.addstr('hello')# box1.addstr(frames[i%4])
#         # box1.refresh()

#     screen.refresh()
#     box1.refresh()

#     screen.getch()

# finally:
#     curses.endwin()

# for i in range(100):
#     # These loops fill the pad with letters; addch() is
#     # explained in the next section
#     for y in range(0, 99):
#         for x in range(0, 99):
#             pad.addch(y,x, ord('a') + (i*x*x+y*y) % 26)

#     # Displays a section of the pad in the middle of the screen.
#     # (0,0) : coordinate of upper-left corner of pad area to display.
#     # (5,5) : coordinate of upper-left corner of window area to be filled
#     #         with pad content.
#     # (20, 75) : coordinate of lower-right corner of window area to be
#     #          : filled with pad content.
#     pad.refresh( 0,0, 5,5, 20,75)

