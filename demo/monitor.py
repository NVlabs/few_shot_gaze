# pip install pgi


from gi.repository import Gdk
display = Gdk.Display.get_default()
screen = display.get_default_screen()
mon_h_mm = screen.get_height_mm()
mon_w_mm = screen.get_width_mm()
print(mon_h_mm)
print(mon_w_mm)

mon_h_pixels = screen.get_height()
mon_w_pixels = screen.get_width()
print(mon_h_pixels)
print(mon_w_pixels)