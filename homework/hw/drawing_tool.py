import gizeh
import moviepy.editor as mpy
import numpy as np

W, H = 256, 256  # width, height, in pixels
duration = 10  # duration of the clip, in seconds
radius = 12


def make_frame(t):
    surface = gizeh.Surface(W, H, bg_color=(.9, .9, .9))
    x = 40 + t * 180 / duration
    left_wheel = gizeh.circle(r=radius, xy=(x - 15, H / 2 + 5), fill=(0, 0, 0))
    right_wheel = gizeh.circle(r=radius, xy=(x + 15, H / 2 + 5), fill=(0, 0, 0))
    roof = gizeh.rectangle(lx=40, ly=35, xy=(x, H / 2 - 11), fill=(0, 0, 0))
    body = gizeh.rectangle(lx=70, ly=20, xy=(x, H / 2 - 5), fill=(0, 0, 0))
    rect = gizeh.rectangle(lx=2, ly=H, xy=(int(0.5 * W), H / 2), fill=(1, 0, 0))
    roof.draw(surface)
    body.draw(surface)
    left_wheel.draw(surface)
    right_wheel.draw(surface)
    rect.draw(surface)

    return surface.get_npimage()


clip = mpy.VideoClip(make_frame, duration=duration)
clip.write_videofile("car.mp4", fps=24)
