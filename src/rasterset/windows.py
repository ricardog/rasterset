
def window_shape(win):
    return (int(win.height), int(win.width))


def round_window(win):
    return win.round_offsets('floor').round_lengths('ceil')
