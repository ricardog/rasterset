def shape(win):
    return (int(win.height), int(win.width))


def round(win):
    return win.round_offsets('floor').round_lengths('ceil')
