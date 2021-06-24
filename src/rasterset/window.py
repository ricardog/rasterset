from rasterio.windows import from_bounds                    # noqa F401


def shape(win):
    return (int(win.height), int(win.width))


def round(win):
    return win.round_offsets('floor').round_lengths('ceil')

def inset(outter, inner):
    assert inner.col_off < outter.width
    assert inner.row_off < outter.height
    assert inner.col_off + inner.width <= outter.width
    assert inner.row_off + inner.height <= outter.height

    return Window(col_off=(outter.col_off + inner.col_off),
                  row_off=(outter.row_off + inner.row_off),
                  width=inner.width, height=inner.height)
