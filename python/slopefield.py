"""Slope field generator.

Draws the slope field for a first-order differential equation dy/dx = f(x, y),
and traces solution curves through any starting points you give it.

Written with the tools from Data 8 / COMSC 028: functions, numpy arrays,
loops, Tables, and matplotlib. No symbolic math and no solver library.

Quick start
-----------
    from slopefield import slope_field

    def f(x, y):
        return x + y

    slope_field(f, starts=[(0, 0), (0, 1), (0, -2)])

Why the code looks the way it does, point by point, is in README.md.
"""

import numpy as np
from datascience import Table
import matplotlib.pyplot as plots


# ----------------------------------------------------------------------
# 1. Evaluating the slope without crashing
# ----------------------------------------------------------------------

def safe_slope(f, x, y):
    """Return f(x, y) as a float.

    A slope rule can fail at some points: x/y divides by zero on the x-axis,
    ln(y) is undefined for y <= 0. Instead of crashing the whole plot, this
    returns inf for a vertical slope and nan for a point with no slope at all.
    """
    try:
        with np.errstate(all='ignore'):
            m = f(float(x), float(y))
    except ZeroDivisionError:
        return np.inf
    except (ValueError, OverflowError):
        return np.nan
    try:
        m = float(m)
    except TypeError:
        # A complex result, e.g. sqrt of a negative under some libraries.
        return np.nan
    return m


def slope_column(f, grid):
    """Return the slope at every row of a grid Table."""
    with np.errstate(all='ignore'):
        return grid.apply(lambda x, y: safe_slope(f, x, y), 'x', 'y')


# ----------------------------------------------------------------------
# 2. The grid of points
# ----------------------------------------------------------------------

def make_grid(f, xmin=-5, xmax=5, ymin=-5, ymax=5, nx=21, ny=21):
    """Return a Table with one row per grid point and columns x, y, slope.

    The grid is nx points across by ny points up, spread over whatever window
    you ask for. Counting points instead of fixing a step size is what keeps
    the field readable in a tall or wide window: a step of 0.5 that looks right
    from -5 to 5 would stack 200 rows on top of each other from -50 to 50.

    The Table is the useful part: printing it shows the slope at every point,
    which is the arithmetic behind the picture.
    """
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    grid_x = np.array([])
    grid_y = np.array([])
    for x in xs:
        for y in ys:
            grid_x = np.append(grid_x, x)
            grid_y = np.append(grid_y, y)

    grid = Table().with_columns('x', grid_x, 'y', grid_y)
    return grid.with_column('slope', slope_column(f, grid))


# ----------------------------------------------------------------------
# 3. Turning slopes into equal-length segments
# ----------------------------------------------------------------------

def segment_ends(grid, length, px_per_x=1.0, px_per_y=1.0):
    """Return four arrays x1, y1, x2, y2: the ends of one segment per point.

    Every segment comes out the same length on screen, however steep it is.
    A slope of 100 would make a naive segment 100 times longer than a slope
    of 1, which is the usual reason these plots are hard to read.

    The normalizing is done in pixels, not in data units, so the lengths stay
    equal even when the x and y axes are at different scales.
    """
    xs = grid.column('x')
    ys = grid.column('y')
    ms = grid.column('slope')

    half_px = length * px_per_x / 2

    x1 = np.array([])
    y1 = np.array([])
    x2 = np.array([])
    y2 = np.array([])

    for i in np.arange(len(ms)):
        m = ms.item(i)
        if np.isnan(m):
            # No slope here. Drawn separately as an undefined-point marker.
            continue
        if np.isinf(m):
            # Vertical. Same length, straight up and down.
            dx_px = 0.0
            dy_px = half_px
        else:
            m_px = m * px_per_y / px_per_x       # slope as seen on screen
            dx_px = half_px / np.sqrt(1 + m_px ** 2)
            dy_px = m_px * dx_px
        dx = dx_px / px_per_x                    # back into data units
        dy = dy_px / px_per_y
        x1 = np.append(x1, xs.item(i) - dx)
        y1 = np.append(y1, ys.item(i) - dy)
        x2 = np.append(x2, xs.item(i) + dx)
        y2 = np.append(y2, ys.item(i) + dy)

    return x1, y1, x2, y2


def undefined_points(grid):
    """Return the x and y arrays of the grid points where there is no slope."""
    bad = grid.where('slope', are_nan())
    return bad.column('x'), bad.column('y')


def are_nan():
    """Predicate for Table.where: keeps rows whose value is nan."""
    return lambda v: bool(np.isnan(v))


# ----------------------------------------------------------------------
# 4. Solution curves
# ----------------------------------------------------------------------

def unit_step(f, x, y, ref_x, ref_y):
    """Return a unit vector pointing along the slope field at (x, y).

    This is the same normalizing the segments use: a slope of m becomes the
    direction (1, m) divided by its own length, so a steep slope gives a
    short step in x and a long one in y instead of a huge jump.

    ref_x and ref_y are the direction of travel so far. The tangent is
    flipped when it disagrees, which keeps the curve moving the same way
    around a vertical tangent instead of doubling back on itself.

    Returns (0, 0) when there is no slope, which tells the caller to stop.
    """
    m = safe_slope(f, x, y)
    if np.isnan(m):
        return 0.0, 0.0
    if np.isinf(m):
        step_x = 0.0
        step_y = 1.0
    else:
        size = np.sqrt(1 + m * m)
        step_x = 1.0 / size
        step_y = m / size
    if step_x * ref_x + step_y * ref_y < 0:
        step_x = -step_x
        step_y = -step_y
    return step_x, step_y


def solve_curve(f, x0, y0, h=0.05, method='rk4', direction=1,
                window=(-5, 5, -5, 5), max_steps=6000, escape=1e6):
    """Follow the slope field from (x0, y0) and return the x and y arrays.

    direction is 1 to set off to the right from the starting point, -1 left.

    method 'rk4' walks along the curve itself, a fixed distance per step.
    Walking along the curve is what lets it round a vertical tangent, the
    place where stepping in x alone throws a curve across the gap and makes
    it rattle. method 'euler' is the textbook rule, y + h*f(x, y) with x
    stepping forward, kept so you can watch how far it drifts.

    The loop stops when the curve leaves the window, when the slope stops
    being a number, or when the value runs away.
    """
    xmin, xmax, ymin, ymax = window
    pad_x = 0.02 * (xmax - xmin)
    pad_y = 0.5 * (ymax - ymin)

    x = float(x0)
    y = float(y0)
    xs = np.array([x])
    ys = np.array([y])

    # Which way we set off. For a vertical start, straight up or down.
    m0 = safe_slope(f, x, y)
    if np.isnan(m0):
        return xs, ys
    if np.isinf(m0):
        ref_x, ref_y = 0.0, float(direction)
    else:
        size0 = np.sqrt(1 + m0 * m0)
        ref_x = direction / size0
        ref_y = direction * m0 / size0

    for i in np.arange(max_steps):
        if method == 'euler':
            step = direction * h
            m = safe_slope(f, x, y)
            if not np.isfinite(m):
                break
            # Euler cannot cope with a near vertical tangent: one step would
            # leap most of the window. Stop rather than draw a false jump.
            if abs(step * m) > 0.25 * (ymax - ymin):
                break
            x_new = x + step
            y_new = y + step * m
        else:
            k1x, k1y = unit_step(f, x, y, ref_x, ref_y)
            if k1x == 0.0 and k1y == 0.0:
                break
            k2x, k2y = unit_step(f, x + h / 2 * k1x, y + h / 2 * k1y, k1x, k1y)
            k3x, k3y = unit_step(f, x + h / 2 * k2x, y + h / 2 * k2y, k1x, k1y)
            k4x, k4y = unit_step(f, x + h * k3x, y + h * k3y, k1x, k1y)
            if (k2x == 0.0 and k2y == 0.0) or (k3x == 0.0 and k3y == 0.0) \
                    or (k4x == 0.0 and k4y == 0.0):
                break
            x_new = x + h * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            y_new = y + h * (k1y + 2 * k2y + 2 * k3y + k4y) / 6
            ref_x = k1x
            ref_y = k1y

        if not np.isfinite(x_new) or not np.isfinite(y_new):
            break
        if abs(y_new) > escape or abs(x_new) > escape:
            break

        x = x_new
        y = y_new
        xs = np.append(xs, x)
        ys = np.append(ys, y)

        if x < xmin - pad_x or x > xmax + pad_x:
            break
        if y < ymin - pad_y or y > ymax + pad_y:
            break

    return xs, ys


def solution_through(f, x0, y0, **kwargs):
    """Return the full curve through (x0, y0), going both directions."""
    right_x, right_y = solve_curve(f, x0, y0, direction=1, **kwargs)
    left_x, left_y = solve_curve(f, x0, y0, direction=-1, **kwargs)
    # The left half is backwards, so flip it and drop its duplicated first point.
    xs = np.append(left_x[::-1], right_x[1:])
    ys = np.append(left_y[::-1], right_y[1:])
    return xs, ys


# ----------------------------------------------------------------------
# 5. Drawing
# ----------------------------------------------------------------------

CURVE_COLORS = ['#C2410C', '#9D174D', '#5B21B6', '#0F766E', '#A16207']


def slope_field(f, xlim=(-5, 5), ylim=(-5, 5), nx=21, ny=21, length=0.7,
                starts=(), h=0.05, method='rk4', show_euler=False,
                equal_axes=True, label=None, save_as=None, figsize=(7, 7),
                return_grid=False):
    """Draw the slope field for dy/dx = f(x, y), plus a curve per start point.

    starts is a list of (x, y) pairs. Each one gets its own curve and color.
    nx and ny set how many segments go across and up.
    length is the segment length as a fraction of the horizontal spacing.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim
    step_x = (xmax - xmin) / (nx - 1)

    grid = make_grid(f, xmin, xmax, ymin, ymax, nx, ny)

    figure = plots.figure(figsize=figsize)
    axes = figure.add_subplot(1, 1, 1)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    if equal_axes:
        axes.set_aspect('equal')

    # Pixels per data unit, so segment length can be set on screen.
    figure.canvas.draw()
    box = axes.get_window_extent()
    px_per_x = box.width / (xmax - xmin)
    px_per_y = box.height / (ymax - ymin)

    x1, y1, x2, y2 = segment_ends(grid, length * step_x, px_per_x, px_per_y)
    for i in np.arange(len(x1)):
        axes.plot([x1.item(i), x2.item(i)], [y1.item(i), y2.item(i)],
                  color='#334155', linewidth=1.1, solid_capstyle='round')

    bad_x, bad_y = undefined_points(grid)
    if len(bad_x) > 0:
        axes.plot(bad_x, bad_y, linestyle='none', marker='o',
                  markersize=2.5, color='#94A3B8')

    axes.axhline(0, color='#64748B', linewidth=0.8)
    axes.axvline(0, color='#64748B', linewidth=0.8)
    axes.grid(True, color='#E2E8F0', linewidth=0.6)
    axes.set_axisbelow(True)

    window = (xmin, xmax, ymin, ymax)
    for n in np.arange(len(starts)):
        x0, y0 = starts[int(n)]
        color = CURVE_COLORS[int(n) % len(CURVE_COLORS)]
        xs, ys = solution_through(f, x0, y0, h=h, method=method, window=window)
        axes.plot(xs, ys, color=color, linewidth=2.2,
                  label='through (' + str(x0) + ', ' + str(y0) + ')')
        if show_euler:
            ex, ey = solution_through(f, x0, y0, h=h, method='euler',
                                      window=window)
            axes.plot(ex, ey, color=color, linewidth=1.2, linestyle='--',
                      alpha=0.8)
        axes.plot([x0], [y0], linestyle='none', marker='o', markersize=5,
                  color=color)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    title = label if label is not None else 'Slope field'
    if show_euler and len(starts) > 0:
        title = title + '   (solid: RK4, dashed: Euler)'
    axes.set_title(title)
    if len(starts) > 0:
        axes.legend(loc='upper left', fontsize=8, framealpha=0.9)

    undefined_count = len(bad_x)
    if undefined_count > 0:
        axes.text(0.99, 0.01,
                  str(undefined_count) + ' points have no slope',
                  transform=axes.transAxes, ha='right', va='bottom',
                  fontsize=8, color='#64748B')

    figure.tight_layout()
    if save_as is not None:
        figure.savefig(save_as, dpi=150)

    if return_grid:
        return grid
    return axes


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    def linear(x, y):
        return x + y

    def over_y(x, y):
        return x / y

    def squared(x, y):
        return y ** 2

    def waves(x, y):
        return np.sin(x) * np.cos(y)

    slope_field(linear, starts=[(0, 0), (0, 1), (0, -2)],
                label="dy/dx = x + y", save_as='out_linear.png')
    slope_field(over_y, starts=[(2, 2), (-3, 1)],
                label="dy/dx = x / y", save_as='out_over_y.png')
    slope_field(squared, starts=[(0, 1), (0, -1)], show_euler=True,
                label="dy/dx = y^2", save_as='out_squared.png')
    slope_field(waves, starts=[(0, 0.5), (1, 3)],
                label="dy/dx = sin(x)cos(y)", save_as='out_waves.png')
    slope_field(linear, xlim=(-5, 5), ylim=(-50, 50), nx=21, ny=17,
                equal_axes=False, starts=[(0, 0), (-4, 30)],
                label="stretched axes, ticks still equal",
                figsize=(7, 5), save_as='out_stretched.png')

    print('Saved five sample plots.')
    print()
    print('Slope table for dy/dx = x + y near the origin:')
    table = make_grid(linear, -1, 1, -1, 1, 5, 5)
    print(table)
