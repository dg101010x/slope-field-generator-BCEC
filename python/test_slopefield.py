"""Tests for slopefield.py. Run with: python3 test_slopefield.py"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
from slopefield import (safe_slope, make_grid, segment_ends, undefined_points,
                        solve_curve, solution_through, slope_field)


def check(name, condition, detail=''):
    if condition:
        print('  ok   ' + name + ('   ' + detail if detail else ''))
    else:
        print('  FAIL ' + name + ('   ' + detail if detail else ''))
        raise AssertionError(name)


# ----------------------------------------------------------------------
print('safe_slope')
# ----------------------------------------------------------------------

check('plain value', safe_slope(lambda x, y: x + y, 2, 3) == 5)
check('divide by zero gives inf', np.isinf(safe_slope(lambda x, y: x / y, 1, 0)))
check('log of zero gives nan', np.isnan(safe_slope(lambda x, y: np.log(y), 1, 0))
      or np.isinf(safe_slope(lambda x, y: np.log(y), 1, 0)))
check('sqrt of negative gives nan',
      np.isnan(safe_slope(lambda x, y: np.sqrt(y), 1, -4)))


# ----------------------------------------------------------------------
print('make_grid')
# ----------------------------------------------------------------------

grid = make_grid(lambda x, y: x + y, -1, 1, -1, 1, 5, 5)
check('row count is 5 by 5', grid.num_rows == 25, str(grid.num_rows))
check('columns are x, y, slope',
      list(grid.labels) == ['x', 'y', 'slope'], str(grid.labels))
check('slope matches x + y',
      np.allclose(grid.column('slope'), grid.column('x') + grid.column('y')))


# ----------------------------------------------------------------------
print('equal segment lengths')
# ----------------------------------------------------------------------

steep = make_grid(lambda x, y: 50 * x + 1, -2, 2, -2, 2, 9, 9)
x1, y1, x2, y2 = segment_ends(steep, 0.35, px_per_x=40.0, px_per_y=40.0)
lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
check('all equal on square axes',
      np.allclose(lengths, lengths.item(0)),
      'spread ' + str(round(float(np.max(lengths) - np.min(lengths)), 12)))

# Now the stretched case: 10x more pixels per x unit than per y unit.
x1, y1, x2, y2 = segment_ends(steep, 0.35, px_per_x=100.0, px_per_y=10.0)
px_len = np.sqrt(((x2 - x1) * 100.0) ** 2 + ((y2 - y1) * 10.0) ** 2)
check('all equal in pixels on stretched axes',
      np.allclose(px_len, px_len.item(0)),
      'spread ' + str(round(float(np.max(px_len) - np.min(px_len)), 10)) + ' px')

vertical = make_grid(lambda x, y: x / y, -1, 1, -1, 1, 5, 5)
vx1, vy1, vx2, vy2 = segment_ends(vertical, 0.35, 40.0, 40.0)
vlen = np.sqrt((vx2 - vx1) ** 2 + (vy2 - vy1) ** 2)
check('vertical segments are the same length too',
      np.allclose(vlen, 0.35), str(round(float(np.max(vlen)), 4)))


# ----------------------------------------------------------------------
print('undefined points')
# ----------------------------------------------------------------------

bad_x, bad_y = undefined_points(make_grid(lambda x, y: np.sqrt(y), -1, 1, -1, 1, 5, 5))
check('sqrt(y) has no slope below the axis', len(bad_x) == 10, str(len(bad_x)))

on_axis = make_grid(lambda x, y: x / y, -1, 1, -1, 1, 5, 5)
check('x/y on the axis is vertical, not undefined',
      len(undefined_points(on_axis)[0]) == 0)
check('x/y on the axis is infinite',
      int(np.sum(np.isinf(on_axis.column('slope')))) == 5,
      str(int(np.sum(np.isinf(on_axis.column('slope'))))))


# ----------------------------------------------------------------------
print('solver against known answers')
# ----------------------------------------------------------------------

# y' = y, y(0) = 1  ->  y = e^x
# Compared point by point, so the check measures the solver and not the
# straight lines drawn between its points.
xs, ys = solve_curve(lambda x, y: y, 0, 1, h=0.05, method='rk4',
                     window=(0, 1, -10, 10))
rk4_error = float(np.max(np.abs(ys - np.e ** xs)))
check('rk4 matches e^x everywhere along the curve',
      rk4_error < 1e-6, 'worst error ' + str(round(rk4_error, 11)))

xs, ys = solve_curve(lambda x, y: y, 0, 1, h=0.05, method='euler',
                     window=(0, 1, -10, 10))
euler_at_one = np.interp(1.0, xs, ys)
euler_error = abs(euler_at_one - np.e)
check('euler is close but worse, as it should be',
      0.001 < euler_error < 0.2,
      'error ' + str(round(euler_error, 6)))

# y' = x + y, y(0) = 1  ->  y = 2e^x - x - 1
xs, ys = solve_curve(lambda x, y: x + y, 0, 1, h=0.05, method='rk4',
                     window=(0, 2, -50, 50))
worst = float(np.max(np.abs(ys - (2 * np.e ** xs - xs - 1))))
check('rk4 matches 2e^x - x - 1 everywhere along the curve',
      worst < 1e-5, 'worst error ' + str(round(worst, 10)))


# ----------------------------------------------------------------------
print('blow-up guard')
# ----------------------------------------------------------------------

# y' = y^2, y(0) = 1 has a vertical asymptote at x = 1.
xs, ys = solve_curve(lambda x, y: y ** 2, 0, 1, h=0.05, method='rk4',
                     window=(-5, 5, -5, 5), max_steps=4000)
check('stops before the asymptote',
      xs.item(-1) < 1.0, 'last x = ' + str(round(float(xs.item(-1)), 4)))
check('stopped early, did not run out of steps', len(xs) < 4000, str(len(xs)))
check('every value is finite', bool(np.all(np.isfinite(ys))))

# Same thing with Euler, which is the case the forums complain about.
xs_e, ys_e = solve_curve(lambda x, y: y ** 2, 0, 1, h=0.05, method='euler',
                         window=(-5, 5, -5, 5), max_steps=4000)
check('euler also stops cleanly', bool(np.all(np.isfinite(ys_e))))


# ----------------------------------------------------------------------
print('vertical tangent')
# ----------------------------------------------------------------------

# y' = x/y has solutions y^2 - x^2 = C, hyperbolas that turn vertical where
# they cross the x-axis. Stepping in x alone throws the curve across that
# turn and it rattles. Through (-3, 1) the constant is 1 - 9 = -8.
xs, ys = solution_through(lambda x, y: x / y, -3, 1, h=0.05,
                          window=(-5, 5, -5, 5))
invariant = ys ** 2 - xs ** 2
drift = float(np.max(np.abs(invariant - (-8.0))))
check('stays on its own hyperbola the whole way',
      drift < 1e-3, 'worst drift ' + str(round(drift, 8)))
check('actually rounds the turn, x goes back on itself',
      bool(np.any(np.diff(xs) < 0)))
check('no rattling, steps stay small',
      float(np.max(np.abs(np.diff(ys)))) < 0.1,
      'biggest y jump ' + str(round(float(np.max(np.abs(np.diff(ys)))), 4)))


# ----------------------------------------------------------------------
print('curve through a point, both directions')
# ----------------------------------------------------------------------

xs, ys = solution_through(lambda x, y: x + y, 0, 1, h=0.05,
                          window=(-5, 5, -5, 5))
check('x increases left to right', bool(np.all(np.diff(xs) > 0)))
check('spans both sides of the start', xs.item(0) < 0 < xs.item(-1),
      str(round(float(xs.item(0)), 2)) + ' to ' + str(round(float(xs.item(-1)), 2)))
check('passes through the start point',
      abs(np.interp(0.0, xs, ys) - 1.0) < 1e-6)


# ----------------------------------------------------------------------
print('whole plot runs')
# ----------------------------------------------------------------------

for name, func in [('x + y', lambda x, y: x + y),
                   ('x / y', lambda x, y: x / y),
                   ('y^2', lambda x, y: y ** 2),
                   ('sin(x)cos(y)', lambda x, y: np.sin(x) * np.cos(y)),
                   ('ln(y)', lambda x, y: np.log(y))]:
    slope_field(func, starts=[(0, 1), (1, -2)], save_as='check_tmp.png')
    check('draws ' + name, True)

print()
print('All checks passed.')
