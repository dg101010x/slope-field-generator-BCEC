# Slope field generator

Draws the slope field for dy/dx = f(x, y) and traces the solution curve through any point you pick.

Two pieces that run the same math.

- **`index.html`** is the web tool. One file, no build step, no dependencies. Type an equation, click the plot to trace a curve.
- **`python/slopefield.py`** is the same program written with the tools from Data 8 / COMSC 028: functions, numpy arrays, loops, Tables, matplotlib. Nothing else.
- **`python/test_slopefield.py`** checks the math against answers you can work out by hand.

## Layout

```
index.html                   the web tool, deployed as-is
python/slopefield.py         the Data 8 version
python/test_slopefield.py    the checks
python/requirements.txt
```

## Running the web one

Open `index.html` in a browser. That is the whole setup. To serve it locally:

```
python3 -m http.server 8000
```

## Using the Python one

```python
from slopefield import slope_field

def f(x, y):
    return x + y

slope_field(f, starts=[(0, 1), (0, -2)])
```

Settings you will actually reach for:

| Argument | Does |
|----------|------|
| `xlim`, `ylim` | the window, e.g. `xlim=(-2, 8)` |
| `nx`, `ny` | how many segments across and up, default 21 each |
| `length` | segment length as a fraction of the spacing, default 0.7 |
| `starts` | list of `(x, y)` points to trace curves through |
| `h` | step size along the curve, default 0.05 |
| `method` | `'rk4'` or `'euler'` |
| `show_euler` | draw Euler dashed on top, to see it drift |
| `equal_axes` | square aspect, on by default |
| `save_as` | `'field.png'`, or `.svg`, or `.pdf` |
| `return_grid` | hand back the Table instead of the axes |

The Table behind the picture:

```python
from slopefield import make_grid
print(make_grid(f, -1, 1, -1, 1, 5, 5))     # x, y, slope at 25 points
```

From the `python/` folder, run `python3 slopefield.py` to save five sample plots, and `python3 test_slopefield.py` to run the checks.

Needs `numpy`, `matplotlib`, and `datascience`:

```
pip install -r python/requirements.txt
```

## Where the feature list came from

Each row is something people complain about in slope field tools, and what this does about it. Sources at the bottom.

| Complaint | What it does |
|-----------|--------------|
| Arrows come out different lengths, and the ones with small slopes are unreadable | Every segment is the same length. Direction carries the slope, length carries nothing. |
| Equal lengths fall apart on stretched axes, or change between plots | Lengths are normalized in pixels, not data units, so they hold at any window shape. There is a test for this. |
| Steep slopes look bad | A near-vertical slope draws a vertical tick of the same length, not a spike. |
| Only works in one window, about -5 to 5 | You count segments instead of fixing a step size, so the grid spaces itself to whatever window you give it. A fixed step of 0.5 is what stacks 200 rows on top of each other the moment you plot y from -50 to 50. |
| No control over grid density or segment length | `nx`, `ny`, `length`. Sliders in the web version. |
| Hard to set initial conditions, hard to compare several | A list of starting points, each its own color. Click the plot in the web version. |
| Solution curves are hard to get; the symbolic route returns implicit or complex answers | Curves are numeric. Nothing is solved symbolically, so nothing comes back as an implicit log equation. |
| Euler misbehaves near asymptotes | Two guards. The curve stops when it leaves the window or runs away, and Euler stops rather than draw a leap it cannot justify. RK4 is the default. |
| Unclear input syntax, and silent wrong answers when it misreads you | Python: you edit one `def f(x, y)`, so there is nothing to misparse. Web: it writes back how it read your equation, and errors name the piece that is wrong. |
| Output is one row of segments, or garbage text, or nothing | Defaults that always draw a full field. |
| Setup problems, missing files | One file, three libraries. |
| No explanation of the math | `make_grid` returns the Table of every slope it used. |
| No way to save the picture | `save_as=` any format matplotlib writes. |

Two more that are my own call, not from a forum:

- **Points with no slope.** `x/y` on the x-axis, `ln(y)` below it. These are counted and marked rather than crashing the plot or silently vanishing.
- **Vertical tangents.** The solutions of `y' = x/y` are hyperbolas that turn vertical where they cross the x-axis. Stepping in x throws the curve across that turn and it rattles. Both versions step along the curve itself instead, a fixed distance per step, which rounds the turn. The test checks the curve stays on `y² - x² = -8` the whole way, and it does, to 2e-9.

## What the tests check

```
rk4 matches e^x everywhere along the curve        worst error 8.9e-09
rk4 matches 2e^x - x - 1 everywhere               worst error 6.2e-08
euler is close but worse, as it should be         error 0.065
stays on its own hyperbola the whole way          worst drift 0.0
stops before the asymptote                        last x = 0.90
all segments equal in pixels on stretched axes    spread 0.0 px
```

## Not in here

Symbolic solutions, second-order equations, systems of two equations, and going backwards from a picture to the equation.

## Sources

Reddit and Quora both block automated reading, so none of the complaints above came from there. These are the threads they did come from.

- [Is there a better slope field program for the TI 83](https://www.physicsforums.com/showthread.php?p=3752780), Physics Forums
- [Get WolframAlpha to plot slope fields](https://www.physicsforums.com/threads/get-wolframalpha-to-plot-slope-fields-to-des.1021543/post-6681178), Physics Forums
- [plotfield thread](https://hpmuseum.org/forum/printthread.php?tid=2253), HP Museum
- [Slope field programs for the TI-83/84](https://www.cemetech.net/projects/uti/viewtopic.php?p=140784), Cemetech
- [Help with MATLAB code to plot slope field](https://www.mathworks.com/matlabcentral/answers/344796-help-with-matlab-code-to-plot-slope-field), MathWorks
- [Quiver with equal arrow lengths](https://python-forum.io/printthread.php?tid=16924), Python Forum
- [Quiver plot with equal arrow lengths](https://www.mathworks.com/matlabcentral/discussions/general/883433-quiver-plot-with-equal-arrow-lengths), MathWorks
- [matplotlib quiver scaling](https://discourse.matplotlib.org/t/matplotlib-quiver/20088), matplotlib Discourse
- [Plotting autonomous differential equations](https://ask.sagemath.org/question/43748/plotting-autonomous-differential-equations/), SageMath Ask
- [SlopeFields 0.1.0](https://pypi.org/project/SlopeFields/0.1.0/), PyPI
