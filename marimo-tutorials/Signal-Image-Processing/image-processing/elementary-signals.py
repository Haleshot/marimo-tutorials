import marimo

__generated_with = "0.6.16"
app = marimo.App()


@app.cell
def __(np):
    t = np.arange(0, 1, 0.001)
    f = int(input("Enter the Frequency : ")) # Enter it as 5
    x = 5 * np.sin(2 * np.pi * f * t)
    y = np.cos(2 * np.pi * f * t)
    return f, t, x, y


@app.cell
def __(plt, t, x):
    # Plotting the Sine Curve

    plt.title("Sine Curve")
    plt.plot(t, x)
    return


@app.cell
def __(plt, t, y):
    # Plotting the Cosine Curve

    plt.title("Cosine Curve")
    plt.plot(t, y)
    return


@app.cell
def __(plt, t, x, y):
    plt.plot(t, x, t, y)
    return


@app.cell
def __(plt, t, x, y):
    # Plotting plots and stem graph using subplots.

    plt.subplot(2, 2, 1)
    plt.plot(t, x)
    plt.subplot(2, 2, 2)
    plt.stem(t, x)

    plt.subplot(2, 2, 3)
    plt.plot(t, y)
    plt.subplot(2, 2, 4)
    plt.stem(t, y)
    return


@app.cell
def __(np, plt):
    # Unit Step Function using If else and where.
    _t = np.arange(0, 11)
    u = []
    for i in _t:
      if i <= 5:
        u.append(1)
      elif i > 5:
        u.append(0)
    print(_t)
    print(u)

    plt.subplot(1, 2, 1)
    plt.plot(_t, u)

    plt.subplot(1, 2, 2)
    plt.stem(_t, u)

    u = np.where(_t <= 5, 1, 0)
    print(u)
    return i, u


@app.cell
def __(np, plt):
    # Ramp - Continuos and Discrete difference plots.
    _t = np.arange(0, 11)
    _u = []
    for _i in _t:
      if _i >= 0:
        r = _t
        _u.append(_i)
    print(_t)
    print(_u)

    plt.subplot(2, 2, 1)
    plt.plot(r, _t)

    plt.subplot(2, 2, 2)
    plt.stem(r, _t)
    return r,


@app.cell
def __(np, plt):
    # Impulse operations

    _t = np.arange(-11, 11)
    _u = np.where(_t == 0, 1, 0)
    plt.subplot(2, 2, 1)
    plt.plot(_t, _u)

    plt.subplot(2, 2, 2)
    plt.stem(_t, _u)
    return


@app.cell
def __(np, plt, t):
    # Exponential operations

    a = np.exp(t)
    b = -np.exp(t)

    plt.subplot(2, 2, 1)
    plt.plot(t, a)

    plt.subplot(2, 2, 2)
    plt.stem(t, a)

    plt.subplot(2, 2, 3)
    plt.plot(t, b)

    plt.subplot(2, 2, 4)
    plt.stem(t, b)
    return a, b


@app.cell
def __(mo):
    mo.md(rf"## Discrete Time Signal for x(n) = a^n")
    return


@app.cell
def __(np, plt):
    n = np.arange(0, 11, 1)

    case1 = -0.5
    case2 = -2
    case3 = 2
    case4 = 0.5

    # Discrete Time Sginal for x(n) = a^n
    x1 = case1 ** n
    x2 = case2 ** n
    x3 = case3 ** n
    x4 = case4 ** n

    plt.subplot(2, 2, 1)
    plt.stem(n, x1)

    plt.subplot(2, 2, 2)
    plt.stem(n, x2)

    plt.subplot(2, 2, 3)
    plt.stem(n, x3)

    plt.subplot(2, 2, 4)
    plt.stem(n, x4)
    return case1, case2, case3, case4, n, x1, x2, x3, x4


@app.cell
def __(mo):
    mo.md(rf"## Continuos Time Signal for x(n) = a^n")
    return


@app.cell
def __(np, plt):
    _n = np.arange(0, 11, 1)

    _case1 = -0.5
    _case2 = -2
    _case3 = 2
    _case4 = 0.5

    # Discrete Time Sginal for x(n) = a^n
    _x1 = _case1 ** _n
    _x2 = _case2 ** _n
    _x3 = _case3 ** _n
    _x4 = _case4 ** _n

    plt.subplot(2, 2, 1)
    plt.plot(_n, _x1)

    plt.subplot(2, 2, 2)
    plt.plot(_n, _x2)

    plt.subplot(2, 2, 3)
    plt.plot(_n, _x3)

    plt.subplot(2, 2, 4)
    plt.plot(_n, _x4)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Time Shifting Operation:
        ### Time Reverse and Delay:
        #### y(n) = x(-n + 1)
        """
    )
    return


@app.cell
def __(n, np, plt):
    _n = np.arange(0, 5)

    for _i in n:
      _x = _n + 1
      _y = - _n + 2


    plt.subplot(1, 2, 1)
    plt.stem(_x, _n)

    plt.subplot(1, 2, 2)
    plt.stem(_y, _n)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Time Scaling
        ### Upscaling

        #### y = [x(n)]/2
        """
    )
    return


@app.cell
def __(np, plt):
    _n = np.arange(1, 6)
    _x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _y = []

    for _i in range(len(_x)):
      if _i % 2 == 0:
        _y.append(_i)

    plt.stem(_y, _n)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Amplitude Scaling
        ### Amplification

        #### y(n) = 2x(n)
        """
    )
    return


@app.cell
def __(np, plt):
    _n = np.arange(0, 11)
    _x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    _y = []

    for _i in _x:
      _a = 2 * _i
      _y.append(_a)

    plt.stem(_y, _n)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Addition and Subtraction operations:
        ### Subtraction operation:
        #### z(n) = x(n) - y(n)
        """
    )
    return


@app.cell
def __(np, plt):
    _n = np.arange(0, 4)
    _x = [1, 2, 3, 4]
    _y = [5, 6, 7, 8]
    _z = []

    for _i in range(4):
      _a = _x[_i] - _y[_i]
      _z.append(_a)
    plt.stem(_n, _z)
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Multiplication Operations:
        ### z(n) = x(n) * y(n)
        """
    )
    return


@app.cell
def __(np, plt):
    _n = np.arange(0, 4)
    _x = [1, 2, 3, 4]
    _y = [5, 6, 7, 8]
    _z = []

    for _i in range(4):
      _a = _x[_i] * _y[_i]
      _z.append(_a)
    plt.stem(_n, _z)
    return


@app.cell
def __(mo):
    mo.callout("Conclusion: From the experiment conducted, I could understand the following: \n 1. Identify and generate various types of elementary signals. 2. Perform simple operations on the signals. 3. Understanding Continuos and Discrete depiction of signals.:", kind="success")
    return


@app.cell(hide_code=True)
def __():
    # import libraries
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


if __name__ == "__main__":
    app.run()
