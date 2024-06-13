import marimo

__generated_with = "0.6.16"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        rf"""
        # Elementary Signals and Operations in Python

        ## Aim

        The aim of this tutorial is to plot various elementary signals in Python and perform simple operations on discrete time signals.

        ## Table of Contents

        - [Aim](#aim)
        - [Software](#software)
        - [Prerequisite](#prerequisite)
        - [Outcome](#outcome)
        - [Theory](#theory)
          - [Elementary Signals](#elementary-signals)
          - [Operations on Signals](#operations-on-signals)
        - [Instructions](#instructions)
        - [Examples](#examples)

        ## Software

        This tutorial is implemented using Python.

        ## Prerequisite

        To understand and work with this tutorial, you should be familiar with the following concepts:

        | Sr. No | Concepts          |
        | ------ | ----------------- |
        | 1.     | Elementary signals |

        ## Outcome

        After successful completion of this experiment, students will be able to:

        - Identify and generate various types of elementary signals.
        - Perform simple operations on the signals.

        ## Theory

        ### Elementary signals

        In this tutorial, we will work with the following elementary signals:

        1. Sine wave: A sinusoidal signal with a specified frequency and amplitude.
        2. Unit step: A signal that is 1 for n ≥ 0 and 0 otherwise.
        3. Unit ramp: A signal that is equal to the index n for n ≥ 0 and 0 otherwise.
        4. Unit impulse: A signal that is 1 at n = 0 and 0 otherwise.
        5. Exponential signal: A signal of the form x(n) = a^n, where a is a constant and n ≥ 0.

        ### Operations on Signals

        We will perform the following operations on discrete time signals:

        1. Time shifting: Shifting a signal by k units in time.
        2. Time reversal: Reversing the order of the signal.
        3. Time scaling: Scaling the time axis of the signal by a factor of k.
        4. Scalar multiplication: Multiplying the signal by a constant scalar.

        ## Instructions

        To work with this tutorial, follow the instructions below:

        1. Understand the concepts of elementary signals and operations on signals.
        2. Use Python to implement the required functionalities.
        3. Run the code to generate plots for the elementary signals.
        4. Perform the specified operations on the discrete time signals.
        5. Observe the results and analyze the differences between continuous time, discrete time, and digital signals.

        ## Examples

        Here are some examples of the elementary signals and operations on signals that you can explore in this tutorial:

        1. Sine wave: Generate and plot a sine wave signal with a specific frequency and amplitude.
        2. Unit step: Create a discrete time signal that represents a unit step function.
        3. Unit ramp: Generate a discrete time signal that represents a unit ramp function.
        4. Unit impulse: Create a discrete time signal that represents a unit impulse function.
        5. Exponential signal: Generate a discrete time signal using the exponential function.
        6. Time shifting: Shift a discrete time signal by a specified number of units.
        7. Time reversal: Reverse the order of a discrete time signal.
        8. Time scaling: Scale the time axis of a discrete time signal by a factor.
        9. Scalar multiplication: Multiply a discrete time signal by a constant scalar.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ### 1. Sine wave: Generate and plot a sine wave signal with a specific frequency and amplitude.
        #### We use an example of amplitude (A or λ) = 5 and frequency (or Time period as chosen by user) as 5.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        The sinusoidal function can be represented as:

        \[
            s(t) = A sin(2πft).
        \]

        \[
        or
        \]
     

        \[
            s(t) = A sin(\frac{2πt}{T}).
        \]



        where:

        ### 
        - A: is called the amplitude of the wave, i.e. the largest value of the wave above or below the horizontal axis.
        - f: is the frequency.
        - t: is the time (in seconds).
        - T: Time period (relation between frequency and time period shown above).
        """
    )
    return


@app.cell
def __(mo):
    _note = mo.md(
        r'''

        Relation between Time period (s) and Frequency (Hz):
        
        \[
            T ∝  \frac{1} {f}\
        \]
        
        '''
    )
    mo.callout(_note, kind="info")
    return


@app.cell
def __(mo):
    frequency = mo.ui.slider(start=1, stop=10, step=1, value=5, label="Frequency")
    amplitude = mo.ui.slider(start=1, stop=10, step=1, value=5, label="Amplitude")
    # time_period = mo.ui.slider(start=1, stop=10, step=1, value=5, label="Time period")
    return amplitude, frequency


@app.cell
def __(amplitude, frequency):
    slider_options = [
        frequency,
        amplitude,
        
    ]
    return slider_options,


@app.cell
def __(component_options, mo, set_component_radio_tracker):
    # will be used to track which option the user has selected (time period or frequency)
    get_slider_tracker, set_slider_tracker = mo.state(set())

    component_slider = mo.ui.radio(
        component_options,
        label="**Component Class**",
        on_change=lambda w: set_component_radio_tracker(
            lambda v: v.union({w})
        ),
    )
    other_component_radio = mo.ui.slider(
        component_options, label="**Component Class**"
    )
    return (
        component_slider,
        get_slider_tracker,
        other_component_radio,
        set_slider_tracker,
    )


@app.cell
def __(component_radio, other_component_radio):
    sliders = [component_radio, other_component_radio]
    return sliders,


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(mo, selected_problem, solved):
    # Solve part 1 callout
    (
        mo.md(
            f"""
            **Part 1** taught you the basics of signal decomposition. In
            **Part 2**, you'll apply what you learned to decompose some real-world
            signals, as well as some more synthetic ones.

            Start by choosing a signal. We
            recommend starting with the CO~2~ signal, which tracks atmospheric
            emissions at the Mauna Loa Observatory.

            {selected_problem.center()}
            """
        )
        if solved.ever
        else mo.md(
            """
            🛑 Part 2 isn't available yet. Keep experimenting with
            component classes in part 1 until you've "solved" the decomposition,
            then return here.
            """
        ).callout(kind="alert")
    )
    return


@app.cell
def __():
    class StickyBool:
        value = False

        def set(self):
            self.value = True
            return self

        def __bool__(self):
            return self.value

    solved_ever = StickyBool()
    return StickyBool, solved_ever


@app.cell
def __(complib, solved_ever):
    class Solved:
        def __init__(self, sticky_bool):
            self.ever = sticky_bool
            self.now = False

    solved = Solved(solved_ever)
    solved.now = _chosen_components == set(
        [complib.Components.TREND_LINE, complib.Components.PERIODIC]
    )
    if solved.now:
        solved_ever.set()
    return Solved, solved


@app.cell
def __(mo, solved):
    # Solved callout
    (
        mo.md(
            f"""
            🎉 **_You did it!_**

            The noise is small and looks random, and the decomposition has
            linear and seasonal components. In fact, the signal was
            generated by adding a line to a sine wave.

            In the real world, where signals are measurements of messy
            data, you won't ever know if you've "solved" a signal
            decomposition problem. Instead you'll have to use your own
            intuition to guide the selection of component classes.

            In this sense, signal decomposition is kind of like
            unsupervised machine learning tasks, like clustering or
            embedding: it's up to you to judge whether or not your
            decomposition is a good one.

            **Part 2 is now available**.
            """
        ).callout(kind="success")
        if solved.now
        else None
    )
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(np):
    t = np.arange(0, 1, 0.001)
    f = int(input("Enter the Frequency : ")) # Enter it as 5
    x = 5 * np.sin(2 * np.pi * f * t)
    y = np.cos(2 * np.pi * f * t)

    # Sine wave of frequency= 5 Hz and amplitude= 5V (default placeholder values)
    return f, t, x, y


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(np, plt):
    def plot_sine_wave(frequency, amplitude):
        x = np.linspace(0, 2*np.pi, num=100)
        plt.figure(figsize=(6.7, 2.5))
        plt.plot(x, amplitude*np.sin(x*2*np.pi * frequency))
        plt.xlabel('$x$')
        plt.xlim(0, 2*np.pi)
        plt.ylim(-2, 2)
        plt.tight_layout()
        plt.title("Sinusoidal Wave")
        return plt.gca()
    return plot_sine_wave,


@app.cell
def __(np, plt):
    def plot_cosine_wave(frequency, amplitude):
        x = np.linspace(0, 2*np.pi, num=100)
        plt.figure(figsize=(6.7, 2.5))
        plt.plot(x, amplitude*np.cos(x*2*np.pi * frequency))
        plt.xlabel('$x$')
        plt.xlim(0, 2*np.pi)
        plt.ylim(-2, 2)
        plt.tight_layout()
        plt.title("Cosine Wave")
        return plt.gca()
    return plot_cosine_wave,


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
    text_input = mo.md(
        r'''

        ## Conclusion:
        
        From the experiment conducted,you should be able to understand the following:
        
        - Identify and generate various types of elementary signals.
        
        - Perform simple operations on the signals.
        
        - Understanding Continuos and Discrete depiction of signals.
        '''
    )
    mo.callout(text_input, kind="success")
    return text_input,


@app.cell
def __():
    # import libraries
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def __(mo):
    mo.sidebar(
        [
            mo.md("# Elementary Signals"),
            mo.nav_menu(
                {
                    "#aim": f"{mo.icon('lucide:home')} Home",
                    "Page Overview": {
                        "#aim": "Aim",
                        "#table-of-contents": "Table of Contents",
                        "#software": "Software",
                        "#prerequisite": "Prerequisites",
                        "#outcome": "Outcomes",
                    },
                    "Theory": {
                        "#elementary-signals": "Elementary Signals",
                        "#operations-on-signals": "Operations on Signals",
                    },
                    "#instructions": "Instructions",
                    "#examples": "Examples",
                    
                    "Implementation": {
                        "#defining-masks-as-discussed-in-theory-above": "Defining Masks",
                        "#horizontal-edge-detection-using-list-slicing": "Horizontal Mask definition",
                        "#vertical-edge-detection-using-list-slicing": "Vertical Mask definition",
                        "#diagonal-edge-detection-using-list-slicing": "Diagonal Mask definition",
                        "#showing-the-differences-of-edge-detection-using-various-masks-defined-above" : "Comparison of user defined masks",
                        "#showing-the-differences-of-edge-detection-using-in-built-masks-from-above" : "Comparison of built-in defined masks",
                        "#for-different-sizes-of-masks-provided-from-user" : "Try it out!"
                    },
                    "#observations": "Observations",
                    "#references": "References",
                    "Links": {
                        "https://github.com/Haleshot/marimo-tutorials/blob/haleshot/marimo-tutorials/Signal-Image-Processing/signal-processing/elementary-signals.py": f"{mo.icon('lucide:github')} GitHub",
                    },
                },
                orientation="vertical",
            ),
        ]
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
