import marimo

__generated_with = "0.6.16"
app = marimo.App()


@app.cell
def __():
    # choice = mo.ui.checkbox(label='Time period instead of frequency')

    # frequency = mo.ui.slider(start=np.pi, stop=2*np.pi, label="Frequency")
    # time_period = mo.ui.slider(start=np.pi, stop=2*np.pi, label="Time period")
    # amplitude = mo.ui.slider(start=1, stop=2, step=0.1, label="Amplitude")
    return


@app.cell
def __(amplitude, mo, time_period):
    mo.md(
        rf"""
        <!-- mo.md(
            f\"""
            Here's a plot of
            $f(x) = {amplitude.value:.02f}\sin((2\pi/{time_period.value:.02f}) x)$:
            \"""
        ) -->
        """
    )
    return


@app.cell
def __():
    # [choice, frequency, time_period, amplitude]
    return


@app.cell
def __(array2, mo, np):
    choice = mo.ui.checkbox(label="Time period instead of frequency", onchange=array2)
    frequency = mo.ui.slider(start=np.pi, stop=2 * np.pi, label="Frequency")
    time_period = mo.ui.slider(start=np.pi, stop=2 * np.pi, label="Time period")
    amplitude = mo.ui.slider(start=1, stop=2, step=0.1, label="Amplitude")
    array = mo.ui.array([choice, frequency, amplitude])
    array2 = mo.ui.array([choice, time_period, amplitude])
    return amplitude, array, array2, choice, frequency, time_period


@app.cell
def __():
    # _array = mo.ui.array([choice, time_period, amplitude])
    # if choice.value:
    #    _array = mo.hstack([_array, array.value], justify="space-between")
    # else:
    #     _array = mo.hstack([array, array.value], justify="space-between")

    # mo.hstack([_array], justify="space-between")

    # mo.hstack([_array], justify="space-between")
    return


@app.cell
def __(array, array2, choice, mo):
    (
        mo.hstack([array2, array2.value], justify="space-between")
        if choice.value
        else mo.hstack([array, array.value], justify="space-between")
    )
    return


@app.cell
def __():
    # mo.hstack([array, array.value], justify="space-between")
    return


@app.cell(hide_code=True)
def __():
    # import libraries
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def __(mo):
    # condition is a boolean, True of False
    choice1 = True
    checkbox = mo.ui.checkbox(label="checkbox")
    checkbox if choice1 else None
    return checkbox, choice1


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
