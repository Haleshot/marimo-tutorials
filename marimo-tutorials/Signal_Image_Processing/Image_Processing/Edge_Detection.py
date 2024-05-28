import marimo

__generated_with = "0.6.6"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo, wishes):
    mo.hstack([wishes, wishes.value], justify="space-between")
    return


@app.cell
def __(mo):
    wish = mo.ui.text(placeholder="Wish")
    wishes = mo.ui.array([wish] * 3, label="Three wishes")
    return wish, wishes


if __name__ == "__main__":
    app.run()
