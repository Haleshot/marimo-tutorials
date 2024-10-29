# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pandas==2.2.3",
#     "pygwalker==0.4.9.12",
#     "openai==1.52.2",
#     "polars==1.12.0",
# ]
# ///

import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __(pd, walk):
    _df = pd.read_csv("../Exploratory-Data-Analysis/assets/books_enriched.csv")

    walk(_df, spec="spec.json")
    return


@app.cell
def __(pd, pyg):
    _df = pd.read_csv("../Exploratory-Data-Analysis/assets/books_enriched.csv")
    pyg.walk(_df, spec="spec.json")
    return


@app.cell(hide_code=True)
def __():
    # import libraries
    import marimo as mo
    import pandas as pd
    from pygwalker.api.marimo import walk
    import pygwalker.api.marimo as pyg
    return mo, pd, pyg, walk


if __name__ == "__main__":
    app.run()
