import marimo

__generated_with = "0.6.17"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    >import marimo as mo
    import pandas as pd
    import numpy as np
    import scipy as sp
    """,
    name="__"
)


@app.cell
def __(mo):
    mo.md(
        rf"""
        <h1> Statistics through Cricket</h1> 
        <u> <h2> Table of Contents </h2> </u>
                <ol>
                    <li> Introduction </li>
                    <li> The Teams </li>
                    <li> Title Holders</li>
                    <li> The Next Title Holder </li>
                    <li> Conclusion </li>
                </ol>
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        <u> <h2> Introduction </h2> </u>
        <p>
           Statistics is everywhere - it's the prcess collecting, organizing, analyzing, and presenting data. A sport like cricket has an immense amount of data that can  be used to analyze performance throughout the game, and tournament.<a href="https://www.icc-cricket.com/tournaments/t20cricketworldcup/index"> The T20 World Cup </a> is an international sporting event that brings the world's best talent together every two years. Through exploratory data analysis (EDA), this notebook will look at the participating teams of this year's tournament, and see if there are any dominating forces, or emerging forces. 
        </p>
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        <h2> The Teams </h2>
        This is the dataset that'll be used for this analysis:
        """
    )
    return


@app.cell
def __(mo, pd):
    Cricket_Data = pd.read_csv("T20_Data.csv")
    mo.ui.table(Cricket_Data)
    return Cricket_Data,


@app.cell
def __(mo):
    mo.md(rf"<caption>Figure 1: Table showing countries playing the T20 cricket format.</caption>")
    return


@app.cell
def __(mo):
    Group_A = mo.md(
        rf"""
        <table>
            <tr>
                <th> Group A </th>
            </tr>
            <tr>
                <td> Canada </td>
            </tr>
            <tr>
                <td> India </td>
            </tr>
            <tr>
                <td> Ireland </td>
            </tr>
            <tr>
                <td> Pakistan </td>
            </tr>
            <tr>
                <td> United States of America (USA) </td>
            </tr>
        </table>
        """
    )

    Group_B = mo.md(
        rf"""
            <table>
            <tr>
                <th> Group B </th>
            </tr>
            <tr>
                <td> Afghanistan </td>
            </tr>
            <tr>
                <td> Papua New Guinea (P.N.G) </td>
            </tr>
            <tr>
                <td> New Zealand </td>
            </tr>
            <tr>
                <td> Uganda </td>
            </tr>
            <tr>
                <td> West Indies </td>
            </tr>
        </table>
        """
    )

    Group_C = mo.md(
        rf"""
            <table>
            <tr>
                <th> Group C </th>
            </tr>
            <tr>
                <td> Australia </td>
            </tr>
            <tr>
                <td> England </td>
            </tr>
            <tr>
                <td> Namibia </td>
            </tr>
            <tr>
                <td> Scotland </td>
            </tr>
            <tr>
                <td> Oman </td>
            </tr>
        </table>
        """
    )

    Group_D = mo.md(
        rf"""
            <table>
            <tr>
                <th> Group D </th>
            </tr>
            <tr>
                <td> Bangladesh </td>
            </tr>
            <tr>
                <td> Nepal </td>
            </tr>
            <tr>
                <td> Netherlands </td>
            </tr>
            <tr>
                <td> South Africa </td>
            </tr>
            <tr>
                <td> Sri Lanka </td>
            </tr>
        </table>
        """
    )
    return Group_A, Group_B, Group_C, Group_D


@app.cell
def __(mo):
    mo.md(rf"<p> These are the teams participating in this year's tournament, and we'll just be focusing on them:</p>")
    return


@app.cell
def __(Group_A, Group_B, Group_C, Group_D, mo):
    Participating_Teams = mo.hstack([
        Group_A,
        Group_B,
        Group_C,
        Group_D,
    ])
    Participating_Teams
    return Participating_Teams,


if __name__ == "__main__":
    app.run()
