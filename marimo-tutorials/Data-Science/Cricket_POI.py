import marimo

__generated_with = "0.9.6"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import pandas as pd
    from scipy import interpolate
    import plotly.graph_objects as go
    return go, interpolate, mo, np, pd, px


@app.cell
def __(mo):
    mo.md(
        r"""
        # Statistical Concepts wtih Cricket

        ### Table of Contents 
        <ol>
            <li> General Overview</li> 
            <li> Point of Intersection </li>
            <li> Poisson Distribution </li>
        </ol>
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## General Overview

        <p> The purpose of this notebook is to showcase some statistical concepts through sports, in this case, cricket. We'll be using a dataset obtained from <a href = "https://www.espncricinfo.com/series/england-in-pakistan-2022-1327226/pakistan-vs-england-2nd-t20i-1327229/match-overs-comparison"> Cricinfo </a>. </p>
        """
    )
    return


@app.cell
def __(pd):
    Over = range(1,21,1)
    England_Innings = [
        3,
        13,
        24,
        32,
        42,
        48,
        53,
        62,
        66,
        80,
        93,
        96,
        103,
        114,
        133,
        151,
        160,
        172,
        180,
        199
    ]

    Pakistan_Innings = [
        12,
        16,
        25,
        32,
        46,
        59,
        68,
        74,
        83,
        87,
        99,
        104,
        125,
        135,
        151,
        164,
        176,
        180,
        197,
        203
    ]

    Full_Game = pd.DataFrame(
        {
            'Over': Over,
            'England': England_Innings,
            'Pakistan': Pakistan_Innings
        }
    )
    return England_Innings, Full_Game, Over, Pakistan_Innings


@app.cell
def __(mo):
    mo.md(r"""## Point of Intersection""")
    return


@app.cell
def __(Full_Game, go):
    Worm_Graph = go.Figure()

    Worm_Graph.add_trace(go.Scatter(
        x=Full_Game['Over'], 
        y=Full_Game['England'],
        mode='lines', name='England'))

    Worm_Graph.add_trace(go.Scatter(
        x=Full_Game['Over'], y=Full_Game['Pakistan'],
        mode='lines', name='Pakistan', 
        line=dict(color='green')))

    Worm_Graph.update_layout(title='England vs Pakistan Run Worm',
                      xaxis_title='Over',
                      yaxis_title='Runs')
    return (Worm_Graph,)


@app.cell
def __(mo):
    Question_1 = mo.ui.text(label='What is the point of intersection?').form()
    Question_1
    return (Question_1,)


@app.cell
def __(Question_1):
    Answer_1 = Question_1.value
    return (Answer_1,)


@app.cell
def __(Answer_1, mo):
    mo.stop(Answer_1 is None, mo.md("Submit the form to continue"))
    return


@app.cell
def __(Answer_1):
    def POI_Question(Answer_1):        
        if Answer_1 == None: 
            return 'NA'
        elif Answer_1 == '(4,32)': 
            return "That's correct!"
        else: 
            return "Ohhh, sorry that's not correct"

    POI_Question(Answer_1)
    return (POI_Question,)


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Poisson Distribution
        <p> Under the Poisson distribution, the probability of a particular event is determined by the rate of that event occuring in any given time window. For example, the probability of a boundary being scored in any given over. </p>
    """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        <p> Mathematically, the probability of a particular event under the distribution can be represented as: </p>

        \[
            P(x) = \frac{e^{-\lambda} \cdot \lambda^x}{x!}
        \]

        <p>In our case, lambda is the run rate (average number of runs), and x is the number of runs we're interested in.</p>

        <p> In python, we can use use the <a href = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html"> scipy.stats </a> library with respect to poisson for finding probabilities, or even generating values under the distribution. </p>
        """
    )
    return


@app.cell
def __(Over, pd):
    England_Runs = [
        3,
        10,
        11,
        8,
        10,
        6,
        5,
        9,
        4,
        14,
        13,
        3,
        7,
        11,
        19,
        18,
        9,
        12,
        8,
        19
    ]

    Pakistan_Runs = [
        12,
        4,
        9,
        7,
        14,
        13,
        9,
        6,
        9,
        4,
        12,
        5,
        21,
        10,
        16,
        13,
        12,
        4,
        17,
        6
    ]

    Full_Game_Runs = pd.DataFrame(
        {
            'Over': Over,
            'England': England_Runs,
            'Pakistan': Pakistan_Runs
        }
    )
    return England_Runs, Full_Game_Runs, Pakistan_Runs


@app.cell
def __(Full_Game_Runs, go):
    Run_Poisson_Graph = go.Figure()

    Run_Poisson_Graph.add_trace(go.Scatter(
        x=Full_Game_Runs['Over'], 
        y=Full_Game_Runs['England'],
        mode='lines', name='England'))

    Run_Poisson_Graph.add_trace(go.Scatter(
        x=Full_Game_Runs['Over'], 
        y=Full_Game_Runs['Pakistan'],
        mode='lines', name='Pakistan', 
        line=dict(color='green')))

    Run_Poisson_Graph.update_layout(title='England vs Pakistan Poisson Graph',
                      xaxis_title='Over',
                      yaxis_title='Runs')
    return (Run_Poisson_Graph,)


@app.cell
def __(mo):
    Question_2 = mo.ui.text(label = "What is the probability of England scoring 10 or more runs in any given over?").form()
    Question_2
    return (Question_2,)


@app.cell
def __(Answer_1, POI_Question):
    def Poisson_Question(Answer_2):        
        if Answer_2 == None: 
            return 'NA'
        elif Answer_2 == '0.5668290964267841': 
            return "That's correct!"
        else: 
            return "Ohhh, sorry that's not correct"

    POI_Question(Answer_1)
    return (Poisson_Question,)


@app.cell
def __(mo):
    mo.callout("Hint: You can use marimo's scratchpad to test out your code!",kind = 'info')
    return


if __name__ == "__main__":
    app.run()
