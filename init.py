"""
This is the starting point for your notebook.
"""

import marimo

__generated_with = "0.6.13"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    # Initial Notebook Options
    init_form = (
        mo.md(
            r"""
            **Define Your Initial Notebook Options Here:**
        
            {title}

            {path}

            {author_name}

            {author_contact}
        
            """
        )
        .batch(
            title=mo.ui.text(
                placeholder="The Elements of Marimo",
                label="The title of your notebook: ",
                full_width=True,
            ),
            path=mo.ui.text(
                placeholder="marimo-tutorials/Computer-Science/..",
                label="The path of your notebook: ",
                full_width=True,
            ),
            author_name=mo.ui.text(
                placeholder="marimo-tutorials",
                label="Your name: ",
                full_width=True,
            ),
            author_contact=mo.ui.text(
                placeholder="marimo-tutorials@example.com",
                label="Your contact info: ",
                full_width=True,
            ),
        )
        .form(bordered=True, label="Custom Constants")
    )
    return (init_form,)


@app.cell(hide_code=True)
def __(init_form, mo):
    mo.vstack(
        [
            mo.accordion(
                items={
                    "Design Patterns of Marimo Tutorials": "Marimo + Interactivity + Visualization + Flexibility + Minimalism",
                    "Notebook Structure": mo.md(
                        """
                ```tree
                .
                ├── TODOs
                ├── Suggested Reading & Suggestions for other Tutorials
                ├── Custom Input (optional but suggested)
                ├── Notebook Title
                ├── Abstract
                ├── Introduction
                ├── Main Body
                ├── Summary
                ├── Experiments (optional)
                ├── Exercises (optional)
                ├── References (optional)
                ├── Appendix (optional)
                ├── Source Code
                ├── Authors
                └── Acknowledgments (optional)
                ```
                """
                    ),
                    "Incide Notebook": mo.md(
                        """
                - Remember to write Navigation
                - Remember to add interactivity (`marimo.ui`) and visualizations as you can but avoid repetition
                - You can quote from time to time
                - You can write some short recaps from time to time
                """
                    ),
                    "Best Practise": mo.md(
                        """
                > Check following notebooks on how to maximize `marimo`:

                - []()
                """
                    ),
                },
                lazy=True,
                multiple=True,
            ),
            mo.md(
                rf"""
        <div id="create">
            <b> Create your starting notebook: <b>
        </div>
        """
            ),
            init_form,
            mo.md(rf"> View your starting notebook below and make changes if needed:"),
        ]
    )
    return


@app.cell
def __():
    # mo.ui.fo
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        <h1 id="todos">TODOs</h1>

        1. todo task 1
        2. todo task 2
        3. …
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        **Suggestions for other derived notebooks.**

        1. suggestion 1
        2. suggestion 2
        3. …
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        **Suggestions for reading other notebooks.**

        1. suggestion 1
        2. suggestion 2
        3. …
        """
    )
    return


@app.cell
def __(mo):
    # Custom Constants
    custom_form = (
        mo.md(
            r"""
            **Customize your constants here:**
        
            {image}
        
            """
        )
        .batch(
            image=mo.ui.text(
                value="../assets/<>/<>",
                label="The path of your data: ",
                full_width=True,
            ),  ## add more rows below
        )
        .form(bordered=True, label="Custom Constants")
    )
    custom_form
    return (custom_form,)


@app.cell
def __(NOTEBOOK_TITLE, mo):
    mo.md(
        f"""
        <h1 id="home">{NOTEBOOK_TITLE}</h1>
        ---

        **Abstract**

        > write abstract here

        > > use hyperlinks like [section x](#x) to help with quick navigation

        <h1 id="intro">Introduction</h1>

        > a concise intro with an outline of the theoretical part of the notebook
        > > optional tips for using the notebook

        The notebook is organized as follows:

        1. **Section 1:**

            - concise descriptions

        2. **Section 2:**

        >
        > > In the `Experiment` section, you can freely explore these theories.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        <h1 id="x"> Section x </h1>

        > a concise section overview

        <h2 id="x-x" align="center">Subsection X</h2>

        > main body

        **Recaps**

        - recap 1
        - recap 2
        - …
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        <h1 id="summary">Summary</h1>

        A summary of the notebook

        **Recaps**

        - recap 1
        - recap 2
        - …
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        <h1 id="experiment">Experiment</h1>

        > write experiment here
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        <h1 id="refs">References</h1>

        1. [Reference 1](https://example.com)
        2. [Reference 2](https://example.com)
        """
    )
    return


@app.cell
def __():
    import marimo as mo

    mo.md(
        rf"""
        <h1 id="src">Source Code</h1>

        > write source code below
        """
    )
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        rf"""
        Implement your code in following order:

        1. imports
        2. define variables
        3. implement what main() does
        4. define functions
        5. define classes
        6. define marimo user interface
        7. implement sidebar (important for the navigation)
        8. define constants
        """
    )
    return


@app.cell
def __(init_form):
    NOTEBOOK_TITLE = init_form.value["title"] if init_form.value else "Notebook Title"
    NOTEBOOK_PATH = init_form.value["path"] if init_form.value else ".."
    AUTHOR_NAME = (
        init_form.value["author_name"] if init_form.value else "marimo-tutorials-team"
    )
    AUTHOR_CONTACT = (
        init_form.value["author_contact"]
        if init_form.value
        else "marimtutorial@example.com"
    )
    return AUTHOR_CONTACT, AUTHOR_NAME, NOTEBOOK_PATH, NOTEBOOK_TITLE


@app.cell
def __(NOTEBOOK_TITLE, mo):
    # implement sidebar in source code
    sidebar_content = {
        "#home": f"{mo.icon('lucide:home')} Home",
        "#intro": "Introduction",
        "Section 1": {
            "#x": "Overview",
            "#x-x": "Subsection x",
        },  ## more sections below
        "#summary": "Summary",
        "#experiment": "Experiment",
        "#refs": "References",
        "#src": "Source Code",
        "Links": {
            "https://github.com/Haleshot/marimo-tutorials": f"{mo.icon('lucide:github')} GitHub",
        },
    }
    mo.sidebar(
        [
            mo.md(f"# {NOTEBOOK_TITLE}"),
            mo.nav_menu(sidebar_content, orientation="vertical"),
        ]
    )
    return (sidebar_content,)


@app.cell
def __(AUTHOR_CONTACT, AUTHOR_NAME, mo):
    mo.md(
        f"""
        **Authors**

        - [{AUTHOR_NAME}]({AUTHOR_CONTACT})
        """
    )
    return


if __name__ == "__main__":
    app.run()
