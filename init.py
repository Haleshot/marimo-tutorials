# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///
"""
This is the starting point for your notebook.
"""

import marimo

__generated_with = "0.8.19"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(HEADER_BODY, HEADER_HTML, mo):
    mo.Html(HEADER_HTML + HEADER_BODY)
    return


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
                ├── Custom Input (optional but suggested)
                ├── Notebook Title
                ├── TLDR
                ├── Introduction
                ├── Main Body
                ├── Summary
                ├── Experiments (optional)
                ├── Exercises (optional)
                ├── References (optional)
                ├── Appendix (optional)
                ├── Source Code
                └── Acknowledgments (optional)
                ```
                """
                    ),
                    "Inside Notebook": mo.md(
                        """
                - Remember to write Navigation
                - Remember to add interactivity (`marimo.ui`) and visualizations as you can but avoid repetition
                - You can quote from time to time
                - You can write some short recaps from time to time
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
            mo.md(
                rf"> View your starting notebook below and make changes if needed:"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        **Some tips from our experience with marimo**

        1. If a notebook contains any side effect, e.g., reading an external .csv file, you'd better use a `marimo.ui.form` for users to config the path of this .csv file.
        2. You can create more ui components and appealing contents with pure html, [anywidget](https://github.com/manzt/anywidget) and more. But when you are doing this, remember to check its appearance under both light and dark themes, and different widths.
        3. Albeit you can create local variables in a cell with a prefix "_", we recommend you do this as little as possible because the `Explore variables` panel will neglect  these variables, making debug these variables hard.
        4. If you wan't your notebook to run properly in our cloud, please check whether the dependencies are supported by wasm. Some popular libraries like `polars` and `openai`, for example, are not supported.
        5. Attach as few assets as possible, we want to keep our repo lightweight.
        6. Functional thinking are preferred in marimo since instances are immutable.
        """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(NOTEBOOK_TITLE, mo):
    mo.md(
        f"""
        <h1 id="home">{NOTEBOOK_TITLE}</h1>
        ---

        **TLDR**

        > write TLDR here

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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        <h1 id="refs">References</h1>

        1. [Reference 1](https://example.com)
        2. [Reference 2](https://example.com)
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo

    mo.md(
        rf"""
        <h1 id="src">Source Code</h1>

        > write source code below
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def __():
    HEADER_HTML = r"""
    <head>
      <style>
        * {
          margin: 0;
          padding: 0;
        }

        body {
          display: flex;
          justify-content: center;
          align-items: center;
        }

        .header-container {
          width: 100%;
          border: 1px;
          border-radius: 12px;
          overflow: hidden;
        }

        .banner {
          height: 200px;
          background-image: url('https://i.ibb.co/SVcC6bb/final.png');
          background-size: cover;
          background-position: center;
        }

        .header-content {
          padding: 20px;
        }

        .title-section {
          margin-bottom: 16px;
        }

        .title-section h1 {
          font-size: 24px;
          margin-bottom: 8px;
          letter-spacing: 1px;
        }

        .meta-info {
          font-size: 14px;
          display: flex;
          flex-wrap: wrap;
        }

        .meta-info span {
          margin-right: 16px;
          margin-bottom: 5px;
        }

        .tags, .contact {
          margin-top: 12px;
          font-size: 13px;
        }
      </style>
    </head>
    """
    return (HEADER_HTML,)


@app.cell(hide_code=True)
def __(AUTHOR_CONTACT, AUTHOR_NAME, NOTEBOOK_TAGS, NOTEBOOK_TITLE):
    HEADER_BODY = f"""<body>
      <div class="header-container">
        <div class="banner"></div>
        <div class="header-content">     
          <div class="title-section">
            <h1>{NOTEBOOK_TITLE}</h1>
          </div>
          <div class="meta-info">
            <span>Author: {AUTHOR_NAME}</span>
          </div>
          <div class="tags">
            Tag: <span>{NOTEBOOK_TAGS}</span>
          </div>
          <div class="contact">
            Contact: <span>{AUTHOR_CONTACT}</span>
          </div>
        </div>
      </div>
    </body>
    </html>"""
    return (HEADER_BODY,)


@app.cell(hide_code=True)
def __(mo):
    # Initial Notebook Options
    init_form = (
        mo.md(
            r"""
            **Define Your Initial Notebook Options Here:**

            {title}

            {path}

            {tags}

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
            tags=mo.ui.text(
                placeholder="#deep-learning #transformers",
                label="Tags of your notebook: ",
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
def __(init_form):
    NOTEBOOK_TITLE = (
        init_form.value["title"] if init_form.value else "Notebook Title"
    )
    NOTEBOOK_PATH = init_form.value["path"] if init_form.value else ".."
    NOTEBOOK_TAGS = (
        init_form.value["tags"]
        if init_form.value
        else "#deep-learning #transformers"
    )
    AUTHOR_NAME = (
        init_form.value["author_name"]
        if init_form.value
        else "marimo-tutorials-team"
    )
    AUTHOR_CONTACT = (
        init_form.value["author_contact"]
        if init_form.value
        else "marimtutorial@example.com"
    )
    return (
        AUTHOR_CONTACT,
        AUTHOR_NAME,
        NOTEBOOK_PATH,
        NOTEBOOK_TAGS,
        NOTEBOOK_TITLE,
    )


if __name__ == "__main__":
    app.run()
