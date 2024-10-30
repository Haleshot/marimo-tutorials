# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget==0.9.13",
#     "traitlets==5.14.3",
# ]
# ///
"""
This is the starting point for your notebook.
"""

import marimo

__generated_with = "0.8.19"
app = marimo.App()


@app.cell(hide_code=True)
def __(header_widget):
    header_widget
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        First of all, create a header for your notebook.

        We've designed a `HeaderWidget` for you to display important information.

        To use `HeaderWidget`, you need to create an instance of it. You can pass a dictionary containing key-value pairs that represent the information you want to display in the header. 

        Here's the code for the above example:

        ```python
        HeaderWidget(
            result={
                "Title": "My Comprehensive Data Analysis Notebook",
                "Author": "Jane Smith",
                "Date": "2024-09-25",
                "Version": "2.1",
                "Description": "This notebook contains an in-depth analysis of customer behavior patterns across multiple e-commerce platforms. It includes data preprocessing, exploratory data analysis, statistical modeling, and machine learning techniques to derive actionable insights for improving customer engagement and conversion rates.",
                "Keywords": "data analysis, e-commerce, customer behavior, machine learning",
                "Data Sources": "Customer transaction logs, website clickstream data, CRM records, social media interactions",
                "Tools Used": "Python, Pandas, Scikit-learn, Matplotlib, Seaborn, TensorFlow",
            }
        )
        ```
        """
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
        4. If you want your notebook to run properly in our cloud, please check whether the dependencies are supported by wasm. Some popular libraries like `polars` and `openai`, for example, are not supported.
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
                label="Path of your data: ",
                full_width=True,
            ),  ## add more rows below
        )
        .form(bordered=True, label="Custom Constants")
    )
    custom_form
    return (custom_form,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        You can access the value of the form above with:

        ```python
        custom_form.value['image']
        ```

        after the submission.
        """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        <h1 id="home">NOTEBOOK_TITLE</h1>
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
        r"""
        <h1 id="x"> Section x </h1>

        > a concise section overview

        <h2 id="x-x" align="center">Subsection X</h2>

        > main body
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
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

        > list your references

        1. [Reference 1](https://example.com)
        2. [Reference 2](https://example.com)
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo

    mo.md(
        r"""
        <h1 id="src">Source Code</h1>

        > write source code below
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def __():
    import anywidget
    import traitlets

    class HeaderWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const result = model.get("result");

            const container = document.createElement("div");
            container.className = "header-container";

            const banner = document.createElement("img");
            banner.className = "banner";
            banner.src = "https://i.ibb.co/SVcC6bb/final.png";
            banner.style.width = "100%";
            banner.style.height = "200px";
            banner.style.objectFit = "cover";
            banner.style.borderRadius = "10px 10px 0 0";
            banner.alt = "Marimo Banner";

            const form = document.createElement("div");
            form.className = "form-container";

            for (const [key, value] of Object.entries(result)) {
                const row = document.createElement("div");
                row.className = "form-row";

                const label = document.createElement("label");
                label.textContent = key;

                const valueContainer = document.createElement("div");
                valueContainer.className = "value-container";

                if (value.length > 100) {
                    const preview = document.createElement("div");
                    preview.className = "preview";
                    preview.textContent = value.substring(0, 100) + "...";

                    const fullText = document.createElement("div");
                    fullText.className = "full-text";
                    fullText.textContent = value;

                    const toggleButton = document.createElement("button");
                    toggleButton.className = "toggle-button";
                    toggleButton.textContent = "Show More";
                    toggleButton.onclick = () => {
                        if (fullText.style.display === "none") {
                            fullText.style.display = "block";
                            preview.style.display = "none";
                            toggleButton.textContent = "Show Less";
                        } else {
                            fullText.style.display = "none";
                            preview.style.display = "block";
                            toggleButton.textContent = "Show More";
                        }
                    };

                    valueContainer.appendChild(preview);
                    valueContainer.appendChild(fullText);
                    valueContainer.appendChild(toggleButton);

                    fullText.style.display = "none";
                } else {
                    valueContainer.textContent = value;
                }

                row.appendChild(label);
                row.appendChild(valueContainer);
                form.appendChild(row);
            }

            container.appendChild(banner);
            container.appendChild(form);
            el.appendChild(container);
        }
        export default { render };
        """

        _css = """
        .header-container {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            max-width: 100%;
            margin: 0 auto;
            overflow: hidden;
        }

        .banner {
            width: 100%;
            height: auto;
            display: block;
        }

        .form-container {
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            font-weight: 300;
            box-shadow: 0 -10px 20px rgba(0,0,0,0.1);
        }

        .form-row {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
            font-weight: 500;
        }

        .value-container {
            font-size: 1em;
            line-height: 1.5;
        }

        .preview, .full-text {
            margin-bottom: 10px;
        }

        .toggle-button {
            border: none;
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .toggle-button:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        @media (max-width: 600px) {
            .form-container {
                grid-template-columns: 1fr;
            }
        }
        """

        result = traitlets.Dict({}).tag(sync=True)

    return HeaderWidget, anywidget, traitlets


@app.cell
def __(HeaderWidget):
    header_widget = HeaderWidget(
        result={
            "Title": "My Comprehensive Data Analysis Notebook",
            "Author": "Jane Smith",
            "Date": "2024-09-25",
            "Version": "2.1",
            "Description": "This notebook contains an in-depth analysis of customer behavior patterns across multiple e-commerce platforms. It includes data preprocessing, exploratory data analysis, statistical modeling, and machine learning techniques to derive actionable insights for improving customer engagement and conversion rates.",
            "Keywords": "data analysis, e-commerce, customer behavior, machine learning",
            "Data Sources": "Customer transaction logs, website clickstream data, CRM records, social media interactions",
            "Tools Used": "Python, Pandas, Scikit-learn, Matplotlib, Seaborn, TensorFlow",
        }
    )
    return (header_widget,)


if __name__ == "__main__":
    app.run()
