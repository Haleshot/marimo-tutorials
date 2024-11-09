# /// script
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

__generated_with = "0.9.10"
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

        For example:

        ```python
        header_widget = HeaderWidget(
            result={
                "Title": "Comprehensive E-Commerce Customer Behavior Analysis",
                "Author": '<a href="https://github.com/Haleshot/marimo-tutorials">Dr. Jane Smith, PhD</a>',
                "Version": "1.2.3",
                "Description": "This advanced notebook presents a multi-faceted analysis of <b>customer behavior patterns</b> across various e-commerce platforms. The primary goal is to derive actionable insights that can significantly enhance customer engagement, optimize conversion rates, and ultimately drive business growth in the competitive e-commerce landscape.",
                "Keywords": "E-Commerce Analytics, Customer Behavior Modeling, Predictive Analytics, Machine Learning, Natural Language Processing, Data Visualization, Time Series Analysis",
                "Data Sources": "1. Customer transaction logs (5 years, 10M+ records)<br>2. Website clickstream data (real-time, 1B+ events)<br>3. CRM records (customer demographics, purchase history)<br>4. Social media interactions (Twitter, Facebook, Instagram)<br>5. Customer support tickets and chat logs<br>6. Product catalog and inventory data",
                "Last Updated": "November 3, 2024",
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

        1. If a notebook contains any io operation, e.g., reading an external .csv file, you'd better use a `marimo.ui.form` for users to config the path of this .csv file.
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
        <h1 id="refs">References</h1>

        - [Reference 1](https://example.com)
        - [Reference 2](https://example.com)
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import anywidget
    import traitlets

    class HeaderWidget(anywidget.AnyWidget):
        _esm = """
        function escapeHTML(str) {
            return str.replace(/[&<>'"]/g, 
                tag => ({
                    '&': '&amp;',
                    '<': '&lt;',
                    '>': '&gt;',
                    "'": '&#39;',
                    '"': '&quot;'
                }[tag] || tag)
            );
        }

        function stripHTML(html) {
            const tmp = document.createElement("DIV");
            tmp.innerHTML = html;
            return tmp.textContent || tmp.innerText || "";
        }

        function renderValue(value) {
            if (typeof value !== 'string') {
                return escapeHTML(String(value));
            }

            const isHTML = /<[a-z][\s\S]*>/i.test(value);
            const strippedValue = isHTML ? stripHTML(value) : value;

            if (strippedValue.length > 100) {
                if (isHTML) {
                    return `
                        <div class="preview">${value.substring(0, 100)}...</div>
                        <div class="full-text" style="display: none;">${value}</div>
                        <button class="toggle-button">Show More</button>
                    `;
                } else {
                    return `
                        <div class="preview">${escapeHTML(value.substring(0, 100))}...</div>
                        <div class="full-text" style="display: none;">${escapeHTML(value)}</div>
                        <button class="toggle-button">Show More</button>
                    `;
                }
            }

            return isHTML ? value : escapeHTML(value);
        }

        function render({ model, el }) {
            const result = model.get("result");
            const container = document.createElement("div");
            container.className = "header-container";

            container.innerHTML = `
                <img class="banner" src="https://raw.githubusercontent.com/Haleshot/marimo-tutorials/main/community-tutorials-banner.png" alt="Marimo Banner">
                <div class="form-container">
                    ${Object.entries(result).map(([key, value]) => `
                        <div class="form-row">
                            <label>${escapeHTML(key)}</label>
                            <div class="value-container">
                                ${renderValue(value)}
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;

            el.appendChild(container);

            container.querySelectorAll('.toggle-button').forEach(button => {
                button.addEventListener('click', () => {
                    const row = button.closest('.form-row');
                    const preview = row.querySelector('.preview');
                    const fullText = row.querySelector('.full-text');

                    if (fullText.style.display === "none") {
                        fullText.style.display = "block";
                        preview.style.display = "none";
                        button.textContent = "Show Less";
                    } else {
                        fullText.style.display = "none";
                        preview.style.display = "block";
                        button.textContent = "Show More";
                    }
                });
            });
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
            height: 200px;
            object-fit: cover;
            border-radius: 10px 10px 0 0;
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
        .value-container a {
                color: #0066cc;
                text-decoration: none;
                transition: color 0.2s ease;
        }
        .value-container a:hover {
            color: #003366;
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


@app.cell(hide_code=True)
def __(HeaderWidget):
    header_widget = HeaderWidget(
        result={
            "Title": "Comprehensive E-Commerce Customer Behavior Analysis",
            "Author": '<a href="https://github.com/Haleshot/marimo-tutorials">Dr. Jane Smith, PhD</a>',
            "Affiliation": '<a href="https://www.datascience.university.edu">University of Data Science</a>',
            "Version": "1.2.3",
            "Description": "This advanced notebook presents a multi-faceted analysis of <b>customer behavior patterns</b> across various e-commerce platforms. The primary goal is to derive actionable insights that can significantly enhance customer engagement, optimize conversion rates, and ultimately drive business growth in the competitive e-commerce landscape.",
            "Keywords": "E-Commerce Analytics, Customer Behavior Modeling, Predictive Analytics, Machine Learning, Natural Language Processing, Data Visualization, Time Series Analysis",
            "Data Sources": "1. Customer transaction logs (5 years, 10M+ records)<br>2. Website clickstream data (real-time, 1B+ events)<br>3. CRM records (customer demographics, purchase history)<br>4. Social media interactions (Twitter, Facebook, Instagram)<br>5. Customer support tickets and chat logs<br>6. Product catalog and inventory data",
            "Prerequisites": "Intermediate to advanced knowledge in statistics, machine learning, and Python programming. Familiarity with e-commerce concepts and business metrics is beneficial.",
            "Acknowledgements": "This work was supported by a grant from the National Science Foundation (NSF-1234567). Special thanks to the E-Commerce Research Consortium for providing anonymized datasets.",
            "License": '<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>',
            "Last Updated": "November 3, 2024",
        }
    )
    return (header_widget,)


@app.cell(hide_code=True)
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
