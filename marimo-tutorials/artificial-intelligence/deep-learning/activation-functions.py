# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anywidget==0.9.13",
#     "traitlets==5.14.3",
#     "numpy==2.1.2",
#     "matplotlib==3.9.2",
# ]
# ///
"""
This is the starting point for your notebook.
"""

import marimo

__generated_with = "0.9.14"
app = marimo.App(app_title="Activation-Functions")


@app.cell(hide_code=True)
def __(header_widget):
    header_widget
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        f"""
        ## Activation Functions
            [Activation functions](https://en.wikipedia.org/wiki/Activation_function) are crucial components in neural networks that introduce 
            non-linearity into the network's learning process. They determine whether a 
            neuron should be fired/activated or not by calculating the weighted sum and adding bias.
            Each function has its unique properties and use cases in different scenarios.
            Select an activation function from the dropdown below to visualize its behavior.
        """
    )
    return


@app.cell(hide_code=True)
def __(activation_functions, alpha_slider, mo, x_range):
    import numpy as np
    import matplotlib.pyplot as plt

    @mo.cache(pin_modules=True)  # if module versions change later on
    def plot_activation():
        # Get current values directly from UI elements (instead of hardocoding)
        func_name = activation_functions.value
        x_range_val = x_range.value
        alpha_val = alpha_slider.value
        
        x = np.linspace(x_range_val[0], x_range_val[1], 1000)
        y = np.zeros_like(x)
        formula = ""
        
        if func_name == "sigmoid":
            y = 1 / (1 + np.exp(-x))
            formula = r"$$\sigma(x) = \frac{1}{1 + e^{-x}}$$"
        elif func_name == "tanh":
            y = np.tanh(x)
            formula = r"$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$"
        elif func_name == "relu":
            y = np.maximum(0, x)
            formula = r"$$ReLU(x) = max(0, x)$$"
        elif func_name == "elu":
            y = np.where(x > 0, x, alpha_val * (np.exp(x) - 1))
            formula = f"$$ELU(x) = \\begin{{cases}} x & \\text{{if }} x > 0 \\\\ {alpha_val:.2f}(e^x - 1) & \\text{{otherwise}} \\end{{cases}}$$"
        elif func_name == "prelu":
            y = np.where(x > 0, x, alpha_val * x)
            formula = f"$$PReLU(x) = \\begin{{cases}} x & \\text{{if }} x > 0 \\\\ {alpha_val:.2f}x & \\text{{otherwise}} \\end{{cases}}$$"
        elif func_name == "leaky_relu":
            y = np.where(x > 0, x, alpha_val * x)
            formula = f"$$LeakyReLU(x) = \\begin{{cases}} x & \\text{{if }} x > 0 \\\\ {alpha_val:.2f}x & \\text{{otherwise}} \\end{{cases}}$$"
        elif func_name == "selu":
            scale = 1.0507009873554804
            y = scale * np.where(x > 0, x, alpha_val * (np.exp(x) - 1))
            formula = r"$$SELU(x) = \lambda \times ELU(x)$$"
        elif func_name == "softplus":
            y = np.log1p(np.exp(x))
            formula = r"$$Softplus(x) = log(1 + e^x)$$"
        elif func_name == "softsign":
            y = x / (1 + np.abs(x))
            formula = r"$$Softsign(x) = \frac{x}{1 + |x|}$$"
        elif func_name == "hard_sigmoid":
            y = np.clip(0.2 * x + 0.5, 0, 1)
            formula = r"$$HardSigmoid(x) = \begin{cases} 0 & \text{if } x \leq -2.5 \\ 0.2x + 0.5 & \text{if } -2.5 < x < 2.5 \\ 1 & \text{if } x \geq 2.5 \end{cases}$$"
        elif func_name == "swish":
            sigmoid = 1 / (1 + np.exp(-x))
            y = x * sigmoid
            formula = r"$$Swish(x) = x \times \sigma(x)$$"
        elif func_name == "mish":
            y = x * np.tanh(np.log1p(np.exp(x)))
            formula = r"$$Mish(x) = x \times tanh(ln(1 + e^x))$$"

        # Clear the previous plot
        plt.clf()
        plt.figure(figsize=(12, 7))
        plt.plot(x, y, linewidth=2, label=func_name)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'{func_name.upper()} Activation Function', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.legend(fontsize=10)
        
        # Set x-axis limits based on range slider value
        plt.xlim(x_range_val)
        
        # Automatically adjust y-axis limits with some padding (QoL/visual improvement)
        y_min, y_max = np.min(y), np.max(y)
        padding = (y_max - y_min) * 0.1
        plt.ylim(y_min - padding, y_max + padding)
        
        # Show alpha slider only for functions that use it
        alpha_functions = ["elu", "prelu", "leaky_relu", "selu"]
        show_alpha = func_name in alpha_functions
        
        controls = mo.hstack([
            mo.vstack([
                mo.md("### Select Activation Function"),
                activation_functions
            ]),
            mo.vstack([
                mo.md("### X-axis Range"),
                x_range
            ])
        ])
        
        if show_alpha:
            controls = mo.hstack([
                controls,
                mo.vstack([
                    mo.md("### Alpha Parameter"),
                    alpha_slider,
                    mo.md(f"Current α: {alpha_slider.value:.2f}")
                ])
            ])
        
        return mo.vstack([
            controls,
            mo.md(formula),
            mo.as_html(plt.gca())
        ])
    return np, plot_activation, plt


@app.cell(hide_code=True)
def __(plot_activation):

    # Display the plot
    plot_activation()
    return


@app.cell(hide_code=True)
def __(mo):
    activation_functions = mo.ui.dropdown(
        value="Sigmoid",
        options=dict(
            sorted(
                {
                    "Sigmoid": "sigmoid",
                    "Tanh": "tanh",
                    "ReLU": "relu",
                    "ELU": "elu",
                    "PReLU": "prelu",
                    "Leaky ReLU": "leaky_relu",
                    "SELU": "selu",
                    "Softplus": "softplus",
                    "Softsign": "softsign",
                    "Hard Sigmoid": "hard_sigmoid",
                    "Swish": "swish",
                    "Mish": "mish"
                }.items()
            )
        ),
    )
    return (activation_functions,)


@app.cell(hide_code=True)
def __(mo):
    # Range slider https://docs.marimo.io/api/inputs/range_slider.html#range-slider 
    # for x-axis control
    x_range = mo.ui.range_slider(
        start=-10,
        stop=10,
        step=0.5,
        value=[-10, 10],
        label="X-axis Range",
        full_width=True
    )

    # Slider https://docs.marimo.io/api/inputs/slider.html#slider for alpha parameter
    alpha_slider = mo.ui.slider(
        start=0.01,
        stop=2.0,
        step=0.01,
        value=0.01,
        label="Alpha Parameter"
    )
    return alpha_slider, x_range


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        <h2 id="refs">References</h2>

        - [Daniel Lee](https://www.linkedin.com/posts/danleedata_which-activation-function-do-you-use-often-activity-7234182901213335556-J_6o?utm_source=share&utm_medium=member_desktop)
        - [Mish](https://github.com/digantamisra98/Mish)
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Appendix
        The cells below consist of assisting variables/functions responsible for implementing the above cells (click on show code).
        """
    )
    return


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

                // Get the actual text content, whether it's a direct string or in an object
                let textContent = typeof value === 'object' ? value.text : value;

                // Check if content is long enough to warrant a show more button
                if (textContent && textContent.length > 100) {
                    const preview = document.createElement("div");
                    preview.className = "preview";
                    preview.innerHTML = textContent.substring(0, 100) + "...";

                    const fullText = document.createElement("div");
                    fullText.className = "full-text";
                    fullText.innerHTML = textContent;

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
                    valueContainer.innerHTML = textContent;
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

        .value-container a {
            color: #0066cc;
            text-decoration: none;
            transition: color 0.2s ease;
        }

        .value-container a:hover {
            color: #003366;
            text-decoration: underline;
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
def __():
    from datetime import datetime


    def get_current_date() -> str:
        """Returns the current date in 'YYYY-MM-DD' format."""
        return datetime.now().strftime("%Y-%m-%d")
    return datetime, get_current_date


@app.cell(hide_code=True)
def __(HeaderWidget, get_current_date):
    header_widget = HeaderWidget(
        result={
            "Title": "Neural Network Activation Functions",
            "Author": {"text": '<a href="https://github.com/Haleshot">Srihari Thyagarajan | Haleshot</a>'},
            "Date": get_current_date(),
            "Version": "1.0",
            "Description": {
                "text": """This notebook provides an interactive visualization of various activation functions 
    commonly used in neural networks. Features marimo's efficient caching system 
    (<a href="https://docs.marimo.io/api/caching.html#caching">mo.cache</a>) and interactive UI elements 
    (<a href="https://docs.marimo.io/api/inputs/dropdown.html#dropdown">mo.ui</a>). Explore mathematical formulas 
    and graphical representations of activation functions including Sigmoid, ReLU, Tanh, ELU, 
    and more."""
            },
            "Keywords": "Neural Networks, Activation Functions, Deep Learning, Interactive Visualization, Mathematical Functions",
            "Domain": "Deep Learning, Machine Learning, Neural Network Architecture",
            "Tools Used": {
                "text": """<a href="https://docs.python.org/3/">Python</a>, <a href="https://matplotlib.org/stable/index.html"> Matplotlib</a>, <a href="https://numpy.org/doc/"> NumPy</a>, <a href="https://docs.marimo.io">Marimo</a> 
                (caching, UI components)"""
            },
            "Functions Covered": "Sigmoid, Tanh, ReLU, ELU, PReLU, Leaky ReLU, SELU, Softplus, Softsign, Hard Sigmoid, Swish, Mish"
        }
    )
    return (header_widget,)


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
