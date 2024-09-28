import marimo

__generated_with = "0.8.21"
app = marimo.App(width="medium", app_title="Non-Adaptive Image Scaling")


@app.cell
def __(header_widget):
    header_widget
    return


@app.cell(hide_code=True)
def __(HeaderWidget):
    header_widget = HeaderWidget(
        result={
            "Title": "Non-Adaptive Image Scaling Algorithms",
            "Author": "Eugene",
            "Contact": "eugeneheiner14@gmail.com",
            "Date": "2024-06-14",
            "Version": "0.2",
            "Keywords": "Image Scaling, Interpolation, Convolution, Lanczos resampling",
            "Tools Used": "Plotly, Numpy, Pillow",
        }
    )
    return (header_widget,)


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


@app.cell(hide_code=True)
def __(mo):
    # Custom Constants
    form = (
        mo.md(
            r"""
        **Customize your constants here:**

        {image}

        """
        )
        .batch(
            image=mo.ui.text(
                value="../assets/image-processing/lena_gray.gif",
                label="The path of your sample image (for scaling): ",
                full_width=True,
            ),
        )
        .form(bordered=True, label="Custom Constants")
    )
    form
    return (form,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h1 id="home">Non-Adaptive Image Scaling Algorithms</h1>
        ---

        **Abstract**

        This notebook provides an introductory exploration into non-adaptive image scaling algorithms, focusing on [NN Interpolation](#nn), [Bilinear Interpolation](#bilinear), and [Bicubic Interpolation](#bicubic). Each algorithm is elucidated through mathematical formulations and accompanied by visualizations for clarity. Transitioning to their [convolutional forms](#convolutions), the notebook delves into advanced concepts including [Lanczos resampling](#lanczos), shedding light on their significance in image processing. Through this systematic analysis, readers gain foundational insights into the theoretical underpinnings and practical applications of these algorithms in image scaling tasks.

        <h1 id="intro">Introduction</h1>

        Image scaling, an essential task in digital image processing, plays a pivotal role in various applications ranging from multimedia content delivery to medical imaging. The process of image scaling involves altering the size of an image while preserving its visual quality and integrity. Non-adaptive image scaling algorithms represent a fundamental category within this domain, offering methods to resize images without adapting to the content characteristics.

        This notebook serves as an introductory guide to non-adaptive image scaling algorithms, presenting a structured exploration of their theoretical foundations, mathematical formulations, and practical applications. 

        The notebook is organized as follows:

        1. **Fundamentals of Image Scaling Algorithms:**
            - A comprehensive overview of NN Interpolation, Bilinear Interpolation, and Bicubic Interpolation, elucidating their mathematical formulations and principles.
            - Visualizations illustrating the effects of these algorithms on image quality and resolution.

        2. **Convolutional Transformations:**

            - Transitioning from traditional formulations to convolutional forms, highlighting the convolutional operations underlying these algorithms.
            - Exploring the role of Fourier Spectrum analysis in understanding the frequency domain characteristics of scaled images (TODO).

        3. **Advanced Techniques and Resampling Methods:**

            - Introduction to advanced techniques such as Lanczos resampling, offering insights into their benefits and limitations.
            - Comparative analysis of different resampling methods to facilitate informed decision-making in practical applications.

        By systematically dissecting these algorithms and methodologies, this notebook aims to equip readers with a comprehensive understanding of non-adaptive image scaling techniques, empowering them to navigate the complexities of image processing tasks effectively. Through theoretical discussions, practical demonstrations, and visual aids, readers will gain valuable insights into the principles governing image scaling algorithms and their implications for real-world applications.

        > In the `Experiment` section, you can freely explore these algorithms, as well as controling the resizing dimension.
        """
    )
    return


@app.cell(hide_code=True)
def __(lena, lena_downscaled, mo, show_images):
    mo.md(
        f"""
        <h1 id="fundamentals">Fundamentals of Image Scaling Algorithms</h1>
        ---

        For the purpose of convenience and simplicity, this notebook focuses solely on grayscale images. The Lena image, a standard test image widely utilized in image processing research, will serve as our primary image dataset.

        To facilitate the analysis, we will first downscale the original 512x512 Lena image to a smaller 64x64 resolution. Subsequently, we will explore the effects of different interpolation techniques by upscaling the downscaled image back to its original size of 512x512.

        {show_images([lena, lena_downscaled], ["Lena 512*512", "Lena 64*64"])}
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Let $P(x,y)$ denote a pixel in the input image of size $M \times N$, and let $U(i,j)$ represent a pixel in the scaled or resized output image of size $M' \times N'$. The transformation $T$ is utilized for scaling, where each pixel in the output image is computed by applying the transformation $T$ on the input image pixels.

        <h2 id="nn" align="center">Nearest Neighbor (NN) Interpolation</h2>

        Nearest Neighbor (NN) Interpolation is a simple technique used in image scaling, where the value of a new pixel is determined by selecting the value of the nearest pixel in the original image. This method is straightforward to implement but may result in blocky or pixelated images, especially when scaling up.

        **Mathematical Formulation:**

        The coordinates \( (x', y') \) of a pixel in the output image are related to the coordinates \( (x, y) \) of the corresponding pixel in the input image by the formula:

        \[
        x = \text{round}(x' \times \frac{M}{M'})
        \]

        \[
        y = \text{round}(y' \times \frac{N}{N'})
        \]

        The value of the pixel at \( (x', y') \) in the output image is then set to the value of the nearest pixel at \( (x, y) \) in the input image.
        """
    )
    return


@app.cell(hide_code=True)
def __(NN_SRC, mo):
    mo.vstack([mo.image(src=NN_SRC, height=400, width=700)], align="center")
    return


@app.cell(hide_code=True)
def __(nn_result):
    nn_result
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h2 id="bilinear" align="center">Bilinear Interpolation</h2>

        Bilinear interpolation is a method used to resample images by computing new pixel values based on the weighted average of neighboring pixels. It offers smoother results compared to nearest neighbor interpolation but is computationally more intensive. Bilinear interpolation considers the values of four nearest pixels to estimate the intensity of a new pixel.

        **Mathematical Formulation:**

        Let \( P(x, y) \) denote the intensity of the pixel at coordinates \( (x, y) \) in the input image. Bilinear interpolation estimates the intensity \( U(i, j) \) of a pixel at coordinates \( (i, j) \) in the output image as follows:

        \[
        U(i, j) = (1 - \alpha) \cdot (1 - \beta) \cdot P(x_{\text{1}}, y_{\text{1}}) + \alpha \cdot (1 - \beta) \cdot P(x_{\text{2}}, y_{\text{1}}) + (1 - \alpha) \cdot \beta \cdot P(x_{\text{1}}, y_{\text{2}}) + \alpha \cdot \beta \cdot P(x_{\text{2}}, y_{\text{2}})
        \]

        where:

        - \( x_{\text{1}} \) and \( y_{\text{1}} \) are the integer parts of \( x \) and \( y \) respectively.
        - \( \alpha = x - x_{\text{1}} \) and \( \beta = y - y_{\text{1}} \) are the fractional parts.
        - \( P(x_{\text{1}}, y_{\text{1}}) \), \( P(x_{\text{2}}, y_{\text{1}}) \), \( P(x_{\text{1}}, y_{\text{2}}) \), and \( P(x_{\text{2}}, y_{\text{2}}) \) represent the intensities of the four nearest pixels in the input image.

        Bilinear interpolation can be viewed as performing linear interpolation in one direction followed by linear interpolation of the interpolated values in the other direction.
        """
    )
    return


@app.cell(hide_code=True)
def __(BILINEAR_SRC, mo):
    mo.vstack(
        [mo.image(BILINEAR_SRC, height=480, width=600, rounded=True)],
        align="center",
    )
    return


@app.cell(hide_code=True)
def __(bilinear_result):
    bilinear_result
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h2 id="bicubic" align="center">Bicubic Interpolation</h2>

        Bicubic interpolation is a method used to resample images by estimating pixel values based on the intensity of sixteen neighboring pixels in a \(4 \times 4\) local neighborhood. It offers higher quality results compared to bilinear interpolation but is computationally more demanding. Bicubic interpolation fits a cubic polynomial to the intensity values in the local neighborhood to estimate the intensity of a new pixel.

        Like Bilinear Interpolation, bicubic interpolation can be viewed as fitting cubic interpolations on four rows of pixels and then applying cubic interpolation on the resulting interpolated values. Here's how this process can be illustrated:

        1. **Fitting Cubic Interpolations on 4 Rows:**
            - Consider a \(4 \times 4\) local neighborhood of pixels.
            - Apply cubic interpolation along each row independently to estimate intermediate values between the pixel intensities.
            - This results in four sets of interpolated values, one for each row.

        2. **Applying Cubic Interpolation on the Interpolated 4 Values:**
            - Apply cubic interpolation vertically on the four sets of interpolated values obtained from the previous step.
            - This results in the final interpolated pixel value for the target pixel location.

        > **The details of cubic interpolation will be present in another notebook:**

        > Spline

        > So the implementation of Bicubic Interpolation is not given in this notebook.
        """
    )
    return


@app.cell(hide_code=True)
def __(BICUBIC_SRC, mo):
    mo.vstack(
        [mo.image(BICUBIC_SRC, height=300, width=600, rounded=True)],
        align="center",
    )
    return


@app.cell(hide_code=True)
def __(bicubic_result):
    bicubic_result
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h1 id="convolutions">Convolutional Transformations</h1>

        For conveniece, we’ll only be concerned with 1-dimensional interpolation from this point forward.

        <h2 id="interpolation" align="center">Interpolation and convolution</h2>

        It is mathematically and practically helpful to frame interpolations as convolutions.

        Let’s imagine that our signal $f(t) = \sin\pi \frac t 5 + \frac 1 2\cos\pi t + \frac 1 3 \sin\pi 2t$ is sampled at regular intervals so that they are $1/\xi$ apart from each other. We’ll say that $\xi$ is the frequency at which the samples are taken, the higher the frequency, the denser the samples.

        For instance, using our previous function and $\xi=2$:
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"> **You can define a custom signal function here.**")
    return


@app.cell
def __(np):
    signal_func = lambda x: np.sin(1 / (abs(x) + 0.0001))
    signal_func = None  # replace `None` with a function, e.g. the func above
    return (signal_func,)


@app.cell
def __(f, generate_samples, visualize_function_with_samples):
    visualize_function_with_samples(f, generate_samples(f, [-2, 2], 2))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        We'll refer to a single sample with $f_k$, where $f_k$ is taken at time $k/\xi$. Given some interpolation function $g$, we can convolve it with our discrete samples to get an interpolated signal $\bar{f}(t)$.

        The interpolated signal $\bar{f}(t)$ is defined as the sum over all possible indices k (from negative infinity to positive infinity) of the product of:

        * The interpolation function $g(t - k/\xi)$, where $k/\xi$ represents the shifted time of the k-th sample.
        * The k-th sample value $f_k$.

        In mathematical notation:

        $$\bar{f}(t) = \sum_{k=-\infty}^{\infty} g\left(t - \frac{k}{\xi}\right) f_k$$

        Within this framework, different interpolation algorithms can be seen as interpolating with a corresponding filter / kernel.

        Linear interpolation can be seen as interpolating with the following function:

        $$
        g(x) = \begin{cases}
        1 + x & \text{for} \; -1 < x \le 0 \\
        1 - x & \text{for} \: 0 < |x| < 1 \\
        0 & \text{otherwise}
        \end{cases}
        $$

        and cubic interpolation can be seen as interpolating with the following function:

        $$
        g(x) = \begin{cases}
        (a+2)|x|^3 - (a+3)|x|^2 + 1 & \text{for} |x| \leq 1 \\
        a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{for} 1 < |x| < 2 \\
        0 & \text{otherwise}
        \end{cases}
        $$

        where 𝑎 is usually set to −0.5 or −0.75

        > You can find the proof [here](https://www.semanticscholar.org/paper/Cubic-convolution-interpolation-for-digital-image-Keys/de5ca64429d18710232ecc0da54998e1a401fcbe)

        We now turn our attention to the sinc function, which is the foundation for Lanczos interpolation and many other interpolation filters.

        The sinc function, denoted by sinc(t), is defined as:

        $$
        \text{sinc}(t) = \frac{\sin(t)}{t}
        $$

        However, the sinc function has a discontinuity at t = 0, where the denominator becomes zero. To address this, we define sinc(0) = 1 artificially. This fills the "hole" at t = 0 and makes the function continuous.

        Notice that in the implementation of `sinc`, we are actually using the following formulae to perform a normalization:

        $$
        \text{sinc}(t) = \frac{\sin(\pi*t)}{\pi*t}
        $$

        Here are graphical representations for the above kernels.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        > **Again, you can add your custom kernel function here (name your function is recommended).**

        > A sample custom kernel function is given below, this is actually osculatory rational interpolation, and we add a `condition_threshold` to show how you can make this custom kernel function dynamic (e.g., you can use a slider ui to control its value).
        """
    )
    return


@app.cell
def __(np):
    def kernel_func(condition_threshold: float = 1):
        """
        Performs oscillatory resampling.

        Args:
          condition_threshold: A parameter used in the different parts of the function.

        Returns:
          The custom kernel function given condition threshold.
        """

        def name_decorator(func):
            def inner(x):
                return func(x)

            inner.__name__ = (
                f"custom_{condition_threshold}"  ## custom the name of the func
            )
            return inner

        @name_decorator
        def _func(x):
            assert 0 < condition_threshold < 2
            abs_x = np.abs(x)
            abs_x_squared = abs_x**2

            # Condition 1: abs(x) <= condition_threshold
            condition_1 = abs_x <= condition_threshold
            resampled_1 = np.where(
                condition_1,
                (-0.168 * abs_x_squared - 0.9129 * abs_x + 1.0808)
                / (abs_x_squared - 0.8319 * abs_x + 1.0808),
                0,
            )  # Set to 0 for non-matching elements

            # Condition 2: condition_threshold < abs(x) <= 2
            condition_2 = (abs_x > condition_threshold) & (abs_x <= 2)
            resampled_2 = np.where(
                condition_2,
                (0.1953 * abs_x_squared - 0.5858 * abs_x + 0.3905)
                / (abs_x_squared - 2.4402 * abs_x + 1.7676),
                0,
            )  # Set to 0 for non-matching elements

            # Combine results based on conditions
            return resampled_1 + resampled_2

        return _func


    custom_func = kernel_func(1)
    return custom_func, kernel_func


@app.cell(hide_code=True)
def __(cubic, custom_func, sinc, triangle, visualize_kernels):
    visualize_kernels(
        [triangle, cubic(), cubic(-0.75), sinc, custom_func], limits=(-4, 4)
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"Let's check their interpolation effects for our signal function `f`:")
    return


@app.cell(hide_code=True)
def __(
    cubic,
    custom_func,
    f,
    mo,
    plot_interpolations,
    sample_freq,
    sinc,
    triangle,
):
    mo.vstack(
        [
            sample_freq,
            plot_interpolations(
                f,
                [-2, 2],
                sample_freq.value,
                [triangle, cubic(), cubic(-0.75), sinc, custom_func],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        <h2 id="problems" align="center">The Problem With Sinc</h2>

        The first problem is evident: to convolve with sinc, we need to consider all the samples we have every time. This is clearly impractical: we’d need to look at every pixel in the input image to generate each pixel in the output.

        The second problem is what happens when fourier spectrum is not of limited width. Let’s consider a simple stepping function, and corresponding interpolation results:
        """
    )
    return


@app.cell
def __(sgn, visualize_kernels):
    visualize_kernels([sgn], limits=(-1, 1))
    return


@app.cell(hide_code=True)
def __(
    cubic,
    custom_func,
    mo,
    plot_interpolations,
    sample_freq,
    sgn,
    sinc,
    triangle,
):
    mo.vstack(
        [
            sample_freq,
            plot_interpolations(
                sgn,
                [-2, 2],
                sample_freq.value,
                [triangle, cubic(), sinc, custom_func],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        The reconstructed signal repeatedly overshoots and undershoot our original function. These undesirable oscillations, known as Gibbs phenomenon, show up all the time in Fourier analysis when dealing with jump discontinuities and finite approximations. They are intimately related to sinc in a sense the Gibbs oscillations are all ghosts of sinc in one form or another.

        Lánczos interpolation will address both problems presented in this section.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        <h1 id="lanczos", align="center">Lánczos</h1>

        > Sinc, Chopped and Screwed

        Let’s first consider the problem of sinc extending into infinity, and therefore requiring to examine all samples. Our first attempt to solve this issue might be just to just set sinc to zero outside a certain window

        We can define a function that sets another function to zero outside a specific interval. Let's denote this function as $\langle f(t) \rangle_a$. Here's the mathematical definition:

        $$\langle f(t) \rangle_a = \begin{cases}
        f(t) & \text{if} -a < t < a \\
        0 & \text{otherwise}
        \end{cases}$$

        In simpler terms, $\langle f(t) \rangle_a$ takes a function $f(t)$ and sets its output to zero whenever the absolute value of the input time $t$ is greater than or equal to $a$. Inside the interval $-a < t < a$ (excluding the endpoints), the function retains its original values from $f(t)$.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo, plot_interpolations, sample_freq, sgn, sinc_a, sinc_a_slider):
    mo.vstack(
        [
            mo.hstack(
                [sample_freq, sinc_a_slider], align="center", justify="center"
            ),
            plot_interpolations(
                sgn,
                [-2, 2],
                sample_freq.value,
                [
                    sinc_a(a=1),
                    sinc_a(a=2),
                    sinc_a(a=4),
                    sinc_a(a=sinc_a_slider.value),
                ],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The issue with the function $\langle f(t) \rangle_a$ for interpolation is its abrupt cut-off at ±a. This sharp transition in the frequency domain (obtained through the Fourier Transform) creates ringing artifacts in the interpolated signal. These ringing artifacts appear as unwanted oscillations around the desired frequencies.

        Here's how Lanczos's trick addresses this issue:

        1. **Lánczos Window (sinc(πt/a)):** The sinc function (pronounced "sinc") has a characteristic shape with a central lobe and decaying side lobes. The Lánczos window uses the sinc function scaled by πt/a, denoted by sinc(πt/a). This function smoothly tapers the original function towards zero outside the interval [-a, a].

        2. **Multiplication with $\langle f(t) \rangle_a$:**  Lanczos's trick involves multiplying the original function truncated with $\langle f(t) \rangle_a$ by the Lánczos window sinc(πt/a). This multiplication in the time domain acts like a convolution in the frequency domain. The Lánczos window effectively smooths the spectrum of the truncated function, reducing the sharp transitions and consequently minimizing ringing artifacts in the interpolated signal.
        """
    )
    return


@app.cell(hide_code=True)
def __(lanczos, lanczos_a_slider, visualize_kernels):
    visualize_kernels(
        [lanczos(1), lanczos(2), lanczos(4), lanczos(lanczos_a_slider.value)],
        limits=(-4, 4),
    )
    return


@app.cell(hide_code=True)
def __(
    lanczos,
    lanczos_a_slider,
    mo,
    plot_interpolations,
    sample_freq,
    sgn,
):
    mo.vstack(
        [
            mo.hstack(
                [sample_freq, lanczos_a_slider], align="center", justify="center"
            ),
            plot_interpolations(
                sgn,
                [-2, 2],
                sample_freq.value,
                [
                    lanczos(a=1),
                    lanczos(a=2),
                    lanczos(a=4),
                    lanczos(a=lanczos_a_slider.value),
                ],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(rf"And let's go back to 2-d")
    return


@app.cell(hide_code=True)
def __(lanczos_result):
    lanczos_result
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        <h1 id="summary">Summary</h1>

        This notebook explored four different algorithms for image scaling: Nearest Neighbor (NN) Interpolation, Bilinear Interpolation, Bicubic Interpolation, and Lanczos Resampling. Each algorithm offers a unique approach to resizing images, each with its own advantages and disadvantages.

        **Nearest Neighbor (NN) Interpolation:**

        - **Pros:**

            - Simple and fast.
            - Minimal computational overhead.
            - Preserves sharp edges in the image.

        - **Cons:**

            - Can produce blocky artifacts, especially when upscaling.
            - Lacks smoothness in the resized image.

        **Bilinear Interpolation:**

        - **Pros:**

            - Smoother results compared to NN interpolation.
            - Preserves some detail in the image.
            - Relatively fast and computationally efficient.

        - **Cons:**

            - Can still produce some artifacts, particularly in regions with high contrast.
            - Not as accurate as higher-order interpolation methods.

        **Bicubic Interpolation:**

        - **Pros:**

            - Produces smoother and more visually appealing results compared to NN and bilinear interpolation.
            - Preserves more detail and reduces artifacts.
            - Allows for higher-quality resizing.

        - **Cons:**

            - More computationally intensive compared to NN and bilinear interpolation.
            - May introduce blurring or ringing artifacts in certain cases.

        **Lanczos Resampling:**

        - **Pros:**

            - Provides high-quality results with minimal aliasing artifacts.
            - Preserves sharpness and detail in the image.
            - Performs well for both upscaling and downscaling.

        - **Cons:**

            - More computationally demanding compared to other interpolation methods.
            - Requires careful parameter tuning for optimal results.
            - Can introduce ringing artifacts around edges if not applied correctly.

        In conclusion, the choice of interpolation algorithm depends on the specific requirements of the application, including the desired level of quality, computational resources, and trade-offs between speed and accuracy. NN interpolation is suitable for quick and simple resizing tasks, while bilinear and bicubic interpolation offer improved quality at the expense of some computational overhead. Lanczos resampling provides the highest quality results but requires more computational resources and parameter tuning.
        """
    )
    return


@app.cell(hide_code=True)
def __(downscaled_dim, exp_result, lena, mo, resizing_dim, show_images):
    custom_downscaled = lena.resize(downscaled_dim.value)
    mo.md(
        f"""
        <h1 id=\"exp\">Experiment</h1>

        {downscaled_dim}

        {resizing_dim}

        {show_images(
            [lena, custom_downscaled],
            [
            "Lena 512*512",
            f"Lena {downscaled_dim.value[0]}*{downscaled_dim.value[1]}",
            ],
        )}

        {exp_result}
        """
    )
    return (custom_downscaled,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        rf"""
        <h1 id="refs">References</h1>

        1. [Lánczos interpolation explained](https://mazzo.li/posts/lanczos.html)

        2. [A Comparative Analysis of Image Scaling Algorithms](https://www.semanticscholar.org/paper/A-Comparative-Analysis-of-Image-Scaling-Algorithms-Suresh-Singh/3756c78e89b651792da0937fa61d6df0a9e348a0)

        3. [Low-Cost Implementation of Bilinear and Bicubic Image Interpolation for Real-Time Image SuperResolution](https://www.semanticscholar.org/paper/Low-Cost-Implementation-of-Bilinear-and-Bicubic-for-Khaledyan-Amirany/238819cdc430f84d9008bde8df34db94453a45b8)

        4. [Comparison gallery of image scaling algorithms](https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms)

        5. [Cubic Convolution Interpolation for Digital Image Processing](https://www.semanticscholar.org/paper/Cubic-convolution-interpolation-for-digital-image-Keys/de5ca64429d18710232ecc0da54998e1a401fcbe)

        6. [Image scaling](https://en.wikipedia.org/wiki/Image_scaling)

        7. [Bicubic interpolation](https://en.wikipedia.org/wiki/Bicubic_interpolation)

        8. [Cubic interpolation](https://www.paulinternet.nl/?page=bicubic)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        <h1 id="code">Source Code</h1>

        > check source code below
        """
    )
    return


@app.cell
def __(np, signal_func):
    # signal function
    f = (
        (
            lambda x: np.sin(np.pi * x / 5)
            + 1 / 2 * np.cos(np.pi * x)
            + 1 / 3 * np.sin(np.pi * 2 * x)
        )
        if signal_func is None
        else signal_func  ## if custom signal func is given
    )
    return (f,)


@app.cell
def __(NDArray, np):
    def triangle(x: NDArray):
        """
        Defines a triangular function with a peak at x=0.

        Args:
            x: The input value.

        Returns:
            The value of the triangular function at x.
        """
        # Combine conditions and calculations using np.piecewise
        return np.piecewise(
            x,
            [
                ((x <= 0) & (x > -1)),
                ((0 < x) & (x < 1)),
            ],
            [
                lambda x: 1 + x,
                lambda x: 1 - x,
                lambda x: 0,
            ],
        )  # Add default case for x outside range
    return (triangle,)


@app.cell
def __(NDArray, np):
    def cubic(a: float = -0.5):
        """
        Defines a piecewise cubic function g(x) based on the input parameter a.

        Args:
          a: A parameter used in the different parts of the function.

        Returns:
          The cubic kernel function given a.
        """

        def decorator_sinc(func: callable):
            def inner(x):
                return func(x)

            inner.__name__ = f"cubic_{a=}"
            return inner

        # Use np.piecewise for efficient handling of conditions
        @decorator_sinc
        def _cubic(x: NDArray):
            return np.piecewise(
                x,
                [(abs(x) <= 1), ((abs(x) > 1) & (abs(x) < 2))],
                [
                    lambda x: (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1,
                    lambda x: a * abs(x) ** 3
                    - 5 * a * abs(x) ** 2
                    + 8 * a * abs(x)
                    - 4 * a,
                    lambda x: 0,
                ],
            )

        return _cubic
    return (cubic,)


@app.cell
def __(NDArray, np):
    def sinc(x: NDArray):
        """
        Calculates the normalized sinc function, which is the mathematical function
        defined as sin(pi * x) / (pi * x) for x not equal to zero, and one for
        x equal to zero. This function avoids division by zero.

        Args:
            x (NDArray): A NumPy array of input values.

        Returns:
            NDArray: A NumPy array containing the normalized sinc values.
        """
        return np.piecewise(
            x,
            [x != 0],
            [
                lambda x: np.sin(np.pi * x) / (np.pi * x),
                lambda x: 1,
            ],
        )
    return (sinc,)


@app.cell
def __(NDArray, np):
    def sinc_a(a: float = 3):
        """
        Defines a piecewise sinc function g(x) based on the input parameter a.

        Args:
          a: A parameter used in the different parts of the function.

        Returns:
          The cubic kernel function given a.
        """

        def decorator_sinc(func: callable):
            def inner(x):
                return func(x)

            inner.__name__ = f"sinc_{a=}"
            return inner

        @decorator_sinc
        def _sinc(x: NDArray):
            return np.piecewise(
                x,
                [(x != 0) & (abs(x) < a), x == 0],
                [
                    lambda x: np.sin(np.pi * x) / (np.pi * x),
                    lambda x: 1,
                    lambda x: 0,
                ],
            )

        return _sinc
    return (sinc_a,)


@app.cell
def __(NDArray, np):
    def lanczos(a: float = 3):
        """
        Defines a piecewise lanczos function g(x) based on the input parameter a.

        Args:
          a: A parameter used in the different parts of the function.

        Returns:
          The cubic kernel function given a.
        """

        def decorator_sinc(func: callable):
            def inner(x):
                return func(x)

            inner.__name__ = f"lanczos_{a=}"
            return inner

        @decorator_sinc
        def _sinc(x: NDArray):
            return np.piecewise(
                x,
                [(x != 0) & (abs(x) < a), x == 0],
                [
                    lambda x: np.sin(np.pi * x / a)
                    / (np.pi * x / a)
                    * np.sin(np.pi * x)
                    / (np.pi * x),
                    lambda x: 1,
                    lambda x: 0,
                ],
            )

        return _sinc
    return (lanczos,)


@app.cell
def __(NDArray, np):
    def generate_samples(
        f: callable, x_range: tuple[float, float], frequency: float
    ) -> NDArray:
        """
        Generates a set of sample points for a given function `f` within a specified range.

        Args:
          f (callable): The function to be sampled.
          x_range (tuple[float, float]): The desired range (minimum and maximum) for the x-axis.
          frequency (int): The number of samples to be generated within the x-range.

        Returns:
          np.ndarray: A 2D NumPy array containing the samples, where
                       the first dimension represents x-values and the second represents y-values.
        """
        x = np.linspace(*x_range, int(frequency * (x_range[1] - x_range[0])) + 1)
        y = f(x)
        return np.stack([x, y], axis=1)
    return (generate_samples,)


@app.cell
def __(NDArray, np):
    def interpolate_with_convolution(
        data: NDArray, g: callable, xi: int, ts: NDArray
    ):
        """
        Performs interpolation of a discrete signal using convolution with an interpolation function.

        Args:
            data: A 2D NumPy array where each row represents (k, f_k) pair.
                    k: The index of the sample.
                    f_k: The value of the k-th sample.
            g: A function representing the interpolation function g(t).
            xi: The frequency of the sample.
            ts: A NumPy array of time points for which to interpolate.

        Returns:
            A NumPy array containing the interpolated values of f(t).
        """
        # Extract k and f_k values from the data array
        k, f_k = (
            data[:, 0],
            data[:, 1],
        )  # Assuming data is in format [(k1, f1), (k2, f2), ...]

        # Convert the interpolation function to a callable form (if needed)
        g_callable = g if callable(g) else lambda x: g(x)

        # Extract interpolated values for the desired time points
        return [np.sum(g_callable(xi * (-k + t)) * f_k) for t in ts]
    return (interpolate_with_convolution,)


@app.cell
def __(
    generate_linspace_with_swing,
    generate_samples,
    interpolate_with_convolution,
    np,
    visualize_function_with_samples,
):
    def plot_interpolations(
        f: callable,
        x_range: tuple[float, float],
        frequency: int,
        interpolation_funcs: list[callable],
    ):
        """
        Creates a Plotly figure visualizing a function (`f`) along with its interpolations
        using provided functions and sampled points.

        Args:
            f (callable): The signal function to be visualized.
            x_range (tuple[float, float]): The desired range (minimum and maximum) for the x-axis.
            frequency (int): The number of samples to be generated within the x-range.
            interpolation_funcs (list[callable]): A list of functions to be used for interpolation.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure containing the visualization.
        """
        # Generate sample points
        samples = generate_samples(f, x_range, frequency)
        # Generate interpolated points for each function
        interpolated_points = generate_linspace_with_swing(*x_range, swing=0.1)
        interpolated_values = [
            np.array(
                [
                    interpolated_points,
                    interpolate_with_convolution(
                        samples, func, frequency, interpolated_points
                    ),
                ]
            ).T
            for func in interpolation_funcs
        ]
        # Extract function names from callables (if possible)
        names = [
            func.__name__ if hasattr(func, "__name__") else f"Interpolation {i+1}"
            for i, func in enumerate(interpolation_funcs)
        ]
        # Create the visualization using the provided function
        return visualize_function_with_samples(
            f, samples, interpolated_values, names
        )
    return (plot_interpolations,)


@app.cell
def __(Figure, generate_linspace_with_swing, go):
    def visualize_kernels(
        kernels: list[callable], limits: tuple[float, float] = (-2, 2)
    ) -> Figure:
        """
        Creates a Plotly figure visualizing a collection of kernel functions.

        Args:
            kernels (list[callable]): A list of kernel functions to be visualized.
            limits (tuple[float, float], optional): The desired range (minimum and maximum) for the x-axis. Defaults to (-2, 2).

        Returns:
            plotly.graph_objects.Figure: A Plotly figure containing the visualization.
        """

        x = generate_linspace_with_swing(limits[0], limits[1])
        fig = go.Figure()

        for kernel in kernels:
            y = kernel(x)
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", name=str(kernel.__name__))
            )  # Ensure name conversion

        return fig
    return (visualize_kernels,)


@app.cell
def __(NDArray, np):
    def generate_linspace_with_swing(
        low: float, high: float, swing: float = 0.0
    ) -> NDArray:
        """
        Generates a NumPy array of linearly spaced values within a specified range,
        including an optional swing factor to extend the range beyond the provided
        low and high bounds.

        Args:
            low (float): The lower bound of the desired range.
            high (float): The upper bound of the desired range.
            swing (float, optional): A value to add to both low and high,
                effectively extending the range by this amount on each end. Defaults to 0.0.

        Returns:
            NDArray: A NumPy array containing the linearly spaced values.
        """

        adjusted_low = low - swing
        adjusted_high = high + swing
        num_elements = int((adjusted_high - adjusted_low) * 100)

        return np.linspace(adjusted_low, adjusted_high, num_elements)
    return (generate_linspace_with_swing,)


@app.cell
def __(Figure, NDArray, generate_linspace_with_swing, go, np):
    def visualize_function_with_samples(
        f: callable,
        samples: NDArray,
        additional_traces: list[NDArray] = None,
        names: list[str] = None,
    ) -> Figure:
        """
        Creates a Plotly figure visualizing a function (`f`) along with provided samples and additional traces (optional).

        Args:
            f (callable): The function to be visualized.
            samples (NDArray): A 2D NumPy array containing sample points (x, y).
            additional_traces (list[NDArray], optional): A list of optional 2D NumPy arrays representing additional traces to be plotted. Defaults to None.
            names (list[str], optional): A list of names for the additional traces. Defaults to None.

        Returns:
            go.Figure: A Plotly figure containing the visualization.
        """

        fig = go.Figure()

        # Plot samples with clear labels
        fig.add_trace(
            go.Scatter(
                x=samples[:, 0],
                y=samples[:, 1],
                mode="markers",
                name="Samples",
                marker={"symbol": "circle-open", "color": "red", "size": 7},
            )
        )

        # Calculate data range for function plot
        x_min, x_max = np.min(samples[:, 0]), np.max(samples[:, 0])

        # Generate points for function evaluation with some swing
        x = generate_linspace_with_swing(x_min, x_max)
        y = f(x)

        # Plot the function with a descriptive name
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Function: f",
                marker={"color": "fuchsia"},
            )
        )

        # Add additional traces with labels (if provided)
        if additional_traces and names:
            for trace, name in zip(additional_traces, names):
                fig.add_trace(
                    go.Scatter(
                        x=trace[:, 0], y=trace[:, 1], mode="lines", name=name
                    )
                )

        return fig
    return (visualize_function_with_samples,)


@app.cell
def __(Image, np):
    def nearest_neighbor_interpolation(
        input_image: Image.Image, output_shape: tuple[int, int]
    ) -> Image.Image:
        """
        Resizes an image using nearest neighbor interpolation.

        Args:
            input_image (PIL.Image.Image): The input image to be resized.
            output_shape (tuple[int, int]): The desired output shape (height, width).

        Returns:
            PIL.Image.Image: The resized image using nearest neighbor interpolation.
        """

        input_height, input_width = input_image.size
        output_height, output_width = output_shape

        # Ensure input image is converted to a NumPy array for calculations
        image = np.array(input_image)

        # Generate grid of output coordinates
        x = np.arange(output_width)
        y = np.arange(output_height)
        xx, yy = np.meshgrid(x, y)

        # Compute scaled coordinates for input image (avoiding division by zero)
        x_scale = input_width / output_width if output_width != 0 else 0
        y_scale = input_height / output_height if output_height != 0 else 0
        x_mapped = np.round(xx * x_scale).astype(int)
        y_mapped = np.round(yy * y_scale).astype(int)

        # Handle out-of-bounds indices using a safe indexing approach
        x_mapped = np.clip(x_mapped, 0, input_width - 1)
        y_mapped = np.clip(y_mapped, 0, input_height - 1)

        # Safe indexing to avoid out-of-bounds errors
        output_image = image[y_mapped, x_mapped]

        # Convert the output back to a PIL Image for consistency
        return Image.fromarray(output_image.astype(np.uint8))
    return (nearest_neighbor_interpolation,)


@app.cell
def __(Image, np):
    def bilinear_interpolation(
        input_image: Image.Image, output_shape: tuple[int, int]
    ) -> Image.Image:
        """
        Resizes an image using bilinear interpolation.

        Args:
            input_image (PIL.Image.Image): The input image to be resized.
            output_shape (tuple[int, int]): The desired output shape (height, width).

        Returns:
            PIL.Image.Image: The resized image using bilinear interpolation.
        """

        input_height, input_width = input_image.size
        output_height, output_width = output_shape

        # Ensure input image is converted to a NumPy array for calculations
        image = np.array(input_image)

        # Generate grid of output coordinates
        x = np.linspace(0, input_width - 1, output_width)
        y = np.linspace(0, input_height - 1, output_height)
        xx, yy = np.meshgrid(x, y)

        # Compute corresponding coordinates in the input image (with clamping)
        x_low = np.clip(np.floor(xx).astype(int), 0, input_width - 1)
        y_low = np.clip(np.floor(yy).astype(int), 0, input_height - 1)
        x_high = np.clip(x_low + 1, 0, input_width - 1)
        y_high = np.clip(y_low + 1, 0, input_height - 1)

        # Compute fractional parts
        alpha = xx - x_low
        beta = yy - y_low

        # Perform bilinear interpolation
        output_image = (
            (1 - alpha) * (1 - beta) * image[y_low, x_low]
            + alpha * (1 - beta) * image[y_low, x_high]
            + (1 - alpha) * beta * image[y_high, x_low]
            + alpha * beta * image[y_high, x_high]
        )

        # Convert the output back to a PIL Image for consistency
        return Image.fromarray(output_image.astype(np.uint8))
    return (bilinear_interpolation,)


@app.cell
def __(Image, mo, plt):
    def show_images(images: list[Image.Image], titles: list[str]) -> mo.ui:
        """
        Displays a list of images using Matplotlib with optional titles.

        Args:
          images (list[PIL.Image.Image]): A list of PIL Image objects to be displayed.
          titles (list[str], optional): A list of titles for each image (same length as images).
        """
        if titles and len(images) != len(titles):
            raise ValueError("Number of images and titles must match.")
        plt.axis("off")
        return mo.hstack(
            [
                mo.vstack([plt.imshow(image, cmap="gray"), title], align="center")
                for image, title in zip(images, titles)
            ]
        )
    return (show_images,)


@app.cell
def __(Image, mo, np, show_images):
    def show_rows_of_images(images: list[Image.Image], titles: list[str]) -> mo.ui:
        """
        Displays a list of images using Matplotlib with optional titles.

        Args:
          images (list[PIL.Image.Image]): A list of PIL Image objects to be displayed.
          titles (list[str], optional): A list of titles for each image (same length as images).
        """
        return mo.vstack(
            [
                show_images(
                    images[2 * i : 2 * (i + 1)], titles[2 * i : 2 * (i + 1)]
                )
                for i in range(int(np.ceil(len(images) // 2)))
            ]
        )
    return (show_rows_of_images,)


@app.cell
def __(Image, form):
    lena = Image.open(form.value["image"]).convert("L")
    lena_downscaled = lena.resize((64, 64))
    return lena, lena_downscaled


@app.cell
def __(mo):
    downscaled_dim = mo.ui.range_slider(
        steps=[32, 64, 128, 256, 256, 128, 64, 32],
        show_value=True,
        label="Custom downscaled dimension of the original image: ",
    )
    return (downscaled_dim,)


@app.cell
def __(mo):
    resizing_dim = mo.ui.range_slider(
        steps=[32, 64, 128, 256, 512, 1024, 1024, 512, 256, 128, 64, 32],
        show_value=True,
        label="Custom resized dimension of the original image: ",
    )
    return (resizing_dim,)


@app.cell
def __(Image, lena_downscaled, resizing_dim, show_rows_of_images):
    exp_result = show_rows_of_images(
        [
            lena_downscaled.resize(resizing_dim.value, Image.NEAREST),
            lena_downscaled.resize(resizing_dim.value, Image.BILINEAR),
            lena_downscaled.resize(resizing_dim.value, Image.BICUBIC),
            lena_downscaled.resize(resizing_dim.value, Image.LANCZOS),
        ],
        [
            f"Lena NN-Interpolated {resizing_dim.value[0]}*{resizing_dim.value[1]}",
            f"Lena Bilinear-Interpolated {resizing_dim.value[0]}*{resizing_dim.value[1]}",
            f"Lena Bicubic-Interpolated {resizing_dim.value[0]}*{resizing_dim.value[1]}",
            f"Lena Lánczos-Interpolated {resizing_dim.value[0]}*{resizing_dim.value[1]}",
        ],
    )
    return (exp_result,)


@app.cell
def __(Image, lena_downscaled, show_images):
    lanczos_result = show_images(
        [
            lena_downscaled,
            lena_downscaled.resize((512, 512), Image.LANCZOS),
        ],
        ["Lena 64*64", "Lena Lanczos-Interpolated 512*512"],
    )
    return (lanczos_result,)


@app.cell
def __(Image, lena_downscaled, show_images):
    bicubic_result = show_images(
        [
            lena_downscaled,
            lena_downscaled.resize((512, 512), Image.BICUBIC),
        ],
        ["Lena 64*64", "Lena Bicubic-Interpolated 512*512"],
    )
    return (bicubic_result,)


@app.cell
def __(bilinear_interpolation, lena_downscaled, show_images):
    bilinear_result = show_images(
        [
            lena_downscaled,
            bilinear_interpolation(lena_downscaled, (512, 512)),
        ],
        ["Lena 64*64", "Lena Bilinear-Interpolated 512*512"],
    )
    return (bilinear_result,)


@app.cell
def __(lena_downscaled, nearest_neighbor_interpolation, show_images):
    nn_result = show_images(
        [
            lena_downscaled,
            nearest_neighbor_interpolation(lena_downscaled, (512, 512)),
        ],
        ["Lena 64*64", "Lena NN-Interpolated 512*512"],
    )
    return (nn_result,)


@app.cell
def __(np):
    sgn = lambda x: np.sign(x)
    return (sgn,)


@app.cell
def __(mo):
    sample_freq = mo.ui.slider(
        steps=[0.5, 1, 1.5, 2, 3, 4, 8, 16],
        value=2,
        show_value=True,
        label="Select frequency for sampling: ",
    )
    bicubic_a_slider = mo.ui.slider(
        steps=[-1 + 0.25 * i for i in range(10)],
        value=-0.75,
        show_value=True,
        label="Select a for bicubic: ",
    )
    sinc_a_slider = mo.ui.slider(
        5,
        10,
        show_value=True,
        label="Select a for sinc: ",
    )
    lanczos_a_slider = mo.ui.slider(
        5,
        10,
        show_value=True,
        label="Select a for lanczos: ",
    )
    return bicubic_a_slider, lanczos_a_slider, sample_freq, sinc_a_slider


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    from numpy.typing import NDArray
    from PIL import Image
    return Image, NDArray, go, mo, np, plt


@app.cell
def __(mo):
    import plotly.io as pio

    pio.templates.default = (
        "simple_white" if mo.app_meta().theme == "light" else "plotly_darf"
    )
    return (pio,)


@app.cell
def __():
    NN_SRC = "../assets/image-processing/nn.png"
    BILINEAR_SRC = "../assets/image-processing/bilinear.png"
    BICUBIC_SRC = "../assets/image-processing/bicubic.png"
    return BICUBIC_SRC, BILINEAR_SRC, NN_SRC


if __name__ == "__main__":
    app.run()
