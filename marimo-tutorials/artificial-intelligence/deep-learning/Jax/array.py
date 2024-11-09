# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imageio==2.36.0",
#     "jax==0.4.35",
#     "jaxlib",
#     "numpy==2.1.2",
#     "requests==2.32.3",
#     "scipy==1.14.1",
#     "marimo",
#     "anywidget==0.9.13",
#     "traitlets==5.14.3",
# ]
# ///

import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell(hide_code=True)
def __(header_widget):
    header_widget
    return


@app.cell(hide_code=True)
def __(mo):
    use_jax = mo.ui.switch(value=True, label="Switch to Jax")
    use_jax.center()
    return (use_jax,)


@app.cell(hide_code=True)
def __(use_jax):
    if use_jax.value:
        import jax.numpy as np
        from jax.scipy.signal import convolve2d
    else:
        import numpy as np
        from scipy.signal import convolve2d
    return convolve2d, np


@app.cell(hide_code=True)
def __(use_jax):
    numpy_or_jax = "Jax" if use_jax.value else "NumPy"
    return (numpy_or_jax,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        JAX has a NumPy-like API imported from the [`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html) module. There are also some higher-level functions from SciPy reimplemented in JAX. This [`jax.scipy`](https://jax.readthedocs.io/en/latest/jax.scipy.html) module is less rich than the entire SciPy library, yet the function we used (the [`convolve2d()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.signal.convolve2d.html) function) is present there.

        Sometimes, you find there is no corresponding function in JAX. For example, we might use the [`gaussian_filter()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) function from [`scipy.ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html) for Gaussian filtering. There is no such function in `jax.scipy.ndimage`.

        In such a case, you might still use the NumPy function with JAX and have two imports, one from NumPy and another for the JAX NumPy interface. It is usually done this way:

        ```python
        ## NumPy
        import numpy as np

        ## JAX
        import jax.numpy as jnp
        ```
        """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(mo, numpy_or_jax):
    mo.md(
        f"""
        # Image processing with {numpy_or_jax} arrays



        Our image processing example will consist of several steps:

        - Loading an image from a file into a {numpy_or_jax} array.
        - Preprocess the image if necessary and convert pixel values to float numbers.
        - Generate a noisy version of the photo to simulate camera noise.
        - Filtering the image to reduce noise and increase sharpness.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo, numpy_or_jax):
    mo.md(
        f"""
        ## Loading an image into a {numpy_or_jax} array

        We will use the `marimo-community` image, of course.

        - if you want to use your own image for this notebook, just modify the `src` below
        """
    )
    return


@app.cell(hide_code=True)
def __():
    src = "https://raw.githubusercontent.com/Haleshot/marimo-tutorials/main/community-tutorials-banner.png"
    return (src,)


@app.cell(hide_code=True)
def __(np):
    import requests
    import imageio.v3 as iio


    def load_image_from_url(url):
        # Fetch the image data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses

        # Convert the image data to a numpy array
        image_array = iio.imread(response.content)

        return np.asarray(image_array)
    return iio, load_image_from_url, requests


@app.cell(hide_code=True)
def __(load_image_from_url, src):
    image = load_image_from_url(src)
    image = image / 255 if image.max() > 1 else image
    return (image,)


@app.cell(hide_code=True)
def __(image, mo):
    mo.image(image)
    return


@app.cell(hide_code=True)
def __(mo, numpy_or_jax):
    mo.md(
        f"""
        ## Performing basic preprocessing operations with image

        ### Crop image with {numpy_or_jax} slicing

        You may want to crop the image to eliminate some irrelevant details near the border.

        With slicing, you can select specific tensor elements along each axis. For example, you can select a specific rectangular image subregion or take only chosen color channels.
        """
    )
    return


@app.cell(hide_code=True)
def __(image, mo):
    width_range_selector = mo.ui.range_slider(
        start=0,
        step=1,
        stop=image.shape[0] - 1,
        label="Select width range of the cropped image",
        show_value=True,
    )
    height_range_selector = mo.ui.range_slider(
        start=0,
        step=1,
        stop=image.shape[1] - 1,
        label="Select height range of the cropped image",
        show_value=True,
    )
    channel_selector = mo.ui.multiselect(
        options={f"Channel {c}": c for c in range(image.shape[-1])},
        value=[f"Channel {c}" for c in range(image.shape[-1])],
        label="Select channels of the cropped image",
    )
    return channel_selector, height_range_selector, width_range_selector


@app.cell(hide_code=True)
def __():
    def get_cropped_image(
        image, width_selector, height_selector, channel_selector
    ):
        def range_selector_value_to_cropped_range(selector):
            value = selector.value
            return value if len(value) == 1 else slice(value[0], value[1] + 1)

        def multiselect_value_to_channels(selector):
            match len(selector.value):
                case 1:
                    return selector.value[0]
                case 2:
                    return sorted(selector.value)
                case _:
                    return list(range(len(selector.options)))

        width_range, height_range = (
            range_selector_value_to_cropped_range(width_selector),
            range_selector_value_to_cropped_range(height_selector),
        )

        channels = multiselect_value_to_channels(channel_selector)
        return image[width_range, height_range, channels]
    return (get_cropped_image,)


@app.cell(hide_code=True)
def __(
    channel_selector,
    cropped_image,
    height_range_selector,
    mo,
    width_range_selector,
):
    mo.vstack(
        [
            width_range_selector,
            height_range_selector,
            channel_selector,
            mo.image(cropped_image),
        ],
        justify="space-around",
    )
    return


@app.cell(hide_code=True)
def __(
    channel_selector,
    get_cropped_image,
    height_range_selector,
    image,
    width_range_selector,
):
    cropped_image = get_cropped_image(
        image,
        width_range_selector,
        height_range_selector,
        channel_selector,
    )
    return (cropped_image,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""we will use the cropped image from here""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ##  Adding noise to the image

        > This is only required for demonstration purposes.

        To simulate a noisy image, we use the [Gaussian noise](https://en.wikipedia.org/wiki/Gaussian_noise), which frequently appears in digital cameras with low-light conditions and high ISO light sensitivities. We use [`random.normal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html#jax.random.normal) from the Jax library.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    noise_eps = mo.ui.slider(
        start=0.01,
        step=0.001,
        stop=0.1,
        show_value=True,
        label="control scaling factor of added noise",
    )
    return (noise_eps,)


@app.cell(hide_code=True)
def __(noise_eps, random):
    def add_noise(val, key, eps=noise_eps.value):
        return val + eps * random.normal(key, val.shape)
    return (add_noise,)


@app.cell(hide_code=True)
def __(mo, noise_eps, noisy_image):
    mo.vstack([noise_eps, mo.image(noisy_image)])
    return


@app.cell(hide_code=True)
def __(add_noise, cropped_image, random):
    noisy_image = add_noise(cropped_image, random.key(1118))
    return (noisy_image,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""we will use the cropped and noisy image from here""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ##  Implementing image filtering

        > This step is the core of our image processing example.

        This step consists of two substeps 

        1. creating a filter kernel
        2. applying the filter kernel to an image
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Creating a filter kernel

        [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) filters typically remove the type of noise we have. Gaussian blur filter belongs to a large family of matrix filters, also called [finite impulse response](https://en.wikipedia.org/wiki/Finite_impulse_response), or FIR, filters in the Digital Signal Processing (DSP) field.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""you can check [our notebook for filters](https://github.com/Haleshot/marimo-tutorials/blob/67e6b965acd4301e814e6471265dc96922038740/marimo-tutorials/Signal-Image-Processing/image-processing/basic_image_scaling.py)"""
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The matrix of Gaussian blur contains different values that tend to have higher values closer to the center. The well-known Gaussian function produces the Gaussian kernel.""")
    return


@app.cell(hide_code=True)
def __(np):
    def gaussian_kernel(kernel_size, sigma=1.0, mu=0.0):
        """A function to generate Gaussian 2D kernel"""
        center = kernel_size // 2
        coords = np.mgrid[
            -center : kernel_size - center, -center : kernel_size - center
        ]
        distances = np.sqrt(np.sum(np.square(coords), axis=0))

        normalization_factor = 1 / (2 * np.pi * sigma**2)
        kernel = normalization_factor * np.exp(
            -((distances - mu) ** 2) / (2 * sigma**2)
        )

        return kernel
    return (gaussian_kernel,)


@app.cell(hide_code=True)
def __(np):
    def upscale_kernel(kernel: np.ndarray, scale: int = 33) -> np.ndarray:
        """
        Upscale a kernel by duplicating each element in width and height.

        Args:
            kernel (np.ndarray): The input 2D array (kernel) to upscale.
            scale (int): The number of times to duplicate each element in both dimensions.

        Returns:
            np.ndarray: The upscaled kernel ready for visualization.
        """
        # Use numpy's repeat to duplicate values along both dimensions
        upscaled = np.repeat(np.repeat(kernel, scale, axis=0), scale, axis=1)
        return upscaled
    return (upscale_kernel,)


@app.cell(hide_code=True)
def __(mo):
    gaussian_kernel_size = mo.ui.slider(
        start=3,
        step=2,
        stop=13,
        value=5,
        show_value=True,
        label="control kernel size of the gaussian blur filter",
    )
    return (gaussian_kernel_size,)


@app.cell(hide_code=True)
def __(gaussian_kernel_size, kernel_gauss, mo, upscale_kernel):
    mo.vstack(
        [gaussian_kernel_size, mo.image(upscale_kernel(kernel_gauss)).center()]
    )
    return


@app.cell(hide_code=True)
def __(gaussian_kernel, gaussian_kernel_size):
    kernel_gauss = gaussian_kernel(gaussian_kernel_size.value)
    return (kernel_gauss,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Applying the filter kernel to an image

        We apply a filter to each color channel separately. The function must apply a [2D convolution](https://en.wikipedia.org/wiki/Multidimensional_discrete_convolution) with the filter kernel inside each color channel. We also __clip the resulting values__ to restrict the range to [0.0, 1.0]. Then, we merge the processed color channels together to form a processed image.
        """
    )
    return


@app.cell(hide_code=True)
def __(convolve2d, np):
    def color_convolution(image, kernel):
        """A function to apply a filter to an image"""
        convolve_channel = lambda channel: np.clip(
            convolve2d(channel, kernel, mode="same"), 0.0, 1.0
        )

        filtered_channels = [
            convolve_channel(image[:, :, i]) for i in range(image.shape[-1])
        ]

        return np.stack(filtered_channels, axis=2)
    return (color_convolution,)


@app.cell(hide_code=True)
def __(color_convolution, kernel_gauss, noisy_image):
    blurred_image = color_convolution(noisy_image, kernel_gauss)
    return (blurred_image,)


@app.cell(hide_code=True)
def __(blurred_image, mo):
    mo.image(blurred_image)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""we will use the blurred image from here""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Unsharp Masking""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        After denoising, the amount of noise is significantly reduced, but the image becomes blurry. While it is expected, we would like to make the image sharper if possible.

        A simple algorithm ([Unsharp Masking]((https://en.wikipedia.org/wiki/Unsharp_masking))) for sharpening an image is:

        1. Blur an image using whatever blur you wish (e.g., Box, Tent, Gaussian)
        2. Subtract the blurred result from the original image to get high-frequency details.
        3. Add the high-frequency details to the original image.

        Algebraically, this can be expressed as:

        ```
        image + (image – blurred)
        ```

        If we wanted to make a 3×3 box blur into a sharpening filter we would calculate it this way:

        $$
        2 * \begin{bmatrix}
            0 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 0
        \end{bmatrix} - \begin{bmatrix}
            1 & 1 & 1 \\
            1 & 1 & 1 \\
            1 & 1 & 1
        \end{bmatrix} * \frac{1}{9} = \begin{bmatrix}
            -1 & -1 & -1 \\
            -1 & 17 & -1 \\
            -1 & -1 & -1
        \end{bmatrix} * \frac{1}{9}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    sharpen_kernel_size = mo.ui.slider(
        start=3,
        step=2,
        stop=13,
        value=5,
        show_value=True,
        label="control kernel size of the sharpen filter",
    )
    return (sharpen_kernel_size,)


@app.cell(hide_code=True)
def __(np):
    def sharpen_kernel(kernel_size):
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        image_matrix = np.pad(
            np.full((1, 1), 1),
            pad_width=(kernel_size // 2),
            mode="constant",
        )

        blurred_matrix = np.ones((kernel_size, kernel_size)) / kernel_size**2

        return 2 * image_matrix - blurred_matrix
    return (sharpen_kernel,)


@app.cell(hide_code=True)
def __(sharpen_kernel, sharpen_kernel_size):
    kernel_sharpen = sharpen_kernel(sharpen_kernel_size.value)
    return (kernel_sharpen,)


@app.cell(hide_code=True)
def __(kernel_sharpen, mo, sharpen_kernel_size, upscale_kernel):
    mo.vstack(
        [sharpen_kernel_size, mo.image(upscale_kernel(kernel_sharpen)).center()]
    )
    return


@app.cell(hide_code=True)
def __(blurred_image, color_convolution, kernel_sharpen):
    restored_image = color_convolution(blurred_image, kernel_sharpen)
    return (restored_image,)


@app.cell(hide_code=True)
def __(mo, restored_image):
    mo.image(restored_image)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""let's visualize these images""")
    return


@app.cell(hide_code=True)
def __(
    blurred_image,
    channel_selector,
    cropped_image,
    gaussian_kernel_size,
    height_range_selector,
    mo,
    noise_eps,
    noisy_image,
    restored_image,
    width_range_selector,
):
    mo.vstack(
        [
            width_range_selector,
            height_range_selector,
            channel_selector,
            noise_eps,
            gaussian_kernel_size,
            gaussian_kernel_size,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**cropped image**"),
                            mo.image(cropped_image),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**noised image**"),
                            mo.image(noisy_image),
                        ]
                    ),
                ]
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**blurred image**"),
                            mo.image(blurred_image),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**restored image**"),
                            mo.image(restored_image),
                        ]
                    ),
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Arrays in JAX

        ## What is the Array?

        [Array](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.html) is the default type for representing arrays in JAX. It can use different backends — CPU, GPU, and TPU. It is equivalent to the numpy.ndarray backed by a memory buffer on a single device as well as on multiple devices. In general, a device is something used by JAX to perform computations.

        A noticeable difference from NumPy here is that NumPy usually accepts Python lists or tuples as inputs to its API functions (not including the `array()` constructor). JAX deliberately chooses __not to accept lists or tuples as inputs to its functions__ because that can lead to silent performance degradation, which is hard to detect. If you want to pass a Python list to a JAX function, you must explicitly convert it into an array. The following code demonstrates working with Python lists in JAX functions.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp, mo):
    with mo.redirect_stdout():
        try:
            jnp.sum([1, 42, 31337])
        except TypeError as e:
            print(e)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Device-related operations

        Special computing devices like GPU or TPU help make your code run faster. Sometimes, the difference may be of orders of magnitude. This is especially important for training large neural networks or performing large-scale simulations. So, you should know how to use this hardware acceleration power. To use accelerated computations, an accelerator (GPU or TPU) needs all the data participating in the computations (the tensors themselves) to reside in a device’s memory. So, the first step to using hardware acceleration is to learn how to transfer data between devices.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Local and global devices

        A host is the __CPU that manages several devices__. A single host can manage several devices (usually up to 8), so a multi-host configuration is needed to use more devices (it will also be multi-process as in this case, there will be many JAX Python processes run independently on each host).

        JAX distinguishes between __local__ and __global__ devices.



        A __local device__ for a process is a device that the process can __directly address and launch computations on__. It is a device __attached directly to the host__ (or computer) where the JAX program runs, for example, a CPU, a local GPU, or 8 TPU cores directly attached to the host. The [`jax.local_devices()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.local_devices.html) function shows a process’s local devices. The [`jax.local_device_count()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.local_device_count.html#jax.local_device_count) function returns the number of devices addressable by this process. Both functions receive a parameter for the XLA backend that could be 'cpu', 'gpu', or 'tpu'. By default, this parameter is `None`, which means the default backend (GPU or TPU if available).

        A __global device__ is a device across all processes. It is __relevant to multi-host and multi-process environments__. As long as each process launches the computation on its local devices, a computation can span devices across processes and use collective operations via direct communication links between the devices (usually, the high-speed interconnects between Cloud TPUs or GPUs). The [`jax.devices()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.devices.html#jax.devices) function shows all available global devices, and [`jax.device_count()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_count.html#jax.device_count) returns the total number of devices across all processes.

        For now, let’s concentrate on single-host environments only; in our case, the global device list will be equal to the local one.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""If you happen to use a macbook as me, you can install [`jax-metal`](https://developer.apple.com/metal/jax/) to enable accelerated JAX"""
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Committed and uncommitted data

        In JAX, the computation follows data placement. There are two different placement properties.


        1. The device where the data resides.
        2. Whether the data is committed to the device or not. When the data is committed, it is sometimes referred to as being sticky to the device.

        You can know where the data is located with the help of the `device()` method.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp):
    jnp.array([1, 42, 31337]).device
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        By default, JAX Array objects are placed uncommitted on the default device. The default device is the first item of the list returned by `jax.devices()` function call (`jax.devices()[0]`). It is the first GPU or TPU if it is present; otherwise, it is the CPU.

        You can use [`jax.default_device()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.default_device.html#jax.default_device) context manager to temporarily override the default device for JAX operations if you want to. You can also use the `JAX_PLATFORMS` environment variable or the `--jax_platforms` command line flag. It is also possible to set a priority order when providing a list of platforms in the `JAX_PLATFORMS` variable.

        __Computations involving uncommitted data are performed on the default device, and the results are also uncommitted on the default device.__

        Say you want a specific GPU to run computations with a specific tensor. You can explicitly put data on a specific device using the [`jax.device_put()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html#jax.device_put) function call with a `device` parameter. In this case, the __data becomes committed to the device__. If you pass `None` as the device parameter, the operation will behave like the identity function if the operand is already on any device. Otherwise, it will transfer the data to the default device uncommitted.
        """
    )
    return


@app.cell(hide_code=True)
def __(jax, jnp, mo):
    with mo.redirect_stdout():
        arr = jnp.array([1, 42, 31337])
        arr_cpu = jax.device_put(arr, jax.devices("cpu")[0])
        print(f"Put to CPU: {arr_cpu.device}")
        print(f"Original Device: {arr.device}")
    return arr, arr_cpu


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""Remember the functional nature of JAX. The jax.device_put() function creates a copy of your data on the specified device and returns it. The original data is unchanged."""
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""There is a reverse operation [`jax.device_get()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_get.html#jax.device_get) to transfer data from a device to the Python process on your host. The returned data is a NumPy ndarray.""")
    return


@app.cell(hide_code=True)
def __(arr, jax):
    arr_host = jax.device_get(arr)
    type(arr_host)
    return (arr_host,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Computations involving committed data are performed on the committed device, and the results will be committed to the same device. __You will get an error when you invoke an operation on arguments committed to different devices__ (but no error if some arguments are uncommitted).""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Asynchronous dispatch

        An important thing to know about how JAX works is [asynchronous dispatch](https://jax.readthedocs.io/en/latest/async_dispatch.html).

        JAX uses asynchronous dispatch. It means that when an operation is executed, JAX does not wait for operation completion and returns control to the Python program. JAX returns an `Array`, which is technically a `future`. In the `future`, a value is not available immediately and, as the name suggests, __will be produced in the future on an accelerator device__. Yet it __already contains the shape and type__; __you can also pass it to the subsequent JAX computation__.

        Asynchronous dispatch is very useful as it allows Python to run ahead and not wait for an accelerator, helping the Python code not be on the critical path. If the Python code enqueues computations on a device faster than they can be executed, and if it does not need to inspect values in between, then the Python program can use an accelerator most efficiently without the accelerator having to wait.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""Not knowing about asynchronous dispatch may mislead you when doing benchmarks, and you might obtain over-optimistic results. This is why we should use [`block_until_ready()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.block_until_ready.html#jax.block_until_ready) method when we benchmark a function computation on different backends with and without JIT compilation."""
    ).callout(kind="warn")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""In the following example, we highlight the difference in time when measuring with and without blocking. Without blocking, we measure only the time to dispatch the work, not counting the computation. Additionally, we measure the computation time on GPU and CPU. We do it by committing the data tensors to the corresponding devices.""")
    return


@app.cell(hide_code=True)
def __():
    import time


    def evaluate_execution_time(event, num_execution: int = 33) -> str:
        """
        Evaluates the median execution time of an event and returns formatted string with appropriate time units.

        Args:
            event: Callable to be timed
            num_execution: Number of times to execute the event

        Returns:
            str: Formatted time string (e.g., "123 µs", "45.6 ms", "1.2 s")
        """
        execution_times = []
        for _ in range(num_execution):
            start_time = time.time()
            event()
            end_time = time.time()
            execution_times.append(end_time - start_time)

        median_time = sorted(execution_times)[num_execution // 2]

        # Format time with appropriate units
        if median_time < 1e-3:
            return f"{median_time * 1e6:.1f} µs"
        elif median_time < 1:
            return f"{median_time * 1e3:.1f} ms"
        else:
            return f"{median_time:.1f} s"
    return evaluate_execution_time, time


@app.cell(hide_code=True)
def __(evaluate_execution_time, jax, jnp, mo, np):
    array_on_a_device = jnp.array(range(1000000), dtype=np.float32).reshape(
        (1000, 1000)
    )
    array_on_cpu = jax.device_put(array_on_a_device, jax.devices("cpu")[0])


    def jnp_dot():
        jnp.dot(array_on_a_device, array_on_a_device)
        return True


    def jnp_dot_cpu():
        jnp.dot(array_on_cpu, array_on_cpu)
        return True


    def jnp_dot_block():
        jnp.dot(array_on_a_device, array_on_a_device).block_until_ready()
        return True


    def jnp_dot_block_cpu():
        jnp.dot(array_on_cpu, array_on_cpu).block_until_ready()
        return True


    with mo.redirect_stdout():
        print(f"device: {array_on_a_device.device}")
        print(f"without blocking: {evaluate_execution_time(jnp_dot)}")
        print(f"with blocking: {evaluate_execution_time(jnp_dot_block)}")
        print("-------------------------")
        print(f"device: {array_on_cpu.device}")
        print(f"without blocking: {evaluate_execution_time(jnp_dot_cpu)}")
        print(f"with blocking: {evaluate_execution_time(jnp_dot_block_cpu)}")
    return (
        array_on_a_device,
        array_on_cpu,
        jnp_dot,
        jnp_dot_block,
        jnp_dot_block_cpu,
        jnp_dot_cpu,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Differences with NumPy

        1. **Accelerator Support**: JAX supports different backends like CPU, GPU, and TPU, allowing precise management of tensor device placement and efficient use of accelerated computations through asynchronous dispatch.

        2. **Non-array Inputs**: JAX functions often do not accept lists or tuples as inputs to prevent performance degradation, unlike NumPy.

        3. **Immutability**: JAX emphasizes immutability, which is a key difference from NumPy.

        4. **Data Types and Type Promotion**: There are special considerations in JAX related to supported data types and type promotion, which differ from NumPy.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Jax arrays are immutable

        JAX is designed to follow the __functional programming paradigm__. 

        One of the basics of functional purity is the code must not have __side effects__. The code that modifies the original arguments is not functionally pure. The only way to create a modified tensor is to create another tensor based on the original one.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp, mo):
    with mo.redirect_stdout():
        aaaaaarray_jnp_immutable = jnp.array(range(10))

        try:
            aaaaaarray_jnp_immutable[0] = 1e10
        except TypeError as e:
            print(e)
    return (aaaaaarray_jnp_immutable,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Out-of-Bounds Indexing

        A common type of bug is to index arrays outside of their bounds. In NumPy, relying on Python exceptions to handle such situations was pretty straightforward. However, when the code runs on an accelerator, it may be difficult or even impossible. Therefore, we need some __non-error behavior for out-of-bounds indexing__. __For index update out-of-bound operations, we’d like to skip such updates, and for index retrieval out-of-bound operations, the index is clamped to the bound of the array as we need something to return__. 

        By default, __JAX assumes that all indices are in-bounds__. There is experimental support for giving more precise semantics to out-of-bounds indexed accesses via the [`mode` parameter of the index update functions](https://jax.readthedocs.io/en/latest/_autosummary/jax.Array.at.html). The possible options are

        - `promise_in_bounds` (default): The user promises that all indexes are in-bounds, so no additional checking is performed. In practice, it means that all out-of-bound indices in `get()` are __clipped__, and in `set()`, `add()`, and other modifying functions are __dropped__.
        - `clip`: clamps out-of-bounds indices into valid range.
        - `drop`: ignores out-of-bound indices.
        - `fill`: is an alias for `drop`, but for `get()`, it will return the value specified in the optional `fill_value` argument.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp, mo):
    with mo.redirect_stdout():
        arrrraaay_jnp_modesssss = jnp.array(range(10))

        print(f"array: {arrrraaay_jnp_modesssss}")

        print(f"default mode when get:{arrrraaay_jnp_modesssss.at[42].get()}")

        print(f"drop mode: {arrrraaay_jnp_modesssss.at[42].get(mode='drop')}")

        print(
            f"fill mode: {arrrraaay_jnp_modesssss.at[42].get(mode='fill', fill_value=-1)}"
        )

        arrrraaay_jnp_modesssss = arrrraaay_jnp_modesssss.at[42].set(100)

        print(f"default mode when set: {arrrraaay_jnp_modesssss}")

        arrrraaay_jnp_modesssss = arrrraaay_jnp_modesssss.at[42].set(
            100, mode="clip"
        )

        print(f"clip mode: {arrrraaay_jnp_modesssss}")
    return (arrrraaay_jnp_modesssss,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Types

        While NumPy aggressively promotes operands to double precision (or `float64`) type, JAX, by default, enforces single-precision (or `float32`) numbers.

        For many machine learning (and especially deep learning) workloads, that’s perfectly fine. For some high-precision scientific calculations, it may not be desired.

        To force `float64` computations, you need to set the `jax_enable_x64` configuration variable at startup. The following code demonstrates this:

        ```python
        # this only works on startup
        from jax.config import config
        config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        # this may not work on TPU backend. Try using CPU or GPU.
        x = jnp.array(range(10), dtype=jnp.float64)
        x.dtype

        >>> dtype('float64')
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""64-bit data types are not supported on every backend. For example, TPU does not support it."""
    ).callout(kind="warn")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### float16/bfloat16 support

        In deep learning, there is a tendency to __use lower precision formats__, most frequently, half-precision or `float16`, or a more special [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), which is not supported in vanilla NumPy.

        With JAX, you can easily switch to using these lower-precision 16-bit types.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp, mo):
    with mo.redirect_stdout():
        xb16 = jnp.array(range(10), dtype=jnp.bfloat16)
        print(f"dtype: {xb16.dtype}")
        x16 = jnp.array(range(10), dtype=jnp.float16)
        print(f"dtype: {x16.dtype}")
    return x16, xb16


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Again, there may be limitations on specific backends.""").callout(
        kind="warn"
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Type promotion semantics

        For binary operations, JAX’s [type promotion rules](https://jax.readthedocs.io/en/latest/type_promotion.html) differ somewhat from NumPy's.

        Interestingly, when you add two 16-bit floats, one being an ordinary float16 and another one being bfloat16, you get float32 type.

        For a more thorough [comparison](https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html#type-promotion-in-jax)
        """
    )
    return


@app.cell(hide_code=True)
def __(mo, x16, xb16):
    with mo.redirect_stdout():
        print(f"dtype of adding bfloat16 and float16: {(xb16+x16).dtype}")

        print(f"dtype of adding 2 bfloat16: {(xb16+xb16).dtype}")

        print(f"dtype of adding 2 float16: {(x16+x16).dtype}")
    return


@app.cell(hide_code=True)
def __():
    import jax
    import jax.numpy as jnp
    return jax, jnp


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __():
    from jax import random
    return (random,)


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
            banner.src = "https://raw.githubusercontent.com/Haleshot/marimo-tutorials/main/community-tutorials-banner.png";
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
def __(HeaderWidget):
    header_widget = HeaderWidget(
        result={
            "Title": "Working with arrays",
            "Author": "Eugene",
            "Date": "2024-10-30",
            "Version": "0.1",
            "Description": "This notebook will cover arrays and their operations in JAX. We will implement an image processing example using matrix filters. Then, we will introduce the JAX Array data structure, and explain JAX-specific things, especially device-related operations. Finally, we highlight the differences between NumPy and JAX APIs.",
            "Keywords": "deep learning, numerical computing",
            "Data Sources": "mairmo-community",
            "Tools Used": "Python, Jax, Scipy",
        }
    )
    return (header_widget,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # References

        - [Deep Learning with Jax, Chapter 3](https://livebook.manning.com/book/deep-learning-with-jax/chapter-3)
        - [Unsharp masking](https://en.wikipedia.org/wiki/Unsharp_masking)
        - [Image Sharpening Convolution Kernels](https://blog.demofox.org/2022/02/26/image-sharpening-convolution-kernels/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
