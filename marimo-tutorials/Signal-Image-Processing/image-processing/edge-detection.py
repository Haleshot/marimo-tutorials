import marimo

__generated_with = "0.6.16"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        rf"""
        # Image Edge Detection using Sobel Operator and Averaging Filter

        ## Aim

        The aim of this tutorial is to apply Sobel's mask on the given test image to obtain the components of the gradient, |g_x|, |g_y|, and |g_x+g_y|. Additionally, a 5x5 averaging filter is applied on the test image followed by implementing the sequence from step a. The tutorial concludes with summarizing the observations after comparing the results obtained in step a and b.

        ## Table of Contents

        - [Aim](#aim)
        - [Software](#software)
        - [Prerequisite](#prerequisite)
        - [Outcome](#outcome)
        - [Theory](#theory)

        ## Software

        This tutorial is implemented using Python.

        ## Prerequisite

        To understand and work with this tutorial, you should be familiar with the following concepts:

        | Sr. No | Concepts        |
        | ------ | --------------- |
        | 1.     | Sobel operator  |

        ## Outcome

        After successful completion of this experiment, students will be able to:

        - Understand the significance of filter masks for edge enhancement.
        - Implement Sobel operator for edge detection.
        - Apply Sobel's mask to obtain the components of the gradient.
        - Apply an averaging filter to the test image and observe the effects.
        - Compare the results obtained from step a and b and summarize the observations.

        ## Theory

        ### Sobel Operator

        The Sobel operator is a commonly used edge detection filter. It consists of two 3x3 masks: F_x and F_y, which are applied to the image to obtain the horizontal and vertical gradient components, respectively.

        ```
        F_x = |-1 -2 -1|      F_y = |-1  0  1|
              | 0  0  0|            | -2 0  2|
              | 1  2  1|            | -1 0  1|
        ```

        To apply the Sobel operator:

        1. Convolve the F_x mask with the original image to obtain the x gradient of the image.

        2. Convolve the F_y mask with the original image to obtain the y gradient of the image.

        3. Add the results of the above two steps to obtain the combined gradient image, |g_x+g_y|.

        ### Averaging Filter

        An averaging filter is a simple low-pass filter that helps in reducing noise and blurring the image. It involves convolving the image with a suitable filter mask.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ### Defining masks as discussed in [theory](#theory) above
         - Horizontal
         - Vertical masks and
         - adding the result of the two to form a diagonal mask.
        """
    )
    return


@app.cell
def __(np):
    Fx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Fy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return Fx, Fy


@app.cell
def __(Fx, Fy):
    diagonal = Fx + Fy
    return diagonal,


@app.cell
def __(mo):
    mo.md(
        rf"""
        ### Horizonal, vertical and diagonal masks used on images here:

        ```
        F_x = |-1 -2 -1|      F_y = |-1  0  1|      diagonal = |-2 -2 0|
              | 0  0  0|            | -2 0  2|                 |-2 0 2 |
              | 1  2  1|            | -1 0  1|                 | 0 2 2 |
        ```
        """
    )
    return


@app.cell
def __(img, np):
    m, n = img.shape
    img_horizontal = np.zeros([m, n])
    return img_horizontal, m, n


@app.cell
def __(mo):
    mo.md(rf"### Horizontal Edge Detection using list slicing")
    return


@app.cell
def __(Fx, img, img_horizontal, m, n, np, plt):
    # Horizontal Edge Detection using list slicing:
    _a = 1

    for _i in range(_a, m - _a):
      for _j in range(_a, n - _a):
        _temp = img[_i - _a:_i + _a + 1, _j - _a:_j + _a + 1]
        img_horizontal[_i, _j] = np.sum(np.multiply(_temp, Fx))

    plt.axis("off")
    plt.imshow(img_horizontal, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(mo):
    mo.md(rf"### Vertical Edge Detection using list slicing")
    return


@app.cell
def __(Fy, img, m, n, np, plt):
    # Vertical Edge Detection using list slicing:
    img_vertical = np.zeros([m, n])
    _a = 1

    for _i in range(_a, m - _a):
      for _j in range(_a, n - _a):
        _temp = img[_i - _a:_i + _a + 1, _j - _a:_j + _a + 1]
        img_vertical[_i, _j] = np.sum(np.multiply(_temp, Fy))

    plt.axis("off")
    plt.imshow(img_vertical, cmap = "gray", vmin = 0, vmax = 255)
    return img_vertical,


@app.cell
def __(mo):
    mo.md(rf"### Diagonal Edge Detection using list slicing:")
    return


@app.cell
def __(diagonal, img, m, n, np, plt):
    # Diagonal Edge Detection using list slicing:
    img_diagonal = np.zeros([m, n])
    _a = 1

    for _i in range(_a, m - _a):
      for _j in range(_a, n - _a):
        _temp = img[_i - _a:_i + _a + 1, _j - _a:_j + _a + 1]
        img_diagonal[_i, _j] = np.sum(np.multiply(_temp, diagonal))

    plt.axis("off")
    plt.imshow(img_diagonal, cmap = "gray", vmin = 0, vmax = 255)
    return img_diagonal,


@app.cell
def __(mo):
    mo.md(rf"### Showing the differences of Edge detection using various masks defined above")
    return


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
                mo.vstack([plt.imshow(image, cmap="gray", vmin = 0, vmax = 255), title], align="center")
                for image, title in zip(images, titles)
            ]
        )
    return show_images,


@app.cell
def __(img_diagonal, img_horizontal, img_vertical, show_images):
    radii = ["Horizonal", "Vertical", "Diagonal"]
    images = [img_horizontal, img_vertical, img_diagonal]
    # plt.figure(figsize = (5, 5))
    # for i in range(len(radii)):    
    #     plt.subplot(1, 5, i + 1)
    show_images(images, radii)
        # plt.title("Edge Detection using {}".format(radii[i]))
        # plt.xticks([])
        # plt.yticks([])
    return images, radii


@app.cell
def __(img_diagonal, img_horizontal, img_vertical):
    all_images = img_horizontal + img_vertical + img_diagonal
    return all_images,


@app.cell
def __():
    # plt.imshow(all_images, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(
    Fx,
    Fy,
    convolve,
    diagonal,
    img_diagonal,
    img_horizontal,
    img_vertical,
):
    # Built in Function:
    signal_x = convolve(img_horizontal, Fx, mode = "same")
    signal_y = convolve(img_vertical, Fy, mode = "same")
    signal_diagonal = convolve(img_diagonal, diagonal, mode = "same")
    return signal_diagonal, signal_x, signal_y


@app.cell
def __(mo):
    mo.md(rf"### Showing output using built in function for horizontal detection")
    return


@app.cell
def __(plt, signal_x):
    plt.axis("off")
    plt.imshow(signal_x, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(mo):
    mo.md(rf"### Showing output using built in function for vertical detection")
    return


@app.cell
def __(plt, signal_y):
    plt.axis("off")
    plt.imshow(signal_y, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(mo):
    mo.md(rf"### Showing output using built in function for diagonal detection")
    return


@app.cell
def __(plt, signal_diagonal):
    plt.axis("off")
    plt.imshow(signal_diagonal, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(mo):
    mo.md(rf"### Showing the differences of Edge detection using in-built masks from above")
    return


@app.cell
def __(show_images, signal_diagonal, signal_x, signal_y):
    _radii = ["Horizonal", "Vertical", "Diagonal"]
    _images = [signal_x, signal_y, signal_diagonal]
    # plt.figure(figsize = (5, 5))
    # for i in range(len(radii)):    
    #     plt.subplot(1, 5, i + 1)
    show_images(_images, _radii)
    return


@app.cell
def __(mo):
    mo.md(rf"### For different sizes of masks provided from user")
    return


@app.cell
def __(mo):
    # _size_of_mask = int(input("Enter the size of the Mask : "))
    slider = mo.ui.slider(start=1, stop=10, step=2, value=5, label="Enter size of mask")
    return slider,


@app.cell
def __(mo, slider):
    mo.hstack([slider, mo.md(f"You have requested for Mask of Size : {slider.value} x {slider.value}")])
    return


@app.cell
def __(img, m, n, np, plt, signal_x, slider):
    _size_of_mask = slider.value
    img_new = img.copy()
    print("You have requested for Mask of Size :  ", _size_of_mask ,"x", _size_of_mask)
    a = _size_of_mask//2

    for i in range(a, m - a):
        for j in range(a, n - a):
            temp = np.sum(img[i - a:i + a + 1, j - a:j + a + 1])
            img_new[i, j] = temp//_size_of_mask**2

    plt.axis("off")
    plt.imshow(signal_x, cmap = "gray", vmin = 0, vmax = 255)
    return a, i, img_new, j, temp


@app.cell
def __(mo):
    # callout = mo.callout("Conclusion: \
    # As we see from the image shown above and in the cell where the difference between three types of images are shown (horizontal, vertical and diagonal, we see that the above image where we applied Averaging filter to the original image and then applied convolution seemed to detect the horizontal images better than the one in which Averaging filter wasn't applied.", kind='success')

    text_input = mo.md(
        r'''
        ## Conclusion:
        
        After applying the Sobel operator with the averaging filter and comparing the results obtained in step a and b, the following observations can be made:

        - The Sobel operator enhances the edges in the image by highlighting the changes in intensity.
        
        - The averaging filter blurs the image and reduces the noise.
        
        - When the Sobel operator is applied after the averaging filter, the edges appear smoother and less pronounced compared to applying the Sobel operator directly on the original image.
        
        - The combined gradient image, |g_x+g_y|, obtained from the Sobel operator shows the overall intensity changes in the image.
        '''
    )
    mo.callout(text_input, kind="success")
    return text_input,


@app.cell
def __():
    # mo.vstack([callout], align="stretch", gap=0)
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## Observations

        After applying the Sobel operator with the averaging filter and comparing the results obtained in step a and b, the following observations can be made:

        - The Sobel operator enhances the edges in the image by highlighting the changes in intensity.
        - The averaging filter blurs the image and reduces the noise.
        - When the Sobel operator is applied after the averaging filter, the edges appear smoother and less pronounced compared to applying the Sobel operator directly on the original image.
        - The combined gradient image, |g_x+g_y|, obtained from the Sobel operator shows the overall intensity changes in the image.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
        ## References

        - The [Signal and Image Processing repository](https://github.com/Haleshot/Signal_Image_Processing/tree/main/Edge_Detection) and [website blog post](https://haleshot.github.io/post/edge_detection/).
        - [Sobel operator Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator#:~:text=The%20Sobel%20operator%2C%20sometimes%20called,creates%20an%20image%20emphasising%20edges.)
        - Assets and images for testing [here](https://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm)
        """
    )
    return


@app.cell
def __():
    # import libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    from scipy.signal import convolve
    import marimo as mo
    return convolve, cv2, mo, np, pd, plt


@app.cell
def __(cv2):
    img = cv2.imread("../assets/house.tif", 0)
    return img,


@app.cell
def __(img):
    print(type(img))
    return


@app.cell(hide_code=True)
def __():
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.sidebar(
        [
            mo.md("# Image Edge Detection"),
            mo.nav_menu(
                {
                    "#aim": f"{mo.icon('lucide:home')} Home",
                    "Page Overview": {
                        "#aim": "Aim",
                        "#table-of-contents": "Table of Contents",
                        "#software": "Software",
                        "#prerequisite": "Prerequisites",
                        "#outcome": "Outcomes",
                        "#theory": "Theory",
                    },
                    "Implementation": {
                        "#defining-masks-as-discussed-in-theory-above": "Defining Masks",
                        "#horizontal-edge-detection-using-list-slicing": "Horizontal Mask definition",
                        "#vertical-edge-detection-using-list-slicing": "Vertical Mask definition",
                        "#diagonal-edge-detection-using-list-slicing": "Diagonal Mask definition",
                        "#showing-the-differences-of-edge-detection-using-various-masks-defined-above" : "Comparison of user defined masks",
                        "#showing-the-differences-of-edge-detection-using-in-built-masks-from-above" : "Comparison of built-in defined masks",
                        "#for-different-sizes-of-masks-provided-from-user" : "Try it out!"
                    },
                    "#observations": "Observations",
                    "#references": "References",
                    "Links": {
                        "https://github.com/Haleshot/marimo-tutorials/blob/main/marimo-tutorials/Signal-Image-Processing/image-processing/edge-detection.py": f"{mo.icon('lucide:github')} GitHub",
                    },
                },
                orientation="vertical",
            ),
        ]
    )
    return


@app.cell
def __():
    # img = cv2.imread("./marimo-tutorials/signal-image-processing/assets/house.tif", 0)
    return


@app.cell
def __():
    # print(type(img))
    return


@app.cell
def __():
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


@app.cell
def __(mo):
    mo.sidebar(
        [
            mo.md("# Image Edge Detection"),
            mo.nav_menu(
                {
                    "#aim": f"{mo.icon('lucide:home')} Home",
                    "Page Overview": {
                        "#aim": "Aim",
                        "#table-of-contents": "Table of Contents",
                        "#software": "Software",
                        "#prerequisite": "Prerequisites",
                        "#outcome": "Outcomes",
                        "#theory": "Theory",
                    },
                    "Implementation": {
                        "#defining-masks-as-discussed-in-theory-above": "Defining Masks",
                        "#horizontal-edge-detection-using-list-slicing": "Horizontal Mask definition",
                        "#vertical-edge-detection-using-list-slicing": "Vertical Mask definition",
                        "#diagonal-edge-detection-using-list-slicing": "Diagonal Mask definition",
                        "#showing-the-differences-of-edge-detection-using-various-masks-defined-above" : "Comparison of user defined masks",
                        "#showing-the-differences-of-edge-detection-using-in-built-masks-from-above" : "Comparison of built-in defined masks",
                        "#for-different-sizes-of-masks-provided-from-user" : "Try it out!"
                    },
                    "#observations": "Observations",
                    "#references": "References",
                    "Links": {
                        "https://github.com/Haleshot/marimo-tutorials/blob/main/marimo-tutorials/Signal-Image-Processing/image-processing/edge-detection.py": f"{mo.icon('lucide:github')} GitHub",
                    },
                },
                orientation="vertical",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
