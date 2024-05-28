import marimo

__generated_with = "0.6.10"
app = marimo.App()


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
    img = cv2.imread("./marimo-tutorials/Signal_Image_Processing/assets/house.tif", 0)
    return img,


@app.cell
def __(img):
    print(type(img))
    return


@app.cell
def __():
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return


@app.cell
def __(mo):
    mo.md(rf"# Defining Horizontal and vertical masks and adding the result of the two to form a diagonal mask.")
    return


@app.cell
def __(np):
    Fx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Fy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return Fx, Fy


@app.cell
def __(Fx):
    Fx
    return


@app.cell
def __(Fy):
    Fy
    return


@app.cell
def __(Fx, Fy):
    diagonal = Fx + Fy
    return diagonal,


@app.cell
def __(diagonal):
    diagonal
    return


@app.cell
def __(img, np):
    m, n = img.shape
    img_horizontal = np.zeros([m, n])
    return img_horizontal, m, n


@app.cell
def __(mo):
    mo.md(rf"# Horizontal Edge Detection using list slicing:")
    return


@app.cell
def __(Fx, img, img_horizontal, m, n, np, plt):
    # Horizontal Edge Detection using list slicing:
    _a = 1

    for _i in range(_a, m - _a):
      for _j in range(_a, n - _a):
        _temp = img[_i - _a:_i + _a + 1, _j - _a:_j + _a + 1]
        img_horizontal[_i, _j] = np.sum(np.multiply(_temp, Fx))

    plt.imshow(img_horizontal, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(mo):
    mo.md(rf"# Vertical Edge Detection using list slicing:")
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

    plt.imshow(img_vertical, cmap = "gray", vmin = 0, vmax = 255)
    return img_vertical,


@app.cell
def __(mo):
    mo.md(rf"# Diagonal Edge Detection using list slicing:")
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

    plt.imshow(img_diagonal, cmap = "gray", vmin = 0, vmax = 255)
    return img_diagonal,


@app.cell
def __(mo):
    mo.md(rf"# Showing the differences of Edge detection using various masks defined above (horizonal, vertical and diagonal).")
    return


@app.cell
def __(img_diagonal, img_horizontal, img_vertical, plt):
    radii = ["Horizonal", "Vertical", "Diagonal"]
    images = [img_horizontal, img_vertical, img_diagonal]
    plt.figure(figsize = (20, 10))
    for i in range(len(radii)):    
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap = "gray", vmin = 0, vmax = 255)
        plt.title("Edge Detection using {}".format(radii[i]))
        plt.xticks([])
        plt.yticks([])
    return i, images, radii


@app.cell
def __(img_diagonal, img_horizontal, img_vertical):
    all_images = img_horizontal + img_vertical + img_diagonal
    return all_images,


@app.cell
def __(all_images, plt):
    plt.imshow(all_images, cmap = "gray", vmin = 0, vmax = 255)
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
    mo.md(rf"# Showing output using built in function for horizontal detection")
    return


@app.cell
def __(plt, signal_x):
    plt.imshow(signal_x, cmap = "gray", vmin = 0, vmax = 255)

    return


@app.cell
def __(mo):
    mo.md(rf"# Showing output using built in function for vertical detection")
    return


@app.cell
def __(plt, signal_y):
    plt.imshow(signal_y, cmap = "gray", vmin = 0, vmax = 255)
    return


@app.cell
def __(mo):
    mo.md(rf"# Showing output using built in function for diagonal detection")
    return


@app.cell
def __(plt, signal_diagonal):

    plt.imshow(signal_diagonal, cmap = "gray", vmin = 0, vmax = 255)
    return


if __name__ == "__main__":
    app.run()
