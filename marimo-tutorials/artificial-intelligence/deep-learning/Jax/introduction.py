# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anywidget==0.9.13",
#     "jax==0.4.35",
#     "marimo",
#     "matplotlib==3.9.2",
#     "numpy==2.1.2",
#     "torch==2.5.0",
#     "torchvision==0.20.0",
#     "traitlets==5.14.3",
# ]
# ///

import marimo

__generated_with = "0.9.10"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# An Introduction to Jax""")
    return


@app.cell(hide_code=True)
def __(header_widget):
    header_widget
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## What is Jax?

        [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) is a high-performance numerical computing library that combines the ease of use of NumPy with the power of automatic differentiation and GPU/TPU acceleration. It is designed for high-performance machine learning research and provides capabilities for automatic differentiation, vectorization, and just-in-time compilation, making it suitable for deep learning and other scientific computing tasks.

        ## Why should we use Jax?

        JAX is beneficial for several reasons:

        1. **Automatic Differentiation**: JAX provides powerful automatic differentiation capabilities, which are essential for deep learning and other numerical computations.

        2. **Functional Programming**: It encourages a functional programming style, which can lead to cleaner and more maintainable code.

        3. **Composability**: JAX allows for easy composition of functions and transformations, making it flexible for a variety of applications.

        4. **Performance**: JAX can compile and run your code on GPUs and TPUs, offering high performance for large-scale computations.

        5. **Integration with Other Libraries**: You can mix JAX with other deep learning frameworks, using their strengths, such as data loaders from PyTorch or TensorFlow.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## How is JAX different from TensorFlow and PyTorch

        JAX differs from [TensorFlow](https://www.tensorflow.org/learn) and [PyTorch](https://pytorch.org/get-started/locally/) primarily in its approach to programming. JAX promotes a functional programming style, whereas TensorFlow and PyTorch are more object-oriented. This functional approach in JAX encourages writing clean and composable code, and it provides powerful function transformations. If you prefer functional programming or need automatic function transformations, JAX might be more suitable for you. Additionally, high-level neural network libraries like [Flax](https://flax.readthedocs.io/en/latest/index.html), built on top of JAX, offer functionally pure classes that are familiar to users of Keras or PyTorch.

        ## What are the limitations of Jax compared to PyTorch and TensorFlow?

        1. **Minimalistic Framework**: JAX is more minimalistic and does not include many high-level APIs out of the box, unlike PyTorch and TensorFlow, which come with comprehensive built-in functionalities like data loaders and high-level neural network modules.

        2. **Functional Programming Paradigm**: JAX enforces a functional programming style, which requires changes in coding habits, such as passing all parameters explicitly and avoiding side effects. This can be a limitation for those accustomed to the object-oriented style of PyTorch and TensorFlow.

        3. **Lack of Built-in High-Level APIs**: While JAX provides powerful low-level primitives, it lacks built-in high-level APIs for neural networks, which are available in PyTorch and TensorFlow. However, libraries like Flax and Haiku can be used to fill this gap.

        4. **No Built-in Data Loaders**: JAX does not provide its own data loaders, relying instead on external libraries like PyTorch or TensorFlow for this functionality.

        5. **Not Always the Best Option**: JAX might not be the best choice for certain applications, such as embedded deployments or when dealing with large legacy codebases that are already using other frameworks.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # The First Jax Program

        In this notebook, we’ll do a deep learning “hello world” exercise. We will build a simple neural network application demonstrating the JAX approach to building a deep learning model. It’s an image classification model that works on the [MNIST handwritten digit dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html). This project will introduce you to three of the main JAX transformations: [`grad()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html) for taking gradients, [`jit()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) for compilation, and [`vmap()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) for auto-vectorization.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Typical architecture of a Jax deep learning project

        1. Choose a dataset for your particular task. In our example, we use the [MNIST dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).
        2. Create a data loader to read your dataset and transform it into a sequence of batches. 
        3. Define a model that works with a single data point. JAX needs pure functions without state and side effects, so you separate model parameters from the function that applies the neural network to data. A neural network model is defined as
            + a set of model parameters 
            + a function performing computations given the parameters and input data
                + You can also use high-level neural network libraries on top of JAX for defining your models.
        4. Define a model that works with a batch of data. In JAX, this is typically done by auto-vectorizing the model function from step 3.
        5. Define a loss function that takes in model parameters and a batch of data. 
        6. Obtain gradients of the loss function with respect to the model parameters. The gradient function is evaluated on the model parameters and the input data and produces gradients for each model parameter. 
        7. Implement a gradient update step. The gradients are used to update the model parameters with some gradient descent procedure. You can update model parameters directly  or use a special optimizer from a separate library.
        8. Implement the complete training loop.
        9. The model is compiled to the target hardware platform using [JIT compilation](https://jax.readthedocs.io/en/latest/jit-compilation.html). __This step may significantly speed up your computations__.
        10. You may also distribute model training across a cluster of computers.
        11. After running the training loop for several epochs, you get a trained model (__an updated set of the parameters__) that can be used for predictions or any other task you designed the model for.
        12. Save your trained model.
        13. Use the model. Depending on your case, you can deploy the model using some production infrastructure or just load it and perform calculations without specialized infrastructure.
            - For saving and restoring model weights, you may use standard Python tools like [`pickle`](https://docs.python.org/3/library/pickle.html), or safer solutions like [`safetensors`](https://huggingface.co/docs/safetensors/index). Higher-level libraries on top of JAX may also provide their own means for loading/saving models.
            - For deployment, there are several options available. For example, you can convert your model to `TensorFlow` or `TFLite`, and use their well-developed ecosystem for model deployment.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Loading and preparing the dataset

        The total number of images in MNIST is 70,000. The dataset provides a train/test split with 60,000 images in the train and 10,000 in the test part. Images are grayscale with a size of 28x28 pixels.

        We have two classes that encapsulate different preprocessing steps.

        1. `FlattenAndCast`: Flattens the image to a 1D array and casts it to float32. This is necessary for models that expect flattened input vectors (like basic neural networks).
        2. `Normalize`: Normalizes the pixel values by dividing by 255.0 to get values in the range [0, 1]. This helps models train more efficiently by scaling the input features.

        We use [`transforms.Compose`](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#compose) from [`torchvision`](https://pytorch.org/vision/stable/index.html#torchvision) to chain both transformation steps, so the dataset automatically applies them whenever an image is loaded.
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import numpy as np
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader


    class FlattenAndCast:
        """Flatten the input to a 1D array and cast it to float32."""

        def __call__(self, pic):
            return np.ravel(np.array(pic, dtype=np.float32))


    class Normalize:
        """Resize image and normalize pixel values to [0, 1]."""

        def __call__(self, img):
            return img / 255.0


    # Compose both transformations together
    transform = transforms.Compose([FlattenAndCast(), Normalize()])
    return (
        DataLoader,
        FlattenAndCast,
        MNIST,
        Normalize,
        np,
        transform,
        transforms,
    )


@app.cell(hide_code=True)
def __(DataLoader, MNIST, transform):
    # Load MNIST dataset with the composed transform
    mnist_train = MNIST(
        "/tmp/mnist/", train=True, download=True, transform=transform
    )
    mnist_test = MNIST(
        "/tmp/mnist/", train=False, download=True, transform=transform
    )
    # Create DataLoaders for training and testing
    train_loader = DataLoader(
        mnist_train, batch_size=32, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        mnist_test, batch_size=32, shuffle=False, num_workers=0
    )
    return mnist_test, mnist_train, test_loader, train_loader


@app.cell(hide_code=True)
def __(mnist_train):
    HEIGHT = 28
    WIDTH = 28
    CHANNELS = 1
    NUM_PIXELS = HEIGHT * WIDTH * CHANNELS
    NUM_LABELS = len(mnist_train.classes)
    return CHANNELS, HEIGHT, NUM_LABELS, NUM_PIXELS, WIDTH


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Examples from the MNIST dataset. Each handwritten image has a class label displayed on top of it.""")
    return


@app.cell(hide_code=True)
def __(mnist_train, np):
    import matplotlib.pyplot as plt


    def display_mnist_examples(mnist_dataset):
        plt.rcParams["figure.figsize"] = [10, 5]

        rows = 3
        cols = 10
        num = rows * cols

        i = 0
        fig, ax = plt.subplots(rows, cols)
        for image, label in zip(
            mnist_dataset.data.numpy()[:num],
            mnist_dataset.targets.numpy()[:num],
            strict=True,
        ):
            ax[int(i / cols), i % cols].axis("off")
            ax[int(i / cols), i % cols].set_title(str(label))
            ax[int(i / cols), i % cols].imshow(
                np.reshape(image, (28, 28)), cmap="gray"
            )
            i += 1

        return plt.gca()


    display_mnist_examples(mnist_train)
    return display_mnist_examples, plt


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## A simple neural network in JAX

        Here, we use a feed-forward neural network known as a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP).

        Our solution will be a simple two-layer MLP, which takes flattened 28x28 images as input, processing them through one hidden layer (512 neurons) to extract complex patterns, and outputting a probability distribution across 10 possible digit classes.

        The process of creating a neural network in JAX is different from the same process in PyTorch/TensorFlow in several ways, namely in using **random number generators for model parameter initialization** and in **how model code and parameters are structured**.

        - **random number generators** in JAX require their state to be provided externally to be functionally pure (the `PRNGKey` plays this role).
        - the **forward pass function** must also be stateless and functionally pure, so __model parameters are passed there as some input data__. This is not the case in PyTorch and Tensorflow, where model parameters are stored inside some objects together with code.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Neural network initialization

        Before training a neural network, we need to initialize all the weights and biases with random numbers.
        """
    )
    return


@app.cell(hide_code=True)
def __(random):
    LAYER_SIZES = [28 * 28, 512, 10]
    PARAM_SCALE = 0.01


    def init_network_params(sizes, key=random.PRNGKey(0), scale=1e-2):
        """
        Initialize the weights and biases for a fully-connected neural network.

        Args:
            sizes (list): A list of integers representing the number of neurons in
                          each layer, including input, hidden and output layers.
            key (jax.random.PRNGKey): A PRNG key to generate random numbers.
            scale (float): The scaling factor for initializing parameters,
                           ensuring small initial values.

        Returns:
            list: A list of tuples (weights, biases) for each layer.
                  - weights: A matrix of shape (n, m) for each layer, where n is
                             the number of neurons in the current layer and m is
                             the number of neurons in the previous layer.
                  - biases: A vector of shape (n,) for each layer.
        """

        def init_layer_params(m, n, key):
            """Initialize weights and biases for a dense layer."""
            w_key, b_key = random.split(key)
            weights = scale * random.normal(w_key, (n, m))
            biases = scale * random.normal(b_key, (n,))
            return weights, biases

        keys = random.split(key, len(sizes) - 1)  # One key per layer
        return [
            init_layer_params(m, n, k)
            for m, n, k in zip(sizes[:-1], sizes[1:], keys)
        ]


    init_params = init_network_params(
        LAYER_SIZES, random.PRNGKey(0), scale=PARAM_SCALE
    )
    return LAYER_SIZES, PARAM_SCALE, init_network_params, init_params


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        JAX implements random number generators that are purely functional, and you must provide each call of a randomized function with an RNG state, which is called a key here, and you should use every key only once, so each time you need a new key, you split an old key into the required amount of new ones.
        """
    ).callout(kind="warn")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Neural network forward pass

        Then, you need a function that performs all the neural network computations, i.e., the forward pass. 

        This is almost the same function as `forward(self, x)` in PyTorch or `call(self, x)` in Tensorflow/Keras, except how you pass model parameters there. In JAX, it looks like `predict(params, x)`. The function name doesn’t matter; you may call it `forward()`, `call()`, or anything else. What matters is that __it’s not a class function but a function with model parameters passed as a function parameter__. So, you have __params instead of self__.

        For the forward pass, we already have initial values for weights and biases. The only missing part is the [activation function](https://en.wikipedia.org/wiki/Activation_function). We will use the popular [Swish activation function](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.swish.html) from the `jax.nn` library.

        $$
        \text{swish}(x) = \frac{x}{1 + e^{-x}}
        $$

        Here, we develop a forward-pass function, often called a predict function. It takes an image to classify and performs all the forward-pass computations to produce activations on the output layer neurons. The neuron with the highest activation determines the class of the input image.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp):
    from jax.nn import swish


    def predict(params, image):
        """Function for per-example predictions."""
        activations = image
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = swish(outputs)

        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits
    return predict, swish


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""In JAX, you typically have two functions for your neural networks: one for initializing parameters and another one for applying your neural network to some input data. The first function returns parameters as some data structure (here, a list of arrays; later, it will be a special data structure called [`PyTree`](https://jax.readthedocs.io/en/latest/pytrees.html)). The second one takes in parameters and the data and returns the result of applying the neural network to the data."""
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(init_params, mo, predict, random):
    random_flattened_image = random.normal(random.PRNGKey(1), (28 * 28 * 1,))
    with mo.redirect_stdout():
        print("generate a random image and test the func")
        print(predict(init_params, random_flattened_image))
    return (random_flattened_image,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## vmap: auto-vectorizing calculations to work with batches
        What if input a batch of images to our `predict` func?
        """
    )
    return


@app.cell(hide_code=True)
def __(init_params, mo, predict, random):
    with mo.redirect_stdout():
        random_flattened_images = random.normal(
            random.PRNGKey(1), (32, 28 * 28 * 1)
        )
        try:
            predict(init_params, random_flattened_images)
        except TypeError as e:
            print("TypeError:", e)
    return (random_flattened_images,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        JAX provides the `vmap()` transformation that transforms a function that works with a single element into one that can work on batches. This is the way you’ll probably use most of the time in JAX, as it is the most convenient one and produces excellent performance. 

        The `in_axes` parameter controls which input array axes to map over (or to vectorize). __Its length must equal the number of positional arguments of the function__. `None` indicates we need not map any axis, and in our example, it corresponds to the first parameter of the `predict()` function, which is `params`. This parameter stays the same for any forward pass, so we do not need to batch over it (yet, if we used separate neural network weights for each call, we’d use this option). The second element of the `in_axes` tuple corresponds to the second parameter of the `predict()` function, `image`. The zero value means we want to __batch over the first (zeroth) dimension__, the dimension that contains different images. In the hypothetical case when the batch dimension will be in another position in a tensor, we’d change this number to the proper index.
        """
    )
    return


@app.cell(hide_code=True)
def __(predict, vmap):
    batched_predict = vmap(predict, in_axes=(None, 0))
    return (batched_predict,)


@app.cell(hide_code=True)
def __(batched_predict, init_params, mo, random_flattened_images):
    with mo.redirect_stdout():
        print(
            "Now, we can apply our modified function to a batch and produce the correct outputs"
        )
        print("Output shape:")
        print(batched_predict(init_params, random_flattened_images).shape)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Autodiff: how to calculate gradients without knowing about derivatives

        We typically use a [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) procedure to train a neural network. While the general idea is still the same as for PyTorch/Tensorflow, this process significantly differs from those frameworks. 

        We will use an almost straightforward mini-batch gradient descent with an exponentially decaying learning rate and without momentum, similar to a vanilla stochastic gradient descent (SGD) optimizer in any deep learning framework.

        ### Loss function

        We need a [loss function](https://en.wikipedia.org/wiki/Loss_function) to evaluate our current set of parameters on the training dataset. The loss function calculates the discrepancy between the model prediction and the ground truth values from the training dataset labels. There are many different loss functions for specific machine learning tasks, and we will use a simple loss function suitable for multi-class classification, the [categorical cross-entropy function](https://gombru.github.io/2018/05/23/cross_entropy_loss/).

        It is almost the same loss function as for other frameworks but with the same difference as for the forward pass function -- it should be functionally pure, and we need to provide it with the model parameters.
        """
    )
    return


@app.cell(hide_code=True)
def __(batched_predict, jnp):
    from jax.nn import logsumexp


    def loss(params, images, targets):
        """Categorical cross entropy loss function."""
        logits = batched_predict(params, images)
        log_preds = logits - logsumexp(logits)
        return -jnp.mean(targets * log_preds)
    return logsumexp, loss


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Gradient update step

        We have to calculate the gradients of the loss function with respect to the model parameters based on the current batch of data. Here comes the `grad()` transformation. The transformation takes in a function (here, the loss function). It creates a function that __evaluates__ the gradient of the loss function with respect to a specific parameter, which is, by default, the first parameter of the function (here, the `params`).

        Here is an important distinction from other frameworks like TensorFlow and PyTorch. In those frameworks, you usually get gradients after performing the forward pass, and the framework tracks all the operations being done on the tensors of interest. JAX uses a different approach. __It transforms your function and generates another function that calculates gradients__. And then, you calculate the gradients by providing all the relevant parameters, the neural network weights, and the data into this function.

        Here, we calculate the gradients and then __update all the parameters in the direction opposite to the gradient__ (hence, the minus sign in the weight update formulas). All the gradients are scaled with the learning rate parameter that depends on the number of epochs (one epoch is a complete pass through the training set). We made an exponentially decaying learning rate, so for later epochs, the learning rate will be lower than for the earlier ones.
        """
    )
    return


@app.cell(hide_code=True)
def __(loss):
    INIT_LR = 1.0
    DECAY_RATE = 0.95
    DECAY_STEPS = 5


    from jax import value_and_grad


    def update(params, x, y, epoch_number):
        loss_value, grads = value_and_grad(loss)(params, x, y)
        lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
        return [
            (w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)
        ], loss_value
    return DECAY_RATE, DECAY_STEPS, INIT_LR, update, value_and_grad


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        In the case you don't track loss values, use `grad` would be enough, and `update` should become to

        ```python
        from jax import grad


        def update(params, x, y, epoch_number):
          grads = grad(loss)(params, x, y)
          lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
          return [(w - lr * dw, b - lr * db)
                  for (w, b), (dw, db) in zip(params, grads)]
        ```
        """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Training loop

        Now, we need to run a training loop for a specified number of epochs. To do it, we need a few more utility functions to calculate accuracy and add some logging to track all the relevant information during training
        """
    )
    return


@app.cell(hide_code=True)
def __(NUM_PIXELS, batched_predict, jnp):
    import time
    from jax.nn import one_hot

    num_epochs = 5


    def batch_accuracy(params, images, targets):
        images, targets = images.numpy(), targets.numpy()
        images = jnp.reshape(images, (len(images), NUM_PIXELS))
        predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
        return jnp.mean(predicted_class == targets)


    def accuracy(params, data):
        accs = []
        for images, targets in data:
            accs.append(batch_accuracy(params, images, targets))
        return jnp.mean(jnp.array(accs))
    return accuracy, batch_accuracy, num_epochs, one_hot, time


@app.cell(hide_code=True)
def __(
    NUM_LABELS,
    NUM_PIXELS,
    accuracy,
    init_params,
    jnp,
    one_hot,
    test_loader,
    time,
    train_loader,
    update,
):
    def train(num_epochs, initial_params):
        params = init_params.copy()

        for epoch in range(num_epochs):
            start_time = time.time()
            losses = []
            for x, y in train_loader:
                x, y = x.numpy(), y.numpy()
                x = jnp.reshape(x, (len(x), NUM_PIXELS))
                y = one_hot(y, NUM_LABELS)
                params, loss_value = update(params, x, y, epoch)
                losses.append(loss_value)
            epoch_time = time.time() - start_time

            start_time = time.time()
            train_acc = accuracy(params, train_loader)
            test_acc = accuracy(params, test_loader)
            eval_time = time.time() - start_time
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Eval in {:0.2f} sec".format(eval_time))
            print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))
    return (train,)


@app.cell(hide_code=True)
def __(init_params, mo, num_epochs, train):
    with mo.redirect_stdout():
        train(num_epochs, init_params)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## JIT: compiling your code to make it faster

        We can make our solution faster with JIT compilation and acceleration provided by XLA. 

        Compiling our functions is easy. We can use either `jit()` function transformation or `@jit` annotation. We will use the latter.

        Here, we compile the two most resource-heavy functions, the `update()` and `batch_accuracy()` functions. All that you need is to add the `@jit` annotation before the function definitions:
        """
    )
    return


@app.cell(hide_code=True)
def __(
    DECAY_RATE,
    DECAY_STEPS,
    INIT_LR,
    NUM_LABELS,
    NUM_PIXELS,
    batched_predict,
    init_params,
    jit,
    jnp,
    loss,
    one_hot,
    test_loader,
    time,
    train_loader,
    value_and_grad,
):
    @jit
    def jit_update(params, x, y, epoch_number):
        loss_value, grads = value_and_grad(loss)(params, x, y)
        lr = INIT_LR * DECAY_RATE ** (epoch_number / DECAY_STEPS)
        return [
            (w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)
        ], loss_value


    @jit
    def jit_batch_accuracy(params, images, targets):
        images = jnp.reshape(images, (len(images), NUM_PIXELS))
        predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
        return jnp.mean(predicted_class == targets)


    def jit_accuracy(params, data):
        accs = []
        for images, targets in data:
            images, targets = images.numpy(), targets.numpy()
            accs.append(jit_batch_accuracy(params, images, targets))
        return jnp.mean(jnp.array(accs))


    def jit_train(num_epochs, initial_params):
        params = init_params.copy()

        for epoch in range(num_epochs):
            start_time = time.time()
            losses = []
            for x, y in train_loader:
                x, y = x.numpy(), y.numpy()
                x = jnp.reshape(x, (len(x), NUM_PIXELS))
                y = one_hot(y, NUM_LABELS)
                params, loss_value = jit_update(params, x, y, epoch)
                losses.append(loss_value)
            epoch_time = time.time() - start_time

            start_time = time.time()
            train_acc = jit_accuracy(params, train_loader)
            test_acc = jit_accuracy(params, test_loader)
            eval_time = time.time() - start_time
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Eval in {:0.2f} sec".format(eval_time))
            print("Training set loss {}".format(jnp.mean(jnp.array(losses))))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))
    return jit_accuracy, jit_batch_accuracy, jit_train, jit_update


@app.cell(hide_code=True)
def __(init_params, jit_train, mo, num_epochs):
    with mo.redirect_stdout():
        jit_train(num_epochs, init_params)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""the speedup is evident :)""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Saving and deploying the model

        When you are done with your model, you first need to save it, and then, depending on your case, you may either want to deploy it somewhere or just occasionally load it and use it for some ad-hoc calculations without the need to have production infrastructure.

        JAX does not provide any special tool for saving a model because, technically speaking, JAX does not know anything about a model. It’s just a framework for high-performance tensor calculations. 

        The simplest way to save the model is by using standard Python tools like `pickle`. 

        In the following listing, we save and restore model parameters stored as a nested Python data structure called a `PyTree`.


        ```python
        import pickle

        model_weights_file = 'mlp_weights.pickle'

        with open(model_weights_file, 'wb') as file:
            pickle.dump(params, file)

        with open(model_weights_file, 'rb') as file:
            restored_params = pickle.load(file)
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Summary

        - JAX lacks its own data loader, so external loaders from PyTorch or TensorFlow can be used.
        - Neural network parameters are passed as external parameters, not stored in objects like in TensorFlow/PyTorch.
        - Model parameters are stored in a nested Python data structure called PyTree.
        - JAX's random number generators are stateless, requiring an external state (PRNGKey).
        - The `vmap()` transformation allows functions to work on batches.
        - Gradients can be calculated using the `grad()` function, and both value and gradient can be obtained with `value_and_grad()`.
        - The `jit()` transformation compiles functions with XLA for optimized GPU/TPU execution.
        - Model weights can be saved and loaded using Python libraries like `pickle` or `safetensors` from HuggingFace.
        - You need to use pure functions with no internal state and side effects for your transformations to work correctly

        To conclude this introduction to JAX, if computational performance is crucial for you, if you value functional programming and clean code, if you are involved in deep learning research, or if you are interested in controlling every aspect of your code, JAX is a great option for you! At the same time, JAX is not always the best option for every problem, and in some cases, like embedded deployments or vast legacy codebases that use other frameworks, PyTorch or TensorFlow could be a better option.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # References

        - [Deep Learning with Jax, Chapter 2](https://livebook.manning.com/book/deep-learning-with-jax/chapter-2/v-11/47)
        - [Training a simple neural network, with PyTorch data loading](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html#hyperparameters)
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
            "Title": "An Introduction to Jax",
            "Author": "Eugene",
            "Date": "2024-10-30",
            "Version": "0.1",
            "Description": "This notebook contains an introduction to Jax, a high-performance numerical computing library, with a starter deep-learning project",
            "Keywords": "deep learning, numerical computing",
            "Data Sources": "MNIST",
            "Tools Used": "Python, Jax",
        }
    )
    return (header_widget,)


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __():
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax import random
    return grad, jit, jnp, random, vmap


if __name__ == "__main__":
    app.run()
