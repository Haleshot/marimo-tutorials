<p align="center">
  <img src="https://i.ibb.co/SVcC6bb/final.png" alt="marimo-tutorials Community Banner">
</p>

<h2 align="center">Interactive Tutorials and Notebooks for marimo</h2>

<p align="center">
  <a href="https://marimo.io/c/@haleshot/marimo-tutorials"><img alt="Open in marimo" src="https://marimo.io/shield.svg"></a>
  <a href="https://github.com/Haleshot/marimo-tutorials/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://github.com/Haleshot/marimo-tutorials/blob/main/CONTRIBUTING.md"><img alt="Contributing" src="https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg"></a>
  <a href="https://github.com/Haleshot/marimo-tutorials/blob/main/CODE_OF_CONDUCT.md"><img alt="Code of Conduct" src="https://img.shields.io/badge/Code%20of%20Conduct-v1.0-ff69b4.svg"></a>
  <a href="https://github.com/Haleshot/marimo-tutorials"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Haleshot/marimo-tutorials?style=social"></a>
  <a href="https://discord.gg/rAhpfKwuaN"><img alt="Discord" src="https://img.shields.io/discord/1234567890?color=7389D8&label=Discord&logo=discord&logoColor=ffffff"></a>
</p>

# marimo-tutorials

Welcome to `marimo-tutorials`, a comprehensive collection of interactive notebooks and tutorials showcasing the power and versatility of [marimo](https://marimo.io), an innovative Python notebook framework. This repository serves as a hub for learning, exploration, and community engagement across various domains of computer science, data science, AI, and more.

## Table of Contents

- [About](#about)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Featured Notebooks](#featured-notebooks)
- [Community Spotlights](#community-spotlights)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

`marimo-tutorials` is designed to provide a rich learning experience for both beginners and advanced users of marimo. Our collection spans multiple disciplines, offering hands-on examples, in-depth tutorials, and real-world applications of marimo's capabilities.

## Repository Structure

```
marimo-tutorials/
│
├── .github/
├── apps/
├── docs/
├── env/
├── marimo-tutorials/
│   ├── artificial-intelligence/
│   │   └── recommendation-systems/
│   │       └── collaborative-filtering/
│   ├── assets/
│   ├── computer-science/
│   ├── Data-Science/
│   ├── Evolutionary-Computing/
│   ├── signal-image-processing/
│   │   ├── image-processing/
│   │   └── signal-processing/
│   ├── Software-Engineering/
│   └── Tutorials/
├── __pycache__/
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── init.py
├── LICENSE
├── preprocessed_books.csv
├── README.md
└── requirements.txt
```

Each subdirectory within `marimo-tutorials/` contains domain-specific notebooks, along with their respective README files and assets.

## Getting Started

To get started with the notebooks in this repository, you'll need to have marimo installed. We recommend using [`uv`](https://github.com/astral-sh/uv) for managing dependencies and running notebooks in isolated environments.

1. Install `uv` if you haven't already:
   ```shell
   pip install uv
   ```

2. Clone this repository:
   ```shell
   git clone https://github.com/Haleshot/marimo-tutorials.git
   cd marimo-tutorials
   ```

3. To run a notebook with its dependencies in an isolated environment:
   ```shell
   uvx marimo run --sandbox path/to/notebook.py
   ```

4. To edit a notebook:
   ```shell
   uvx marimo edit --sandbox path/to/notebook.py
   ```

### Isolated Environments with marimo

As highlighted in the [marimo blog](https://marimo.io/blog/sandboxed-notebooks), it's now possible to create marimo notebooks that have their package requirements serialized into them as a top-level comment. Given a notebook with inlined requirements, marimo can run it in an isolated virtual environment with a single command:

```shell
marimo edit --sandbox notebook.py
```

This creates a fresh virtual environment, or sandbox, and installs the dependencies before opening the notebook. marimo's opt-in package management features can even track imports and automatically add them to your notebook's inlined requirements. This means you can create and share standalone notebooks without shipping `requirements.txt` files alongside them.

## Featured Notebooks

Here are some featured notebooks from our collection:

### Artificial Intelligence

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="marimo-tutorials/artificial-intelligence/recommendation-systems/collaborative-filtering/">
        <img src="assets/collaborative-filtering.png" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="marimo-tutorials/artificial-intelligence/machine-learning/machine-learning/">
        <img src="assets/machine-learning.png" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="marimo-tutorials/artificial-intelligence/recommendation-systems/collaborative-filtering/">Collaborative Filtering</a>
    </td>
    <td>
      <a href="marimo-tutorials/artificial-intelligence/machine-learning/machine-learning/">Machine Learning</a>
    </td>
  </tr>
  <tr>
    <td>
      <a target="_blank" href="https://marimo.io/p/@placeholder/collaborative-filtering">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.io/p/@placeholder/machine-learning">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
  </tr>
</table>

### Data Science

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="marimo-tutorials/Data-Science/Exploratory-Data-Analysis/">
        <img src="assets/exploratory-data-analysis.png" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="marimo-tutorials/Data-Science/Exploratory-Data-Analysis/">Exploratory Data Analysis</a>
    </td>
  </tr>
  <tr>
    <td>
      <a target="_blank" href="https://marimo.io/p/@placeholder/exploratory-data-analysis">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
  </tr>
</table>

### Signal and Image Processing

<table border="0">
  <tr>
    <td>
      <a target="_blank" href="marimo-tutorials/signal-image-processing/image-processing/">
        <img src="assets/image-processing.png" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
    <td>
      <a target="_blank" href="marimo-tutorials/signal-image-processing/signal-processing/">
        <img src="assets/signal-processing.png" style="max-height: 150px; width: auto; display: block" />
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <a href="marimo-tutorials/signal-image-processing/image-processing/">Image Processing</a>
    </td>
    <td>
      <a href="marimo-tutorials/signal-image-processing/signal-processing/">Signal Processing</a>
    </td>
  </tr>
  <tr>
    <td>
      <a target="_blank" href="https://marimo.io/p/@placeholder/image-processing">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
    <td>
      <a target="_blank" href="https://marimo.io/p/@placeholder/signal-processing">
        <img src="https://marimo.io/shield.svg"/>
      </a>
    </td>
  </tr>
</table>

## Community Spotlights

While this repository hosts a variety of tutorials and notebooks, the official community spotlights are featured in a separate repository: [marimo-team/spotlights](https://github.com/marimo-team/spotlights). 

The spotlights repository features a project or marimo notebook from the community every Thursday. If you've created a notebook in this repository that you believe would be a good fit for the community spotlight, feel free to open an [issue](https://github.com/marimo-team/spotlights/issues) in the spotlights repository.

## Notebook Structure

All notebooks in this repository follow a uniform structure defined in the `init.py` file. This structure ensures consistency across tutorials and makes it easier for users to navigate and understand the content. Key elements of this structure include:

- Standard import statements
- Configuration settings
- Helper functions
- Main content sections
- Interactive elements

By adhering to this template, contributors can focus on creating high-quality content while maintaining a consistent user experience across all tutorials.

## Contributing

We welcome contributions from the community! Whether it's adding new tutorials, improving existing ones, or suggesting new features, your input is valuable. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, suggestions, or support, please open an issue in this repository or reach out to us (maintainers) through our community channels:

marimo socials:

- LinkedIn: [marimo LinkedIn](https://www.linkedin.com/company/marimo-io/)
- Twitter: [@marimo_io](https://twitter.com/marimo_io)
- Discord (proj-marimo-tutorials-channel): [marimo community](https://discord.gg/JE7nhX6mD8)
