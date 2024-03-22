# Working Environment

This README provides instructions on setting up your local working environment with Python 3 along with Jupyter Notebook for the assignments.
We highly recommend to use a virtual environment such as venv or Anaconda.
Before you begin, ensure you have the following installed/working on your system:
- [Python 3](https://www.python.org/downloads/) (Python 3.8 or later)
- [venv](https://docs.python.org/3/library/venv.html) (optional, for using Python's built-in venv module)
- [Anaconda](https://www.anaconda.com) (optional, for using conda)

## Example Setup with Anaconda

### 1. Create a Conda Environment

Create a new conda environment with Python 3:

```bash
conda create --name seqlrn # or 'python3 -m venv seqlrn'
```

### 2. Activate the Environment

Activate the created environment:

```bash
conda activate seqlrn # or 'source seqlrn/bin/activate'
```

### 3. Installing Packages

Once the virtual environment is activated, you can install packages and work within it without affecting the system-wide Python installation. The required packages vary among assignments and require installation respectively.

You can install additional packages required for your assignment within the virtual environment:

```bash
pip install package_name # or 'conda install package_name'
```

Install the following packages to get started for the first assignment:

```bash
pip install numpy librosa matplotlib
```

## Using Jupyter Notebook

Hint: You can also use IDEs such as [Visual Studio Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) with Jupyter Notebook support.

### 1. Installing Jupyter

Before you begin, ensure you have either [Jupyter Notebook](https://jupyter.org/install) or [JupyterLab](https://jupyter.org/install) installed on your system:

```bash
pip install notebook # or 'pip install jupyterlab'
```

### 2. Running Jupyter Notebook

Open your terminal (or command prompt on Windows) and navigate to the assignment directory where you want to use the notebooks. Then, run the following command:

```bash
jupyter notebook # or 'jupyter lab'
```