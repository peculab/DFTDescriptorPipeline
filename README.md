# DFTDescriptorPipeline

## Overview

This repository provides a robust Python script, `descriptors/extractor_regr.py`, and accompanying examples for automated **feature selection and regression model development** based on Density Functional Theory (DFT) derived molecular descriptors.

The primary goal of this toolkit is to **identify the minimal, most statistically significant combination of physically meaningful molecular descriptors** that can accurately model and predict a target experimental property. It leverages a systematic search algorithm with feature grouping to ensure a balanced and interpretable model, which is critical for establishing quantitative structure-activity/property relationships (QSAR/QSPR) in computational chemistry.

### Key Features of `extractor_regr.py`:

* **Automated Feature Grouping:** Detects and groups related descriptors (e.g., Steric parameters, Electronic parameters, etc.) to enforce feature diversity and chemical interpretability in the resulting model.
* **Systematic Regression Search:** Performs an exhaustive or balanced search for the best regression model (using Ordinary Least Squares, **OLS**) based on user-defined constraints (e.g., maximum number of features, minimum $R^2$ threshold).
* **Balancing Constraint:** Allows for imposing limits on the number of features selected from each group, leading to more balanced and chemically intuitive models.
* **Dependency Management:** Automatically checks for and installs the necessary `morfeus-ml` library.
* **Visualization & Output:** Generates search results in a CSV file and plots the performance of the best-found model.

---

## Repository Structure

The project follows a clean, modular structure:

```tree
.
├── descriptors/
│   └── extractor_regr.py  # The core feature selection and regression script
├── examples/
│   ├── azoarene/azoarene_v2.ipynb
│   ├── heck_boronic_acids/heck_boronic_acids.ipynb
│   ├── indigo_aryl_alkyl/indigo_aryl_alkyl.ipynb
│   └── indigo_diaryl/indigo_diaryl.ipynb
└── README.md
```

## Getting Started: Running the Examples

The easiest way to understand and use this toolkit is by running the provided Jupyter Notebook examples, ideally within a cloud environment like Google Colab, which handles the necessary environment setup.

### Prerequisites

1.  **Python 3.x**
2.  **Required Libraries:** The script will automatically attempt to install `morfeus-ml`. Other standard libraries like `pandas`, `numpy`, `scikit-learn`, and `matplotlib` are also required.

### 1. Set up the Environment (Crucial Step)

Because the examples are designed to run in environments like Colab, you need to ensure the core script (`extractor_regr.py`) is accessible to the notebooks.

If you are running locally, place the `extractor_regr.py` file inside a directory named `descriptors/` at the root of your project, as shown in the structure above.

If you are running in **Google Colab**, you typically need to:

1.  Upload the `extractor_regr.py` file to your Colab environment.
2.  Create the necessary directory structure *within* the notebook execution:

    ```python
    !mkdir -p descriptors
    # In a Colab environment, you would then upload or move the file:
    # Example: !mv /content/extractor_regr.py /content/descriptors/
    # (Assuming you uploaded it to the root)
    ```
    *Note: When cloning the entire repository, this structure is automatically handled.*

### 2. Execute the Example Notebooks

The `examples/` directory contains four Jupyter notebooks demonstrating different chemical reaction systems:

| Example File | Description | Target Property |
| :--- | :--- | :--- |
| `azoarene/azoarene_v2.ipynb` | Regression analysis on a set of Azobenzene derivatives. | $\lambda_{max}$ or similar photochemical property. |
| `heck_boronic_acids/heck_boronic_acids.ipynb` | Modeling results from Heck or similar coupling reactions with boronic acids. | Reaction yield or selectivity. |
| `indigo_aryl_alkyl/indigo_aryl_alkyl.ipynb` | Analysis of Indigo derivatives with mixed aryl/alkyl substituents. | Spectroscopic or electronic property. |
| `indigo_diaryl/indigo_diaryl.ipynb` | Analysis of Indigo derivatives with diaryl substituents. | Spectroscopic or electronic property. |

**Steps to run an example:**

1.  **Open the Notebook:** Open any of the `.ipynb` files (e.g., `azoarene_v2.ipynb`) in your preferred Jupyter environment (local or Colab).
2.  **Execute Cells:** Run the cells sequentially. The first few cells typically handle setup, data loading (e.g., reading a `csv` file containing the property and DFT descriptors), and ensuring `extractor_regr.py` is ready.
3.  **Core Function Call:** Look for the cell that imports and calls the main function from your script:
    ```python
    from descriptors.extractor_regr import process_regression
    
    # ... prepare your data (df, target_col, etc.) ...
    
    df, results, best_model = process_regression(
        df_input=df,
        target='LogRate', # Replace with the actual target column in the notebook
        ... # other parameters like max_features, r2_threshold
    )
    ```
4.  **Review Results:** The output will display the search progress, the identified best model, and generate a `regression_search_results.csv` file and a plot visualizing the best model's fit.

By exploring these examples, you will see how to define the input DataFrame, specify the target column, and utilize the various parameters of the `process_regression` function within `extractor_regr.py` to tailor the feature selection process to your specific chemical system.
