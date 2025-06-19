![PyPI version](https://img.shields.io/pypi/v/beaverfe)
[![PyPI Downloads](https://static.pepy.tech/badge/beaverfe)](https://pepy.tech/projects/beaverfe)

![Beaver FE Logo](assets/beaverfe-logo.png)

---

# **Beaver FE**

*A Versatile Toolkit for Automated Feature Engineering in Machine Learning*

**Beaver FE** is a Python library that streamlines feature engineering for machine learning. It provides robust tools for preprocessing tasks such as scaling, normalization, feature creation (e.g., binning, mathematical operations), and encoding. It improves data quality and boosts model performance with minimal manual effort.

## 📌 Table of Contents
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
  - [Automated Feature Engineering](#automated-feature-engineering)
  - [Manual Transformations](#manual-transformations)
  - [Saving and Loading Pipelines](#saving-and-loading-pipelines)
- [Benchmark Results](#benchmark-results) 
- [Core API](#core-api)
- [Available Transformations](#available-transformations)
  - [Missing Values & Outliers](#missing-values-and-outliers)
    - [Missing Values Indicator](#missing-values-indicator)
    - [Missing Values Handler](#missing-values-handler)
    - [Handle Outliers](#handle-outliers)
  - [Data Distribution & Scaling](#data-distribution-and-scaling)
    - [Non-Linear Transformation](#non-linear-transformation)
    - [Quantile Transformations](#quantile-transformations)
    - [Scale Transformations](#scale-transformations)
    - [Normalization](#normalization)
  - [Numerical Features](#numerical-features)
    - [Spline Transformations](#spline-transformations)
    - [Numerical Binning](#numerical-binning)
    - [Mathematical Operations](#mathematical-operations)
  - [Categorical Features](#categorical-features)
    - [Categorical Encoding](#categorical-encoding)
  - [Periodic Features](#periodic-features)
    - [Date Time Transforms](#date-time-transforms)
    - [Cyclical Features Transforms](#cyclical-features-transforms)
  - [Features Reduction](#features-reduction)
    - [Column Selection](#column-selection)
    - [Dimensionality Reduction](#dimensionality-reduction)
- [Contributing](#contributing)
- [License](#license)


<a id="getting-started"></a>
## 🚀 Getting Started

Install Beaver FE using pip:

```bash
pip install beaverfe
```

<a id="usage-examples"></a>
## 📖 Usage Examples

<a id="automated-feature-engineering"></a>
### 🤖 Automated Feature Engineering

Automatically optimize feature transformations using a given model and metric:

```python
from beaverfe import auto_feature_pipeline, BeaverPipeline
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
transformations = auto_feature_pipeline(x, y, model, scoring="accuracy", direction="maximize")

bfe = BeaverPipeline(transformations)
x_train = bfe.fit_transform(x_train, y_train)
x_test = bfe.transform(x_test, y_test)
```


<a id="manual-transformations"></a>
### 🔧 Manual Transformations

```python
from beaverfe import BeaverPipeline
from beaverfe.transformations import (
    MathematicalOperations,
    NumericalBinning,
    OutliersHandler,
    ScaleTransformation,
)

# Define transformations
transformations = [
    OutliersHandler(
        transformation_options={
            "sepal length (cm)": ("median", "iqr"),
            "sepal width (cm)": ("cap", "zscore"),
        },
        thresholds={
            "sepal length (cm)": 1.5,
            "sepal width (cm)": 2.5,
        },
    ),
    ScaleTransformation(
        transformation_options={
            "sepal length (cm)": "min_max",
            "sepal width (cm)": "robust",
        },
        quantile_range={
            "sepal width (cm)": (25.0, 75.0),
        },
    ),
    NumericalBinning(
        transformation_options={
            "sepal length (cm)": ("uniform", 5),
        }
    ),
    MathematicalOperations(
        operations_options=[
            ("sepal length (cm)", "sepal width (cm)", "add"),
        ]
    ),
]

bfe = BeaverPipeline(transformations)

x_train = bfe.fit_transform(x_train, y_train)
x_test = bfe.transform(x_test, y_test)
```

<a id="saving-and-loading-pipelines"></a>
### 💾 Saving and Loading Transformations

Save your pipeline for reuse across sessions:

```python
import pickle
from beaverfe import BeaverPipeline

bfe = BeaverPipeline(transformations)

# Save pipeline parameters
with open("beaverfe_transformations.pkl", "wb") as f:
    pickle.dump(bfe.get_params(), f)

# Load pipeline parameters
with open("beaverfe_transformations.pkl", "rb") as f:
    params = pickle.load(f)

bfe.set_params(**params)
```

---

<a id="benchmark-results"></a>
## 📊 Benchmark Results

Beaver FE was evaluated on several datasets and models to assess its impact on model performance. The table below compares baseline accuracy versus accuracy after applying Beaver FE transformations:

| Dataset | Model                | Baseline | BeaverFE | Improvement |
|---------|----------------------|----------|----------|-------------|
| **adult** |                      |          |          |             |
|          | LDA                  | 0.848    | 0.905    | +6.72%      |
|          | LogisticRegression   | 0.822    | 0.900    | +9.49%      |
|          | XGBoost              | 0.921    | 0.923    | +0.22%      |
| **bank**  |                      |          |          |             |
|          | LDA                  | 0.874    | 0.911    | +4.23%      |
|          | LogisticRegression   | 0.854    | 0.909    | +6.44%      |
|          | XGBoost              | 0.927    | 0.929    | +0.22%      |
| **credit**|                      |          |          |             |
|          | LDA                  | 0.717    | 0.761    | +6.14%      |
|          | LogisticRegression   | 0.696    | 0.747    | +7.33%      |
|          | XGBoost              | 0.760    | 0.757    | -0.39%      |


> 🚨 *Note*: A slight decrease was observed in one case. This shows that although Beaver FE generally improves performance, results may vary depending on the model and dataset.

![Benchmark Performance Chart](assets/benchmark_results.png)

---

<a id="core-api"></a>
## 🧩 Core API

### **auto_feature_pipeline**

Automatically finds and applies optimal transformations to improve model performance.

```python
from beaverfe import auto_feature_pipeline
```

#### **Parameters:**

- `X` (`np.ndarray`): Feature matrix.
- `y` (`np.ndarray`): Target variable.
- `model`: A machine learning model implementing a `fit` method.
- `scoring` (`str`): Evaluation metric (e.g., `"accuracy"`, `"f1"`, `"roc_auc"`).
- `direction` (`str`, optional): Optimization direction: `"maximize"` or `"minimize"`. Default is `"maximize"`.
- `cv` (`int` or callable, optional): Cross-validation strategy (e.g., number of folds or a custom splitter). Default is `None`.
- `groups` (`np.ndarray`, optional): Group labels for cross-validation. Useful for grouped CV.
- `verbose` (`bool`, optional): Whether to display progress logs. Default is `True`.

#### **Transformation Flags:**

Each step of the pipeline can be selectively enabled or disabled.

* `preprocessing` (`bool`, default=`True`):
  Applies initial cleaning steps, including:

  * Missing value indicators and imputation
  * Outlier detection and handling
  * Extraction of datetime features

* `feature_generation` (`bool`, default=`True`):
  Applies feature creation techniques such as:

  * Spline transformations
  * Binning of numeric features
  * Arithmetic operations
  * Categorical encodings
  * Cyclical date transformations

* `normalization` (`bool`, default=`True`):
  Transforms feature distributions using:

  * Non-linear transformations
  * Quantile transformations
  * Normalization/scaling

* `dimensionality_reduction` (`bool`, default=`True`):
  Reduces feature space through:

  * Feature selection (based on performance)
  * Projection-based dimensionality reduction (e.g., PCA)

#### **Execution Order:**

Transformations are applied in the following order:

1. Preprocessing (missing values, outliers, datetime)
2. Feature Generation (splines, binning, math ops, encodings)
3. Normalization (non-linear transforms, quantiles, scaling)
4. Dimensionality Reduction (column selection, PCA)

#### **Returns:**

* `List[dict]`: A list of transformation configurations that can be passed to `BeaverPipeline`.

---

### **BeaverPipeline**

A wrapper to apply a sequence of transformations.

```python
from beaverfe import BeaverPipeline
```

#### **Constructor Parameters:**
- `transformations` (`list`, optional): List of transformation objects or dictionaries.

#### **Public Methods:**

- `fit(X, y=None)`  
    Fits each transformation in the pipeline to the dataset.
    - **Returns:** `self`

- `transform(X, y=None)`  
    Applies each fitted transformation in sequence.
    - **Returns:** Transformed feature matrix (`np.ndarray` or `pd.DataFrame`)

- `fit_transform(X, y=None)`  
    Combines `fit` and `transform` for each transformation.
    - **Returns:** Transformed feature matrix.

- `get_params(deep=True)`  
    Retrieves the parameters of the pipeline (mainly the transformations).
    - **Returns:** Dictionary of parameters.

- `set_params(**params)`  
    Sets or updates the pipeline parameters.
    - **Returns:** `self`

---

<a id="available-transformations"></a>
## 🔍 Available Transformations

Grouped by feature type or transformation category:

<a id="missing-values-and-outliers"></a>
### 📌 Missing Values & Outliers

#### **Missing Values Indicator**

Adds binary flags for missing values.

- Parameters:
    - `features`: List of column names to check for missing values. If None, all columns are considered.

```python
from beaverfe.transformations import MissingValuesIndicator

MissingValuesIndicator(
    features=[
        'sepal width (cm)',
        'petal length (cm)',
    ]
)
```

#### **Missing Values Handler**

Fills missing values.

- Parameters:
    - `transformation_options`: Dictionary that specifies the handling strategy for each column. Options: `fill_0`, `mean`, `median`, `most_frequent`, `knn`.
    - `n_neighbors`: Number of neighbors for K-Nearest Neighbors imputation (used with `knn`).

```python
from beaverfe.transformations import MissingValuesHandler

MissingValuesHandler(
    transformation_options={
        'sepal width (cm)': 'knn',
        'petal length (cm)': 'mean',
        'petal width (cm)': 'most_frequent',
        
    },
    n_neighbors= {
        'sepal width (cm)': 5,
    }
)
```

#### **Handle Outliers**

Detects and mitigates outliers using methods like `iqr`, `zscore`, `lof`, or `iforest`.

- Parameters:
    - `transformation_options`: Dictionary specifying the handling strategy. The strategy is a tuple where the first element is the action (`cap` or `median`) and the second is the method (`iqr`, `zscore`, `lof`, `iforest`).
    - `thresholds`: Dictionary with thresholds for `iqr` and `zscore` methods.
    - `lof_params`: Dictionary specifying parameters for the LOF method.
    - `iforest_params`: Dictionary specifying parameters for Isolation Forest.

```python
from beaverfe.transformations import OutliersHandler

OutliersHandler(
    transformation_options={
        'sepal length (cm)': ('median', 'iqr'),
        'sepal width (cm)': ('cap', 'zscore'),
        'petal length (cm)': ('median', 'lof'),
        'petal width (cm)': ('median', 'iforest'),
    },
    thresholds={
        'sepal length (cm)': 1.5,
        'sepal width (cm)': 2.5,    
    },
    lof_params={
        'petal length (cm)': {
            'n_neighbors': 20,
        }
    },
    iforest_params={
        'petal width (cm)': {
            'contamination': 0.1,
        }
    }
)
```

<a id="data-distribution-and-scaling"></a>
### 📌 Data Distribution & Scaling

#### **Non-Linear Transformation**

Applies logarithmic, exponential, or Yeo-Johnson transformations.

- Parameters:
    - `transformation_options`: A dictionary specifying the transformation to be applied for each column. Options include: `log`, `exponential`, and `yeo_johnson`.

```python
from beaverfe.transformations import NonLinearTransformation

NonLinearTransformation(
    transformation_options={
        "sepal length (cm)": "log",
        "sepal width (cm)": "exponential",
        "petal length (cm)": "yeo_johnson",
    }
)
```

#### **Quantile Transformations**

Transforms data to follow a normal or uniform distribution.

- Parameters:
    - `transformation_options`: Dictionary specifying the transformation type. Options: `uniform`, `normal`.

```python
from beaverfe.transformations import QuantileTransformation

QuantileTransformation(
    transformation_options={
        'sepal length (cm)': 'uniform',
        'sepal width (cm)': 'normal',
    }
)
```

#### **Scale Transformations**

Scales numerical data using different scaling methods.

- Parameters:
    - `transformation_options`: Dictionary specifying the scaling method for each column. Options: `min_max`, `standard`, `robust`, `max_abs`.
    -  `quantile_range`: Dictionary specifying the quantile ranges for robust scaling.

```python
from beaverfe.transformations import ScaleTransformation

ScaleTransformation(
    transformation_options={
        'sepal length (cm)': 'min_max',
        'sepal width (cm)': 'standard',
        'petal length (cm)': 'robust',
        'petal width (cm)': 'max_abs',
    },
    quantile_range={
        "petal length (cm)": (25.0, 75.0),
    },
)
```

#### **Normalization**

Normalizes data using L1 or L2 norms.

- Parameters:
    - `transformation_options`: Dictionary specifying the normalization type. Options: `l1`, `l2`.

```python
from beaverfe.transformations import Normalization

Normalization(
    transformation_options={
        'sepal length (cm)': 'l1',
        'sepal width (cm)': 'l2',
    }
)
```

<a id="numerical-features"></a>
### 📌 Numerical Features

#### **Spline Transformations**

Applies Spline transformation to numerical features.

- Parameters:
    - `transformation_options`: Dictionary specifying the spline transformation settings for each column. Options include different numbers of knots and degrees.

```python
from beaverfe.transformations import SplineTransformation

SplineTransformation(
    transformation_options={
        'sepal length (cm)': {'degree': 3, 'n_knots': 3},
        'sepal width (cm)': {'degree': 3, 'n_knots': 5},
    }
)
```

#### **Numerical Binning**

Bins numerical columns into categories. You can now specify the column, the binning method, and the number of bins in a tuple.

- Parameters:
    - `transformation_options`: Dictionary specifying the binning method and number of bins for each column. Options for binning methods are `uniform`, `quantile` or `kmeans`.

```python
from beaverfe.transformations import NumericalBinning

NumericalBinning(
    transformation_options={
        "sepal length (cm)": ("uniform", 5),
        "sepal width (cm)": ("quantile", 6),
        "petal length (cm)": ("kmeans", 7),
    }
)
```

#### **Mathematical Operations**

Performs mathematical operations between columns.

- Parameters:
    - `operations_options`: List of tuples specifying the columns and the operation.

- **Options**:
    - `add`: Adds the values of two columns.
    - `subtract`: Subtracts the values of two columns.
    - `multiply`: Multiplies the values of two columns.
    - `divide`: Divides the values of two columns.
    - `modulus`: Computes the modulus of two columns.
    - `hypotenuse`: Computes the hypotenuse of two columns.
    - `mean`: Calculates the mean of two columns.

```python
from beaverfe.transformations import MathematicalOperations

MathematicalOperations(
    operations_options=[
        ('sepal length (cm)', 'sepal width (cm)', 'add'),
        ('petal length (cm)', 'petal width (cm)', 'subtract'),
        ('sepal length (cm)', 'petal length (cm)', 'multiply'),
        ('sepal width (cm)', 'petal width (cm)', 'divide'),
        ('sepal length (cm)', 'petal width (cm)', 'modulus'),
        ('sepal length (cm)', 'sepal width (cm)', 'hypotenuse'),
        ('petal length (cm)', 'petal width (cm)', 'mean'),
    ]
)
```

<a id="categorical-features"></a>
### 📌 Categorical Features

#### **Categorical Encoding**

Encodes categorical variables using various methods.

- Parameters:
    - `encodings_options`: Dictionary specifying the encoding method for each column.
    - `ordinal_orders`: Specifies the order for ordinal encoding.

- **Encodings**:
    - `backward_diff`: Uses backward difference coding to compare each category to the previous one.
    - `basen`: Encodes categorical features using a base-N representation.
    - `binary`: Converts categorical variables into binary representations.
    - `catboost`: Implements the CatBoost encoding, which is a target-based encoding method.
    - `count`: Replaces categories with the count of occurrences in the dataset.
    - `dummy`: Applies dummy coding, similar to one-hot encoding but with one less category to avoid collinearity.
    - `glmm`: Uses Generalized Linear Mixed Models to encode categorical variables.
    - `gray`: Converts categories into Gray code, a binary numeral system where two successive values differ in only one bit.
    - `hashing`: Uses a hashing trick to encode categorical features into a fixed number of dimensions.
    - `helmert`: Compares each level of a categorical variable to the mean of subsequent levels.
    - `james_stein`: Applies James-Stein shrinkage estimation for target encoding.
    - `label`: Assigns each category a unique integer label.
    - `loo`: Uses leave-one-out target encoding to replace categories with the mean target value, excluding the current row.
    - `m_estimate`: A variant of target encoding that applies an m-estimate to regularize values.
    - `onehot`: Converts categorical variables into binary vectors where each category is represented by a separate column.
    - `ordinal`: Replaces categories with ordinal values based on their ordering.
    - `polynomial`: Applies polynomial contrast coding to categorical variables.
    - `quantile`: Maps categorical variables to quantiles based on their distribution.
    - `rankhot`: Encodes categories based on their ranking, similar to one-hot but considering order.
    - `sum`: Uses sum coding to compare each level to the overall mean.
    - `target`: Encodes categories using the mean of the target variable for each category.
    - `woe`: Applies Weight of Evidence (WoE) encoding, useful in logistic regression by transforming categorical data into log odds.

```python
from beaverfe.transformations import CategoricalEncoding

CategoricalEncoding(
    transformation_options={
        'Sex': 'label',
        'Size': 'ordinal',
    },
    ordinal_orders={
        "Size": ["small", "medium", "large"]
    }
)
```

<a id="periodic-features"></a>
### 📌 Periodic Features

#### **Date Time Transforms**

Extracts time-based features like day, month, hour, etc.

- Parameters:
    - `features`: List of columns to extract date/time features from. If None, all datetime columns are considered.

```python
from beaverfe.transformations import DateTimeTransformer

DateTimeTransformer(
    features=["date"]
)
```

#### **Cyclical Features Transforms**

Encodes cyclical values using sine and cosine representations.

- Parameters:
    - `transformation_options`: Dictionary specifying the period for each cyclical column.

```python
from beaverfe.transformations import CyclicalFeaturesTransformer

CyclicalFeaturesTransformer(
    transformation_options={
        "date_minute": 60,
        "date_hour": 24,
    }
)
```

<a id="features-reduction"></a>
### 📌 Features Reduction

#### **Column Selection**

Selects a subset of columns for further transformation.

- Parameters:
    - `features`: List of column names to select.

```python
from beaverfe.transformations import ColumnSelection

ColumnSelection(
    features=[
        "sepal length (cm)",
        "sepal width (cm)",
    ]
)
```

#### **Dimensionality Reduction**

Reduces the dimensionality of the dataset using various techniques, such as PCA, Factor Analysis, ICA, LDA, and others.

- Parameters:
    - `features`: List of column names to apply the dimensionality reduction. If None, all columns are considered.
    - `method`: The dimensionality reduction method to apply.
    - `n_components`: Number of dimensions to reduce the data to.

- **Methods**:
    - `pca`: Principal Component Analysis.
    - `factor_analysis`: Factor Analysis.
    - `ica`: Independent Component Analysis.
    - `kernel_pca`: Kernel PCA.
    - `lda`: Linear Discriminant Analysis.
    - `truncated_svd`: Truncated Singular Value Decomposition.
    - `isomap`: Isomap Embedding.
    - `lle`: Locally Linear Embedding.

- **Notes**:
For `lda`, the y target variable is required, as it uses class labels for discriminant analysis.

```python
from beaverfe.transformations import DimensionalityReduction

DimensionalityReduction(
    method="pca",
    n_components=3
)
```

---

<a id="contributing"></a>
## 🛠️ Contributing

We welcome contributions! Please submit pull requests, open issues, or share suggestions to improve Beaver FE.

---

<a id="license"></a>
## 📄 License

Beaver FE is open-source software distributed under the MIT License.

---

**🚀 Power up your ML workflows with intelligent, flexible feature engineering — with just a few lines of code. Try Beaver FE today!**
