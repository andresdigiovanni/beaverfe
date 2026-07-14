# ⚙️ Available Transformations

Grouped by feature type or transformation category:

## Table of Contents

- [❌ Missing Values & Outliers](#-missing-values--outliers)
  - [Missing Values Indicator](#missing-values-indicator)
  - [Missing Values Handler](#missing-values-handler)
  - [Handle Outliers](#handle-outliers)
- [📊 Data Distribution & Scaling](#-data-distribution--scaling)
  - [Non-Linear Transformation](#non-linear-transformation)
  - [Quantile Transformations](#quantile-transformations)
  - [Scale Transformations](#scale-transformations)
  - [Normalization](#normalization)
- [🔢 Numerical Features](#-numerical-features)
  - [Spline Transformations](#spline-transformations)
  - [Numerical Binning](#numerical-binning)
  - [Mathematical Operations](#mathematical-operations)
- [🏷️ Categorical Features](#-categorical-features)
  - [Categorical Encoding](#categorical-encoding)
- [⏰ Periodic Features](#-periodic-features)
  - [Date Time Transforms](#date-time-transforms)
  - [Cyclical Features Transforms](#cyclical-features-transforms)
- [✂️ Features Reduction](#-features-reduction)
  - [Column Selection](#column-selection)
  - [Dimensionality Reduction](#dimensionality-reduction)

---

## ❌ Missing Values & Outliers

### **Missing Values Indicator**

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

### **Missing Values Handler**

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
    n_neighbors={
        'sepal width (cm)': 5,
    }
)
```

### **Handle Outliers**

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

---

## 📊 Data Distribution & Scaling

### **Non-Linear Transformation**

Applies logarithmic, exponential, Box-Cox or Yeo-Johnson transformations.

- Parameters:
    - `transformation_options`: A dictionary specifying the transformation to be applied for each column. Options include: `log`, `exponential`, `yeo_johnson`, `box_cox`.

> **Note:** `box_cox` requires all values in the column to be strictly positive.

```python
from beaverfe.transformations import NonLinearTransformation

NonLinearTransformation(
    transformation_options={
        "sepal length (cm)": "log",
        "sepal width (cm)": "exponential",
        "petal length (cm)": "yeo_johnson",
        "petal width (cm)": "box_cox",
    }
)
```

### **Quantile Transformations**

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

### **Scale Transformations**

Scales numerical data using different scaling methods.

- Parameters:
    - `transformation_options`: Dictionary specifying the scaling method for each column. Options: `min_max`, `standard`, `robust`, `max_abs`.
    - `quantile_range`: Dictionary specifying the quantile ranges for robust scaling.

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

### **Normalization**

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

---

## 🔢 Numerical Features

### **Spline Transformations**

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

### **Numerical Binning**

Bins numerical columns into categories.

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

### **Mathematical Operations**

Performs mathematical operations between columns.

- Parameters:
    - `operations_options`: List of tuples specifying the columns and the operation.

- **Options**: `add`, `subtract`, `multiply`, `divide`, `modulus`, `hypotenuse`, `mean`, `power`, `min`, `max`, `log_ratio`.

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
        ('sepal length (cm)', 'petal width (cm)', 'power'),
        ('sepal width (cm)', 'petal length (cm)', 'min'),
        ('sepal width (cm)', 'petal length (cm)', 'max'),
        ('sepal length (cm)', 'petal length (cm)', 'log_ratio'),
    ]
)
```

---

## 🏷️ Categorical Features

### **Categorical Encoding**

Encodes categorical variables using various methods.

- Parameters:
    - `transformation_options`: Dictionary specifying the encoding method for each column.
    - `ordinal_orders`: Specifies the order for ordinal encoding.

- **Encodings**: `backward_diff`, `basen`, `binary`, `catboost`, `count`, `dummy`, `glmm`, `gray`, `hashing`, `helmert`, `james_stein`, `label`, `loo`, `m_estimate`, `onehot`, `ordinal`, `polynomial`, `quantile`, `rankhot`, `sum`, `target`, `woe`.

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

---

## ⏰ Periodic Features

### **Date Time Transforms**

Extracts time-based features like day, month, hour, etc.

- Parameters:
    - `features`: List of columns to extract date/time features from. If None, all datetime columns are considered.

```python
from beaverfe.transformations import DateTimeTransformer

DateTimeTransformer(
    features=["date"]
)
```

### **Cyclical Features Transforms**

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

---

## ✂️ Features Reduction

### **Column Selection**

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

### **Dimensionality Reduction**

Reduces the dimensionality of the dataset using various techniques, such as PCA, Factor Analysis, ICA, LDA, and others.

- Parameters:
    - `features`: List of column names to apply the dimensionality reduction. If None, all columns are considered.
    - `method`: The dimensionality reduction method to apply.
    - `n_components`: Number of dimensions to reduce the data to.

- **Methods**: `pca`, `factor_analysis`, `ica`, `kernel_pca`, `lda`, `truncated_svd`, `isomap`, `lle`.

> **Note:** For `lda`, the y target variable is required, as it uses class labels for discriminant analysis.

```python
from beaverfe.transformations import DimensionalityReduction

DimensionalityReduction(
    method="pca",
    n_components=3
)
```
