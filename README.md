# Advanced Predictive Modeling Portfolio: Real Estate & Automotive Markets

## 📖 Overview
This repository showcases two distinct advanced data science projects demonstrating expertise in **Deep Learning (Regression)** and **Discrete Choice Modeling (Classification/Econometrics)**.

1.  **Real Estate Price Prediction:** A deep learning approach to valuing properties, utilizing hyperparameter-tuned Neural Networks.
2.  **Used Car Consumer Choice Analysis:** A statistical framework using Decision Tree, Linear Regression, and customized `FastCombinedLogit` models to predict consumer purchasing behavior and acceptable price.
---

## 🏠 Project 1: Real Estate Price Prediction with Deep Learning

### 🎯 Objective
To build a robust predictive model for real estate valuation, handling high-dimensional data and capturing non-linear price dynamics.

### 🛠️ Methodology
The project explored multiple modeling techniques to minimize prediction error:
* **Random Forest Regressor:** Used as a baseline for capturing non-linear interactions.
* **Ensemble Method:** A weighted averaging of Random Forest and MLP predictions.
* **Multi-Layer Perceptron (MLP):** A deep neural network architecture designed for regression.

### 💡 Key Technical Implementation
* **Data Preprocessing:** Implemented `StandardScaler` for feature normalization to ensure fast convergence for gradient descent.
* **Architecture:** Designed a deep fully connected network with **6 hidden layers** (32 nodes each) using `ReLU` activation (optimized from `tanh` for better gradient flow).
* **Optimization:** Utilized the `Adam` solver with rigorous **Hyperparameter Tuning** (adjusting epochs, batch size, and learning rates).

### 🏆 Results & Conclusion
While ensemble methods often provide stability, extensive experiments revealed that the **standalone Neural Network** captured the underlying data manifold most effectively.

| Model | Performance Note |
| :--- | :--- |
| Random Forest | Good baseline, but struggled with extreme high-value outliers. |
| Ensemble (RF + MLP) | Averaged the errors but failed to surpass the tuned NN. |
| **Tuned Neural Network** | **Best Performance** |

**Final Result:** The hyperparameter-tuned Neural Network achieved a Mean Absolute Error (MAE) of **187,840.06**, significantly outperforming both the Random Forest and the Ensemble method.

---

## 🚗 Project 2: Used Car Choice & Market Share Analysis

### 🎯 Objective
To decode consumer decision-making processes in the used car market. The goal was not just to predict *which* car a user buys, but to understand *why* (Utility Theory) and to simulate aggregate market shares.

### 🛠️ Methodology
* **Discrete Choice Modeling (DCM):** Implemented a custom `FastCombinedLogit` class (inheriting from `GenericLikelihoodModel` in `statsmodels`) to handle high-dimensional choice sets (J=17 alternatives).
* **Optimization:** Used the **BFGS** algorithm for Maximum Likelihood Estimation (MLE) to estimate parameters.
* **Utility Function:** Modeled utility as a function of:
    * **Alternative-Specific Constants (ASCs):** Capturing brand brand equity.
    * **Shared Variables:** Price, Mileage, Age, Fuel Type, etc.

### 📊 Key Visualizations & Insights

#### 1. Drivers of Choice (Coefficients)
*Visualization of model coefficients with 95% confidence intervals.*
> **Insight:** The model successfully identified **Price** and **Car Age** as the most significant negative drivers of utility, while specific features like **Hybrid/Electric Fuel Types** showed positive utility premiums in specific segments.

#### 2. Market Share Calibration
*Comparison of Actual vs. Predicted Market Shares (In-Sample).*
> **Insight:** The model demonstrates excellent calibration for major brands (Toyota, Honda), confirming that the ASCs successfully absorbed aggregate brand preferences.
> *Note on Small Sample Bias:* An anomaly was observed in the `Ford SUV` segment during demonstration runs (N=200), illustrating the impact of data sparsity on coefficient convergence. Full-scale training corrects this artifact.

### 🚀 Technical Highlights
* **Custom Class Implementation:** Extended `statsmodels` to add a custom `.predict()` method for calculating probabilities via Softmax.
* **Efficiency:** Optimized the Hessian calculation to reduce training time from >10 minutes to <30 seconds for demonstration batches.
* **Prediction Validation:** Validated model performance using Top-1 Accuracy and Probability Boxplots, showing the model assigns significantly higher probabilities to chosen vehicles compared to rejected alternatives.

### 🔄 Extension: Price Prediction Module
*Complementing the Choice Model with Intrinsic Valuation*

While the **FastCombinedLogit** model treats price as an external input to determine user utility, predicting the fair market value of a vehicle is equally critical for a holistic market analysis. To bridge this gap, I developed a parallel pricing engine.

* **Modeling Approach:**
    * **Linear Regression:** Established a baseline to understand linear depreciation trends (e.g., impact of *Car Age* and *Odometer* on Price).
    * **Decision Tree Regressor:** Implemented to capture non-linear relationships and interaction effects (e.g., how the value of a specific brand drops precipitously after a certain mileage threshold).
* **Optimization:**
    * Conducted rigorous **Hyperparameter Tuning** on the Decision Tree (optimizing `max_depth`, `min_samples_split`, and `min_samples_leaf`) to balance bias and variance, ensuring the model generalizes well to unseen inventory data.
* **Synergy:** This pricing module serves as a crucial input generator for the Choice Model, allowing for dynamic simulation of how price fluctuations impact market share.
---

## 🧰 Tech Stack
* **Languages:** Python 3.x
* **Machine Learning:** `Scikit-learn` (RandomForest, MLPRegressor), `PyTorch` (Structure concepts)
* **Statistics:** `Statsmodels` (GenericLikelihoodModel), `SciPy` (Optimization)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`

