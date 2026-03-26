# Understanding Random Forests: Attribute Bagging and Variance Reduction

This repository contains a comprehensive tutorial and practical implementation of the **Random Forest** algorithm. The project specifically explores the mechanics of **Attribute Bagging** (Feature Bagging) and its critical role in reducing model variance and preventing overfitting in complex datasets.

Using a real-world customer churn scenario, this project demonstrates how ensemble learning can transform a collection of "weak" individual decision trees into a robust, high-performance predictive system.

## 📂 Project Structure

* **`24178639_Tutorial_On_Attribute_bagging_technique_for_Customer_churn_prediction.ipynb`**: A hands-on Jupyter Notebook featuring data preprocessing, model training, hyperparameter tuning, and detailed evaluation.
* **`24178639_MLNN_Tutorial-Random Forests_Understanding Attribute Bagging.pdf`**: A theoretical guide covering ensemble learning principles, feature randomness, and the ethical implications of AI.
* **`customer_churn_business_dataset.csv`**: The primary dataset used for the analysis, containing features like tenure, monthly logins, CSAT scores, and revenue.
* **`plot_1_mlnn_tutorial.png`**: Visualization of Model Performance / Confusion Matrix.
* **`plot_2_mlnn_tutorial.png`**: Visualization of Feature Importance.
* **`plot_3_mlnn_tutorial.png`**: Visualization of Attribute Bagging effects.

## 🚀 Key Technical Concepts

### 1. Attribute Bagging
Unlike standard bagging which only resamples data rows, **Attribute Bagging** (or Feature Bagging) forces each split in a decision tree to consider only a random subset of available features. This technique ensures that even highly dominant features do not lead to highly correlated trees, thereby increasing the diversity of the forest.

### 2. Variance Reduction
Decision trees are known for having high variance (overfitting). By aggregating predictions across many decorrelated trees, the Random Forest "averages out" individual errors, resulting in a model that generalizes much better to unseen data.

## 🛠️ Implementation Details

* **Language**: Python
* **Key Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.
* **Primary Insights**: The model identifies **Tenure Months**, **Monthly Logins**, and **CSAT Score** as the most significant predictors of customer churn.

## 🔧 Installation and Usage

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Deshan-Chathuranga/Random_Forest_Algorithm_With_Attribute_Bagging.git](https://github.com/Deshan-Chathuranga/Random_Forest_Algorithm_With_Attribute_Bagging.git)
2. **Install the required packages:**:
```bash
 pip install pandas numpy matplotlib seaborn scikit-learn
3. **Run the Notebook**:
   Open the .ipynb file in any Jupyter environment or Google Colab to see the full analysis and results.