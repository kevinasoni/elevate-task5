# elevate-task5

# Heart Disease Prediction: Tree-Based Model Comparison

## Objective
Apply and compare tree-based machine learning models for classification and feature analysis on the heart disease dataset.

---

## Workflow & Key Outputs

1. **Data Preparation**
    - Used `heart.csv` (contains features + "target" for disease).

2. **Decision Tree (DT) Classifier**
    - Baseline DT trained; **test accuracy** reported.
    - Visualized the tree structure (exported DOT file), and confusion matrix.

3. **Overfitting Analysis**
    - Controlled tree complexity using `max_depth`; plotted accuracy vs depth.
    - Helpful for identifying overfitting.

4. **Random Forest (RF) Classifier**
    - Built/compared ensemble; typically more robust.
    - Compared test and cross-validation accuracy.
    - Plotted feature importances as a horizontal bar chart.

5. **Cross-Validation**
    - Accuracy compared via 5-fold CV on both DT and RF.

6. **Evaluation Outputs**
    - `dt_cm.png`, `rf_cm.png`: Confusion matrices for DT and RF.
    - `dt_depth_vs_acc.png`: DT's depth vs accuracy.
    - `rf_feat_imp.png`: RF feature importances.

---

## Usage

1. Clone this repo and add `heart.csv`.
2. Run `pip install scikit-learn pandas matplotlib seaborn graphviz` (Graphviz required for DOT-to-PNG).
3. Execute: `python main.py`
4. View the saved PNG images for results and add them to your report.

---

