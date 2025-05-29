# DTS002_ElcDataAny

This is a simple school project for the DTS002 course, focusing on the analysis and modeling of electricity data.

## Project Structure

*   `DataAny.ipynb`: Contains all the code for data preprocessing, exploratory data analysis (EDA), feature engineering, model training (including MLP), and hyperparameter optimization using Optuna.
*   `GlobalElectricityStatistics.csv`: The original dataset.
*   `GlobalElectricityStatistics_cleaned.csv`: The cleaned and preprocessed dataset.
*   `best_params.json`: Stores the best hyperparameter combination for the MLP model, found using Optuna.
*   `performance_metrics.json`: Stores the performance metrics of the trained MLP model.

## Data Analysis and Modeling Techniques

This project primarily utilizes the following technologies:

*   **Core Language:** Python
*   **Machine Learning Model:**
    *   **Multilayer Perceptron (MLP):** Serves as the main predictive model for analyzing electricity data and making relevant predictions. The model implementation might be based on Scikit-learn or other deep learning frameworks.
*   **Hyperparameter Optimization:**
    *   **Optuna:** Employed to automatically search and optimize the hyperparameters of the MLP model to achieve the best performance.

## Note

This project is a course assignment, primarily intended for learning and practicing data analysis and machine learning techniques. It is not intended for production use.