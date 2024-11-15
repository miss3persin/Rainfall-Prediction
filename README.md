
# Rainfall Prediction Model 🌧️🌤️

This project implements a **Rainfall Prediction Model** using a **RandomForestClassifier**. The goal is to predict whether it will rain or not based on various meteorological features such as pressure, humidity, wind speed, etc.

## 📁 Dataset Overview
The dataset contains several meteorological features used for predicting rainfall. This is what it looks like after fixing some multicollinearity issues:
- **pressure**: Atmospheric pressure (hPa)
- **dewpoint**: Dew point temperature (°C)
- **humidity**: Relative humidity (%)
- **cloud**: Cloud cover (%)
- **sunshine**: Sunshine duration (hours)
- **winddirection**: Wind direction (degrees)
- **windspeed**: Wind speed (km/h)

The target variable is binary:
- `0` = No Rain
- `1` = Rain

## 🧪 Data Preprocessing
- **Handling Imbalanced Data**: The dataset was imbalanced, so we used downsampling to balance the classes.
- **Feature Engineering**: Selected relevant features and performed hyperparameter tuning.
- **Train-Test Split**: The dataset was split into training and test sets.

## 🔍 Model Architecture
We used a **RandomForestClassifier** with the following architecture:
- **Number of Trees (`n_estimators`)**: 50, 100, 200
- **Feature Selection (`max_features`)**: 'sqrt', 'log2'
- **Tree Depth (`max_depth`)**: None, 10, 20, 30
- **Minimum Samples per Split (`min_samples_split`)**: 2, 5, 10
- **Minimum Samples per Leaf (`min_samples_leaf`)**: 1, 2, 4

### Hyperparameter Tuning
Performed using **GridSearchCV** to find the best parameters:
- **Best Parameters**: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 50}

## 🧾 Model Evaluation
### Training Data Evaluation
- **Cross-validation scores**: [0.6842, 0.8158, 0.8378, 0.8378, 0.9189]
- **Mean Cross-validation score**: 0.819

### Test Data Evaluation
- **Test Set Accuracy**: 0.7447 (74.47%)
- **Confusion Matrix**:
  ```
  [[17  7]
   [ 5 18]]
  ```
- **Classification Report**:
  ```
               precision    recall  f1-score   support

           0       0.77      0.71      0.74        24
           1       0.72      0.78      0.75        23

    accuracy                           0.74        47
   macro avg       0.75      0.75      0.74        47
weighted avg       0.75      0.74      0.74        47
  ```

## 🚀 Usage Example
You can use the trained model to predict if it will rain today based on new meteorological data.

```python
import pandas as pd

# Sample input data: (pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed)
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)

input_df = pd.DataFrame([input_data], columns=['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])

# Making predictions
prediction = best_rf_model.predict(input_df)

if prediction[0] == 1:
    print('It will most likely rain today 🌧️')
else:
    print('It will most likely not rain today 🌤️')
```

## 🛠️ Things That Could Improve the Model
- **SMOTE**: Use SMOTE (Synthetic Minority Over-sampling Technique) instead of downsampling to handle class imbalance.
- **PCA**: Implement PCA (Principal Component Analysis) for dimensionality reduction rather than dropping correlated features.
- **Try Simpler Models**: Consider using simpler models like Logistic Regression, but make sure to perform feature scaling.
- **Model Selection**: Try different models along with hyperparameter tuning for better results.

## 📝 Requirements
To run this project, you need to install the following libraries:
- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install all required packages using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 📂 Project Structure
```
|-- Rainfall_Prediction_Model/
    |-- rainfall_prediction.py
    |-- rainfall_dataset.csv
    |-- rainfall_model.pkl
    |-- README.md
```

## 🖥️ How to Run
1. Clone this repository.
2. Run the `rainfall_prediction.py` script to train the model or use the saved model `rainfall_model.pkl`.
3. Use the usage example above to predict rainfall.
