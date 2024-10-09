
---

# Employee Attrition Predictor

### Project Duration: May 2024 – Jun 2024 

## Project Overview

The **Employee Attrition Predictor** is a machine learning project focused on analyzing employee data to predict the likelihood of attrition. This project leverages multiple machine learning models to uncover patterns and trends that contribute to employee turnover.

### Models Used:
- **Logistic Regression**
- **Random Forest**
- **Sequential Deep Learning Model using TensorFlow**

Each model was trained and evaluated on employee data, yielding the following accuracy rates:
- **Logistic Regression**: 90%
- **Random Forest**: 89%
- **Deep Learning Model**: 88%

## Key Features

- **Data Analysis**: Preprocessing and exploration of employee data to identify relevant features influencing attrition.
- **Model Training**: Implementation of multiple machine learning models for comparison of performance.
- **Performance Metrics**: Evaluation using accuracy to measure the effectiveness of each model.

## Getting Started

### Prerequisites

To run this project, you'll need the following:
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- Numpy

Install the required libraries using:

```bash
pip install tensorflow scikit-learn pandas numpy
```

### Running the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/EmployeeAttritionPredictor.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd EmployeeAttritionPredictor
   ```

3. **Run the script:**
   ```bash
   python train_model.py
   ```

## Models Overview

### 1. Logistic Regression
A simple linear model that achieved an accuracy of 90%. This model is effective for binary classification problems like attrition prediction.

### 2. Random Forest
A robust ensemble model with 100 decision trees. It reached an accuracy of 89%, providing good performance while handling feature importance.

### 3. Sequential Deep Learning Model (TensorFlow)
A neural network built using the TensorFlow framework. Despite its complexity, the model attained an accuracy of 88%.

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 90%      |
| Random Forest       | 89%      |
| Deep Learning (TF)  | 88%      |

## Conclusion

This project demonstrates the power of machine learning in predicting employee attrition. Logistic Regression, Random Forest, and Sequential Deep Learning models were successfully implemented to predict attrition with high accuracy. The results indicate that even simple models can yield competitive performance for classification tasks.

## Future Improvements

- **Feature Engineering**: Further tuning and selection of features could potentially improve model accuracy.
- **Additional Models**: Exploring other machine learning models like XGBoost or SVM.
- **Hyperparameter Tuning**: Fine-tuning the parameters of the Random Forest and Deep Learning models to boost performance.

## License

This project is licensed under the MIT License.

---

