# Student Pass / Fail Prediction using Machine Learning

## Project Overview
This project predicts whether a student will pass or fail the final exam using early exam scores.

## Problem
Teachers usually know students’ performance only after final exams, when it is too late to help.

## Solution
A machine learning model predicts pass or fail using:
- First exam score (G1)
- Second exam score (G2)

Final exam score (G3) is used only to label results, not for prediction.

## How It Works
1. Load student performance data
2. Convert final score into Pass / Fail
3. Train a Logistic Regression model
4. Predict results for new students

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Logistic Regression

## Results
- Model Accuracy: ~91%

## Example
Input:
G1 = 10  
G2 = 12  

Output:
PASS

## Use Case
Helps teachers identify weak students early and provide support before final exams.

## Author
Harsenth
