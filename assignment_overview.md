Overview
Welcome to Assignment 2 on Image Classification.

Task
In this Assignment, you will build a deep learning model to classify images into one of the 16 object categories, including animals, vehicles, household objects, and everyday items. You are to use the given labelled training set to design and implement an image classification model using convolutional neural networks (CNNs). The details are as follows:

Building a custom MobileNet-like architecture (< 4.0M parameters) to extract discriminative features from images
Develop a complete image classification pipeline that could determine the top-1 classification accuracy of 16 categories on the images.
Apply advanced data augmentation techniques (random flips, rotations, colour jittering, etc.) to improve model generalisation and robustness.
Optimising your model's performance using SGD with momentum or Adam and Cosine annealing warm restarts scheduling.
Try to achieve high accuracy on the validation set of the task on the competitive Kaggle leaderboard evaluation.
The predictions on the given unlabelled images must be submitted using the "Submit Prediction" button to know how well your model is performing.
You have a maximum daily submission of 10. Students are encouraged to build models that generalize well.
Submission of Predictions
To evaluate your trained model, you are to download the unlabelled test set, and use your model to perform inference on the test set. The submission should be uploaded using the "Submit Prediction" button at the top-right corner. The predictions should be submitted in a csv file format below:

</> csv
Id,Prediction
test_000001.JPEG,airplane
test_000002.JPEG,bear
test_000003.JPEG,bicycle
The Id column must contain the test image filenames exactly as they appear in sample_submission.csv. The Prediction column must contain one of the valid class labels from class_names.csv.
Evaluation
Submissions will be evaluated using classification accuracy:


A higher accuracy indicates better classification performance.

Submission of Code
At the last day of this assignment challenge, students are to submit their source code in a zip via this assignment link on Sakai. Students' source codes will be runned and tested, so students should make sure that their codes are well-organised with a description text file of how to run the codes. Students must adhere to the assignment rules and academic honesty and code honnor as described on the course website.

Grading Scheme
The formular for the grading scheme is

NB
The Assignment is designed to help you apply key concepts in image classification, including data preprocessing, model training, validation, prediction, and performance evaluation. Your score on the leaderboard will be based on classification accuracy, but your final assessment will also consider your methodology, originality, explanation, and compliance with the assignment rules.

This is an academic exercise. You are expected to work independently, follow ethical guidelines, and submit a work that reflects your own understanding and effort. The goal is not only to achieve a good score, but also to demonstrate sound deep learning practice.

Start

11 days ago
Close

9 days to go
Submission Format Summary
Submit a CSV file with exactly two columns:

</> csv
Id,Prediction
Id must match the test image filename exactly as shown in sample_submission.csv.
Prediction must be one of the valid class labels in class_names.csv.
Every test image must have one prediction.
Do not add extra columns, rename files, or change the column names
</> csv
Id,Prediction
test_000001.JPEG,cat
test_000002.JPEG,dog
test_000003.JPEG,truck