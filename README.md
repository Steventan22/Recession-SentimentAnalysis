# Recession-SentimentAnalysis

## Overview
In the current era, global recession news dominates social media platforms, particularly in Indonesia. Social news platforms like Twitter play a pivotal role in disseminating information, generating public responses, and sparking discussions on the issue. This collaborative research, led by two dedicated faculty members and five enthusiastic students, focuses on mining public opinions from Twitter using a sentiment analysis approach to gain invaluable insights into the collective sentiment towards global recession news.

## Background
Understanding public sentiment towards global economic events is crucial for policymakers, investors, and the general public. With the prevalence of social media, particularly Twitter, analyzing the sentiments expressed by users provides real-time insights into public reactions and concerns. The research aims to uncover and evaluate sentiments related to global recession news circulating on Twitter in Indonesia.

## Objectives
- Data Collection: Gather data from Twitter to understand public opinions and responses related to global recession news.
- Data Pre-processing:Clean and preprocess the collected data to ensure accuracy and consistency.
- Sentiment Labeling: Utilize lexical-based methods like Valence Aware Dictionary and Sentiment Reasoner (VADER) and Textblob for sentiment labeling.
- Data Sampling: Apply sampling techniques such as Synthetic Minority Oversampling Technique (SMOTE) and Random Over Sampling to address the imbalance in the collected dataset.
- Modeling: Implement machine learning models including Support Vector Machines (SVM), K-Nearest Neighbour, and Naive Bayes for sentiment analysis.
- Model Evaluation: Evaluate the performance of the models and identify the most accurate and effective algorithm for sentiment analysis.

##Research Findings
The research encountered a challenge with almost 300,000 collected data points from NodeXL being unbalanced. Models trained on balanced datasets yielded better evaluation results. The Bernoulli-Naive Bayes algorithm, combined with the VADER labeling technique and SMOTE sampling after splitting data, achieved the highest accuracy of 84%. Additionally, using the Random Over Sampling technique after splitting data resulted in an accuracy of 81%. Notably, employing the SMOTE and Random Over Sampling technique before splitting data on the SVM algorithm reached the best accuracy of 93%, outperforming the SVM model without sampling, which achieved 84%.

## Disclaimer
The research findings are based on the data available at the time of the study and may be subject to changes in public sentiment. Further analysis and validation may be required for comprehensive insights.

## Tools Utilized
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
