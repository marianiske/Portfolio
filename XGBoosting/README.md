# Football Match Prediction
Description: Since XGBoost is a powerful and widely used approach for tabular prediction tasks, I chose a football match dataset to showcase my skills in building a data pipeline, engineering meaningful features, and training a predictive model.

# Pre-Processing
I collected the dataset from [football-data.co.uk](https://www.football-data.co.uk/data.php) and enriched it with additional features based on a simple stochastic model using among others expected goals (xG) and win probabilties by using a simpler [model](Webdata/win_probs.py). The resulting dataset includes all matches from the top five European leagues since the 2014/15 season, comprising more than 40,000 observations with 42 features each. I did not shuffle the data to avoid data leackage. 

# Labels
I encoded the target labels as 0 for a home win, 1 for a draw, and 2 for an away win. This is a practical choice for XGBoost because the model expects class labels in a numeric format for multiclass classification. The numeric encoding does not imply a meaningful ordering between the outcomes, but simply provides a compact representation of the three possible match results. Keeping the labels separate from the feature set also ensures a clean distinction between input variables and the prediction target.

# Model
I trained an XGBClassifier for a three-class prediction task: home win, draw, or away win. The model uses the multi:softprob objective to predict class probabilities, which is more informative than only predicting the most likely outcome. A small learning rate (0.03) combined with many boosting rounds (5000) allows the model to learn gradually, while early stopping ensures that training stops once validation performance no longer improves. To reduce overfitting, I limited tree depth, required a minimum child weight, used row and feature subsampling, and applied L2 regularization. The model was evaluated with multiclass log loss, which is well suited for probability-based predictions.

# Results
The model achieved an accuracy of 76% on the validation set. Performance was strongest for home wins and away wins, while draws were the most difficult class to predict. This is consistent with the underlying class distribution and the general difficulty of modeling draws in football prediction. The confusion matrix shows a tendency to misclassify some draws and away wins as home wins, suggesting a mild bias toward the most common outcome. Confidence histograms indicate that correct predictions are generally associated with higher predicted probabilities, which suggests that the model's probability estimates are meaningful.
![Results](results.png)

# Limitations
Draws are inherently difficult to predict with this model due to the class imbalance. One can also observe a home advantage bias. In addition for that we did not consider some features like travel distance, playing with a new coach or transfers, which can make a huge impact in reality. More accurate probabilities may still require calibration.
