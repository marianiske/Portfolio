# Football Match Prediction
Description: Since XGBoost is a powerful and widely used approach for tabular prediction tasks, I chose a football match dataset to showcase my skills in building a data pipeline, engineering meaningful features, and training a predictive model.

# Pre-Processing
I collected the dataset from [football-data.co.uk](https://www.football-data.co.uk/data.php) and enriched it with additional features based on a simple stochastic model using expected goals (xG) and win probabilties by using a simpler [model](Webdata/win_probs.py). The resulting dataset includes all matches from the top five European leagues since the 2014/15 season, comprising more than 40,000 observations with 42 features each.

# Labels
I encoded the target labels as 0 for a home win, 1 for a draw, and 2 for an away win. This is a practical choice for XGBoost because the model expects class labels in a numeric format for multiclass classification. The numeric encoding does not imply a meaningful ordering between the outcomes, but simply provides a compact representation of the three possible match results. Keeping the labels separate from the feature set also ensures a clean distinction between input variables and the prediction target.

# Model
