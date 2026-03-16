import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from Models.xg_boost_HDA import XGBMatchModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
)

def build_dataset():

    X_train_parts, X_val_parts = [], []
    y_train_parts, y_val_parts = [], []
    
    leagues = ["Bundesliga", "EPL", "Serie_A", "Ligue_1", "La_Liga"]
    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    
    for league in leagues:
        league_dfs = []
        league_ys = []
        for year in years:
            option = f"{league}_{year}"
            
            try:
                # Features
                df_data = pd.read_json(f"Data/dataset_{option}.json")
                df_data = df_data.rename(columns={"Avg Quota <2.5": "Avg_Quota_lt_2.5"})
                df_data = df_data.apply(pd.to_numeric, errors="coerce")
                
                # Labels
                df_label = pd.read_json(f"Data/labels_{option}.json")
                df_hda = df_label[["H", "D", "A"]]
                y = pd.Series(df_hda.values.argmax(axis=1), name="Win")
                
                mask = df_data.notna().all(axis=1)
                
                df_data = df_data.loc[mask].reset_index(drop=True)
                y = y.loc[mask].reset_index(drop=True)
                
                league_dfs.append(df_data)
                league_ys.append(y)
            except: 
                continue
            
        X_league = pd.concat(league_dfs, ignore_index=True)
        y_league = pd.concat(league_ys, ignore_index=True)
        
        #league-wise split
        split_idx = int(len(X_league) * 0.7)
        
        X_train_parts.append(X_league.iloc[:split_idx].reset_index(drop=True))
        X_val_parts.append(X_league.iloc[split_idx:].reset_index(drop=True))
        y_train_parts.append(y_league.iloc[:split_idx].reset_index(drop=True))
        y_val_parts.append(y_league.iloc[split_idx:].reset_index(drop=True))
    
    X_train = pd.concat(X_train_parts, ignore_index=True)
    X_val   = pd.concat(X_val_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)
    y_val   = pd.concat(y_val_parts, ignore_index=True)
    
    return X_train, X_val, y_train, y_val

def evaluate_model(model, X_val, y_val, label_map=None):
    if label_map is None:
        label_map = {0: "H", 1: "D", 2: "A"}
        
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    print("Accuracy:          ", round(accuracy_score(y_val, y_pred), 4))
    print("Balanced Accuracy: ", round(balanced_accuracy_score(y_val, y_pred), 4))
    print()
    print(classification_report(y_val, y_pred, target_names=class_names, digits=4))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    true_counts = pd.Series(y_val).value_counts().sort_index()
    axes[0, 0].bar(class_names, true_counts.values)
    axes[0, 0].set_title("Class distribution in the validation set")
    axes[0, 0].set_ylabel("Number of matches")

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0, 1], colorbar=False)
    axes[0, 1].set_title("Confusion Matrix")

    report = classification_report(
        y_val, y_pred, target_names=class_names, output_dict=True, digits=4
    )
    metrics_df = pd.DataFrame({
        "precision": [report[c]["precision"] for c in class_names],
        "recall":    [report[c]["recall"]    for c in class_names],
        "f1-score":  [report[c]["f1-score"]  for c in class_names],
    }, index=class_names)

    x = np.arange(len(class_names))
    width = 0.22
    axes[1, 0].bar(x - width, metrics_df["precision"], width, label="Precision")
    axes[1, 0].bar(x,         metrics_df["recall"],    width, label="Recall")
    axes[1, 0].bar(x + width, metrics_df["f1-score"],  width, label="F1")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Metrics per class")
    axes[1, 0].legend()

    pred_conf = y_prob[np.arange(len(y_pred)), y_pred]
    correct_mask = (y_pred == np.array(y_val))

    axes[1, 1].hist(pred_conf[correct_mask], bins=20, alpha=0.7, label="correct prediction")
    axes[1, 1].hist(pred_conf[~correct_mask], bins=20, alpha=0.7, label="incorrect prediction")
    axes[1, 1].set_title("confidence of the prediction")
    axes[1, 1].set_xlabel("max. predicted probability")
    axes[1, 1].set_ylabel("number")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return y_pred, y_prob, report

def main():
    
    X_train, X_val, y_train, y_val = build_dataset()
    model = XGBMatchModel()
    model.fit(X_train, y_train, X_val, y_val)
    
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    
    label_map = {0: "H", 1: "D", 2: "A"}
    y_pred, y_prob, report = evaluate_model(model, X_val, y_val, label_map=label_map)
    
    joblib.dump({
        "model": model,          
        "features": list(X_train.columns),
        "label_map": {0: "H", 1: "D", 2: "A"}
    }, "xgb_hda_bundle.joblib")
    
if __name__ == '__main__':
    main() 
