from tensorflow import keras
import json
from brain_dataset import BrainDataSet
from UNet import UNet

def main():
    ds = BrainDataSet()
    
    train_ds = ds.train
    val_ds = ds.val
    
    model = UNet().model
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
            keras.metrics.Precision(name="precision", thresholds=0.5),
            keras.metrics.Recall(name="recall", thresholds=0.5),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.AUC(name="auc_pr", curve="PR"),
            keras.metrics.BinaryIoU(name="iou", target_class_ids=[1], threshold=0.5),
        ]
    )
    
    history = model.fit(
            train_ds,
            epochs=30,
            validation_data=val_ds,
        )
    
    with open("history.json", "w") as f:
        json.dump(history.history, f)
        
if __name__ == '__main__':
    main()