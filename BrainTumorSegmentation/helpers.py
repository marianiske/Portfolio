import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

def show_bbox(original_image, prediction):
  result = original_image.copy()
  mask = prediction > 0.5
  contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
  return result

def show_prediction(val_ds, model, batch_number, idx, bounding_box = False):

    for images, masks in val_ds.take(batch_number):
        test_image = images[idx] 
        test_mask = masks[idx]   
        break
    
    test_image_for_model = tf.expand_dims(test_image, axis=0)
    
    prediction_batch = model(test_image_for_model)
    prediction_single = prediction_batch[0]
    
    if bounding_box:
        result_image = show_bbox(test_image.numpy(), prediction_single.numpy())
    else:
        result_image = prediction_single.numpy()
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(test_image.numpy()) 
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(test_mask.numpy().squeeze(), cmap='gray')
    
    last_title = "Prediction with Bounding Box" if bounding_box else "Prediction"
    plt.subplot(1, 3, 3)
    plt.title(last_title)
    plt.imshow(result_image)
    
    png_title = 'bounding_box' if bounding_box else 'prediction'
    plt.savefig(f"graphics/{png_title}_{batch_number}_{idx}.png")
    plt.show()

def show_prediction_diff(val_ds, model, batch_number, idx):

  for images, masks in val_ds.take(batch_number):
      test_image = images[idx]
      test_mask = masks[idx]
      break

  test_image_for_model = tf.expand_dims(test_image, axis=0)

  prediction_batch = model(test_image_for_model)
  prediction_single = prediction_batch[0]

  pred_mask = prediction_single > 0.5

  diff = test_mask.numpy().squeeze() - pred_mask.numpy().squeeze()

  cmap = mcolors.ListedColormap(['red', 'gray', 'blue'])
  bounds = [-1.5, -0.5, 0.5, 1.5]
  norm = mcolors.BoundaryNorm(bounds, cmap.N)

  plt.figure(figsize=(18, 6))

  plt.subplot(1, 3, 1)
  plt.title("Original Image")
  plt.imshow(test_image.numpy())

  plt.subplot(1, 3, 2)
  plt.title("Ground Truth Mask")
  plt.imshow(test_mask.numpy().squeeze(), cmap='gray')

  plt.subplot(1, 3, 3)
  plt.title("Prediction Difference (Red: FP, Gray: Match, Blue: FN)")
  img = plt.imshow(diff, cmap=cmap, norm=norm)
  plt.colorbar(img, ticks=[-1, 0, 1], format=ticker.FuncFormatter(lambda x, p: { -1:'False Positive', 0:'Match', 1:'False Negative'}.get(x, '')))
  
  plt.tight_layout()
  plt.savefig(f"prediction_diff_{batch_number}_{idx}.png")
  plt.show()