import pandas as pd
import numpy as np
import os
import torch


def count_labels_preds(labels_test):
    """Return a value counts of labels predicted on test set
    Args : 
        labels_test (pd.DataFrame) : DataFrame of classes predicted on every pixel of test set"""

    return pd.Series(labels_test.values.ravel()).value_counts()



def pred_and_save(test_loader, model,  labels_path,output_filename):
    """Predicts on test set and save the csv file at the submission format
    Args :
        test_loader : Loader on test set 
        model : Segmentation model
        labels_path (str) : Path for DataFrame of labels predicted on every pixel of train set
        output_filename : Filename path """
    
    all_preds = []
    filenames = []
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Inference and save CSV
    with torch.no_grad():
        for imgs, names in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B,H,W)
            for p, n in zip(preds, names):
                all_preds.append(p.flatten())
                filenames.append(n)
                
    labels_train = pd.read_csv(labels_path, index_col=0, header=0)
    
    # Create the submission DataFrame
    df = pd.DataFrame(np.stack(all_preds, axis=0), columns= labels_train.T.columns)
    df = df.T
    df.columns = filenames
    
    # Save to CSV
    df.to_csv(output_filename, index=True)
    print(f"Test predictions saved to {output_csv}")
