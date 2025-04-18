from tqdm import tqdm
import numpy as np
import pandas as pd


def extract_embeddings(model, dataloader,device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, label in tqdm(dataloader):
            images = images.to(device)
            
            embedding = model(images)
            embeddings.append(embedding[0].cpu().numpy())
            labels.append(label)
    return np.vstack(embeddings), np.hstack(labels)


def save_embeddings_to_csv(embeddings, labels, filename):
    
    data = pd.DataFrame(embeddings)
    data['label'] = labels  
    data.to_csv(filename, index=False)


