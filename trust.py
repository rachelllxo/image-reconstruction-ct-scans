import torch
import numpy as np

def compute_trust_map(model, img, runs=10):     
    model.train() 

    preds = []
    with torch.no_grad():
        for _ in range(runs):
            out = model(img)
            preds.append(out.cpu().numpy())

    preds = np.stack(preds)
    mean_recon = preds.mean(axis=0)
    if mean_recon.max() < 1e-5:
        mean_recon = img.cpu().numpy() 
    
    uncertainty = preds.var(axis=0)

    eps = 1e-8
    trust = 1 - (uncertainty / (uncertainty.max() + eps))

    return mean_recon[0, 0], trust[0, 0]