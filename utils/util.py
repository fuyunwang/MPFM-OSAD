import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
def init_gmm_prototypes(features_n, K):
    N, D = features_n.shape
    device = features_n.device


    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=32)
    cluster_labels = kmeans.fit_predict(features_n.cpu().numpy())
    mu = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)


    bincount = torch.bincount(torch.tensor(cluster_labels, device=device), minlength=K)
    pi = bincount.float() / N


    s2_total = 0.0
    for k in range(K):
        mask = (cluster_labels == k)
        if mask.sum() > 0:
            cluster_feats = features_n[mask]
            dist_sq = torch.sum((cluster_feats - mu[k]) ** 2)
            s2_total += dist_sq.item()

    s2 = torch.tensor(s2_total / (N * D), device=device, dtype=torch.float32)


    return mu, pi, s2

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;