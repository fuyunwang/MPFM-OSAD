import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAnomalyHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, top_o=0.1):
        super(LocalAnomalyHead, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.top_o = top_o

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)  # [B, 1, H, W]

        x = x.view(int(x.size(0)), -1)  # [B, H*W]

        if isinstance(self.top_o, float) and self.top_o < 1.0:
            topk_num = max(int(x.size(1) * self.top_o), 1)
        else:
            topk_num = min(int(self.top_o), x.size(1))

        x = torch.topk(x, topk_num, dim=1)[0]
        x = torch.mean(x, dim=1).view(-1, 1)
        return x


class NormalHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.0):
        super(NormalHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # [B, C, H, W] -> [B, C, 1, 1]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # [B, C]

        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)  # [B, 1]
        return x


class ResidualHead(nn.Module):

    def __init__(self, feat_dim, hidden_dim=64, dropout=0.0):
        super(ResidualHead, self).__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)  # [B, 1]
        return x


class AnomalyScoringModules(nn.Module):
    def __init__(self, in_channels, feat_dim, K, top_o=100):
        super().__init__()
        self.K = K
        self.top_o = top_o

        self.M_a_net = LocalAnomalyHead(in_channels, hidden_dim=512, top_o=top_o)
        self.M_n_net = NormalHead(in_channels, hidden_dim=512, dropout=0.1)
        self.M_r_net = ResidualHead(feat_dim, hidden_dim=512, dropout=0.1)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_Mg_score(self, z_mapped, mu, pi, s2):
        dist_sq = torch.cdist(z_mapped, mu, p=2) ** 2
        log_prob_gaussian = -0.5 * z_mapped.shape[1] * torch.log(2 * torch.pi * s2) - 0.5 * dist_sq / s2
        log_joint = torch.log(pi + 1e-8).unsqueeze(0) + log_prob_gaussian
        log_marginal = torch.logsumexp(log_joint, dim=1)  # [B]
        return -log_marginal

    def forward(self, v_map, z_mapped, mu, pi, s2, labels=None):
        B = v_map.shape[0]

        score_g = self.compute_Mg_score(z_mapped, mu, pi, s2)

        local_anomaly_map = self.M_a_net(v_map).view(B, -1)
        k = min(self.top_o, local_anomaly_map.shape[1])
        topk_scores, _ = torch.topk(local_anomaly_map, k, dim=1)
        score_a = topk_scores.mean(dim=1)

        score_n = self.M_n_net(v_map).squeeze(1)

        with torch.no_grad():
            dist_sq = torch.cdist(z_mapped, mu, p=2) ** 2
            c_star_idx = torch.argmin(dist_sq, dim=1)
        mu_c_star = mu[c_star_idx]
        residual = (z_mapped - mu_c_star) / torch.sqrt(s2)
        score_r = self.M_r_net(residual).squeeze(1)

        scores_dict = {'S_g': score_g, 'S_a': score_a, 'S_n': score_n, 'S_r': score_r}

        if labels is None:
            final_score = score_g + score_a + score_r - score_n
            return final_score, scores_dict

        labels_f = labels.float()
        L_score = (
                self.bce_loss(score_g, labels_f) +
                self.bce_loss(score_a, labels_f) +
                self.bce_loss(score_n, labels_f) +
                self.bce_loss(score_r, labels_f)
        )
        return L_score, scores_dict