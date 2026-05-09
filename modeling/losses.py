import torch
import torch.nn as nn

class DeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        confidence_margin = 5.
        ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        dev_loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        return torch.mean(dev_loss)


class MIMRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z_mapped, mu, pi, s2):
        B, D = z_mapped.shape
        K = mu.shape[0]


        dist_sq = torch.cdist(z_mapped, mu, p=2) ** 2

        log_prob_gaussian = -0.5 * D * torch.log(2 * torch.pi * s2) - 0.5 * dist_sq / s2

        log_pi = torch.log(pi + self.eps)
        log_joint = log_pi.unsqueeze(0) + log_prob_gaussian

        log_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)

        log_posterior = log_joint - log_marginal
        posterior = torch.exp(log_posterior)  # [B, K]

        cond_entropy_term = torch.sum(posterior * log_posterior, dim=1).mean()

        marginal_entropy_term = torch.sum(pi * log_pi)
        L_mim = cond_entropy_term - marginal_entropy_term
        return L_mim