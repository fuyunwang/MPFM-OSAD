import torch
import torch.nn as nn
import math


class MPFL(nn.Module):

    def __init__(self, backbone, flow_network):

        super().__init__()
        self.backbone = backbone
        self.flow_network = flow_network

        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_features(self, x):

        with torch.no_grad():
            z_0 = self.backbone(x)
        return z_0

    def compute_gmm_nll(self, u_target, gm_dict):

        means = gm_dict['means']  # [B, K, C, H, W]
        logweights = gm_dict['logweights']  # [B, K, 1, H, W]
        logstds = gm_dict['logstds']  # [B, 1, 1, 1, 1]

        B, K, C, H, W = means.shape

        u_expanded = u_target.unsqueeze(1)


        dist_sq = torch.sum((u_expanded - means) ** 2, dim=2, keepdim=True)

        # 计算方差 s^2
        var = torch.exp(logstds * 2)

        D = C
        log_prob_gaussian = -0.5 * D * math.log(2 * math.pi) \
                            - 0.5 * D * (logstds * 2) \
                            - 0.5 * dist_sq / var


        log_joint = logweights + log_prob_gaussian

        # log( \sum \pi_k N ) -> [B, 1, 1, H, W]
        log_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)

        # 返回 NLL
        nll = -log_marginal.mean()
        return nll

    def forward(self, x, is_normal=True):
        z_0 = self.extract_features(x)

        B, C, H, W = z_0.shape
        device = z_0.device

        t = torch.rand((B,), device=device, dtype=z_0.dtype)

        z_T = torch.randn_like(z_0)

        t_expanded = t.view(B, 1, 1, 1)  # 对齐维度用于广播
        z_t = (1 - t_expanded) * z_0 + t_expanded * z_T
        u_target = z_T - z_0

        gm_pred = self.flow_network(hidden_states=z_t, timestep=t)

        # 6. 计算负对数似然 NLL (Eq. 12)
        nll = self.compute_gmm_nll(u_target, gm_pred)

        if is_normal:
            # (Eq. 17)
            loss_flow = nll
        else:
            loss_flow = -nll

        with torch.no_grad():
            t_zero = torch.zeros((B,), device=device, dtype=z_0.dtype)
            gm_z0_mapped = self.flow_network(hidden_states=z_0, timestep=t_zero)

            pi = gm_z0_mapped['logweights'].exp()
            means = gm_z0_mapped['means']
            z_mapped = (pi * means).sum(dim=1)


            z_mapped_flat = z_mapped.mean(dim=(2, 3)) if len(z_mapped.shape) == 4 else z_mapped

            v_map = z_0

        return loss_flow, z_mapped_flat, v_map