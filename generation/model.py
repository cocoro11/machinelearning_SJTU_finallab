# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    num_classes: int
    max_len: int = 200
    enc_hidden: int = 256
    dec_hidden: int = 512
    z_dim: int = 128
    class_emb: int = 64
    dropout: float = 0.1


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=5,
            hidden_size=cfg.enc_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc_mu = nn.Linear(cfg.enc_hidden * 2, cfg.z_dim)
        self.fc_logvar = nn.Linear(cfg.enc_hidden * 2, cfg.z_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)  # h: (2,B,H)
        h = torch.cat([h[0], h[1]], dim=1)  # (B,2H)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.class_emb = nn.Embedding(cfg.num_classes, cfg.class_emb)
        self.in_fc = nn.Linear(cfg.z_dim + cfg.class_emb + 5, cfg.dec_hidden)
        self.lstm = nn.LSTM(
            input_size=cfg.dec_hidden,
            hidden_size=cfg.dec_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.fc_dxdy = nn.Linear(cfg.dec_hidden, 4)  # mu_x, mu_y, logstd_x, logstd_y
        self.fc_pen = nn.Linear(cfg.dec_hidden, 3)   # p1,p2,p3 logits

    def forward(self, x_in: torch.Tensor, z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x_in.shape
        yemb = self.class_emb(y)  # (B,E)
        zcat = torch.cat([z, yemb], dim=1).unsqueeze(1).expand(B, L, -1)  # (B,L,z+E)
        h_in = torch.cat([x_in, zcat], dim=2)  # (B,L,5+z+E)
        h_in = self.in_fc(h_in)
        out, _ = self.lstm(h_in)
        dxdy_params = self.fc_dxdy(out)
        pen_logits = self.fc_pen(out)
        return dxdy_params, pen_logits

    @torch.no_grad()
    def sample(self, z: torch.Tensor, y: torch.Tensor, max_len: int, temperature: float = 0.8) -> torch.Tensor:
        B = z.size(0)
        device = z.device

        yemb = self.class_emb(y)
        zcat = torch.cat([z, yemb], dim=1)  # (B,z+E)

        prev = torch.zeros((B, 1, 5), device=device)
        prev[:, :, 2] = 1.0  # start with pen_down

        h, c = None, None
        ended = torch.zeros((B,), dtype=torch.bool, device=device)
        outs = []

        for _ in range(max_len):
            inp = torch.cat([prev, zcat.unsqueeze(1)], dim=2)
            inp = self.in_fc(inp)
            out, (h, c) = self.lstm(inp, (h, c)) if h is not None else self.lstm(inp)

            dxdy_params = self.fc_dxdy(out)  # (B,1,4)
            pen_logits = self.fc_pen(out)    # (B,1,3)

            mu = dxdy_params[:, :, 0:2]
            logstd = dxdy_params[:, :, 2:4].clamp(-6, 2)
            std = torch.exp(logstd) * temperature
            dxdy = mu + std * torch.randn_like(std)

            logits = pen_logits[:, 0, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            pen = torch.multinomial(probs, num_samples=1).squeeze(1)  # 0/1/2

            step = torch.zeros((B, 5), device=device)
            step[:, 0:2] = dxdy[:, 0, :]
            step[torch.arange(B), 2 + pen] = 1.0

            step[ended, :] = 0.0
            step[ended, 4] = 1.0

            ended = ended | (pen == 2)
            outs.append(step.unsqueeze(1))
            prev = step.unsqueeze(1)

        return torch.cat(outs, dim=1)


class SketchVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encoder(seq, mask)
        z = self.reparameterize(mu, logvar)

        B, L, _ = seq.shape
        start = torch.zeros((B, 1, 5), device=seq.device)
        start[:, :, 2] = 1.0
        x_in = torch.cat([start, seq[:, :-1, :]], dim=1)

        dxdy_params, pen_logits = self.decoder(x_in, z, label)
        return {"mu": mu, "logvar": logvar, "z": z, "dxdy_params": dxdy_params, "pen_logits": pen_logits}


def loss_fn(out: Dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor, beta: float = 0.5) -> Dict[str, torch.Tensor]:
    dxdy_params = out["dxdy_params"]
    pen_logits = out["pen_logits"]
    mu, logvar = out["mu"], out["logvar"]

    t = target[:, :, 0:2]
    mean = dxdy_params[:, :, 0:2]
    logstd = dxdy_params[:, :, 2:4].clamp(-6, 2)
    var = torch.exp(2 * logstd)

    # Gaussian NLL (dx,dy)
    nll = 0.5 * (((t - mean) ** 2) / (var + 1e-6) + 2 * logstd + torch.log(torch.tensor(2.0 * 3.1415926, device=t.device)))
    nll = nll.sum(dim=2)  # (B,L)
    rec_loss = (nll * mask).sum() / (mask.sum() + 1e-6)

    # pen CE
    pen_target = target[:, :, 2:5].argmax(dim=2)
    ce = F.cross_entropy(pen_logits.reshape(-1, 3), pen_target.reshape(-1), reduction="none").reshape_as(mask)
    pen_loss = (ce * mask).sum() / (mask.sum() + 1e-6)

    # KL
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl.mean()

    total = rec_loss + pen_loss + beta * kl_loss
    return {"total": total, "rec": rec_loss, "pen": pen_loss, "kl": kl_loss}
