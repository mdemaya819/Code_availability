
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pinball_loss_torch(y, q_pred, quantiles, device):
    # y (B,H) ; q_pred (B,H,Q)
    q = torch.tensor(quantiles, device=device).view(1, 1, -1)
    e = y.unsqueeze(-1) - q_pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))


def train_model(model, train_ds, val_ds, *, quantiles=None, epochs=50,
                batch_size=256, lr=1e-3, weight_decay=1e-4, patience=8,
                num_workers=0, device=None, seed=0, verbose=True):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3,
                                                       factor=0.5)
    is_q = quantiles is not None and len(quantiles) > 1
    mse = nn.MSELoss()
    pin = (device == "cuda")
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, drop_last=False, pin_memory=pin)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin)

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train(); tr = 0.0; n = 0
        for X, Y in tl:
            X, Y = X.to(device), Y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = (pinball_loss_torch(Y, pred, quantiles, device)
                    if is_q else mse(pred, Y))
            if not torch.isfinite(loss):        # garde-fou : ne jamais propager un NaN
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(X); n += len(X)
        tr /= max(n, 1)

        model.eval(); va = 0.0; m = 0
        with torch.no_grad():
            for X, Y in vl:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                loss = (pinball_loss_torch(Y, pred, quantiles, device)
                        if is_q else mse(pred, Y))
                va += loss.item() * len(X); m += len(X)
        va /= max(m, 1)
        sched.step(va)
        if verbose:
            print(f"  epoch {ep:3d} | train {tr:.4f} | val {va:.4f}")
        if va < best - 1e-5:
            best, best_state, bad = va, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop @ epoch {ep} (best val {best:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best
