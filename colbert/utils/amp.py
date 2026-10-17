import torch

from contextlib import contextmanager
from colbert.utils.utils import NullContextManager, print_message


class MixedPrecisionManager():
    def __init__(self, activated, dtype=torch.float16):
        self.activated = activated
        self.dtype = dtype

        self.scaler = torch.cuda.amp.GradScaler() if activated and dtype == torch.float16 else None

    def context(self):
        return torch.cuda.amp.autocast(dtype=self.dtype) if self.activated else NullContextManager()

    def backward(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer, scheduler=None):
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)

            self.scaler.step(optimizer)
            self.scaler.update()
        elif self.activated:
            grad_norm = torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0, error_if_nonfinite=False)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print_message(f"#> Skipping optimizer step: gradient norm is {grad_norm.item()}.")
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
