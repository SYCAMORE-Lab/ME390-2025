# Source: https://github.com/dmizr/phuber/blob/master/phuber/metrics.py
import torch


class LossMetric:
    """Keeps track of the loss over an epoch"""

    def __init__(self) -> None:
        self.running_loss = 0
        self.count = 0

    def update(self, loss: float, batch_size: int) -> None:
        self.running_loss += loss * batch_size
        self.count += batch_size

    def compute(self) -> float:
        return self.running_loss / self.count

    def reset(self) -> None:
        self.running_loss = 0
        self.count = 0


class AccuracyMetric:
    """Keeps track of the top-k accuracy over an epoch
    Args:
        k (int): Value of k for top-k accuracy
    """

    def __init__(self, k: int = 1) -> None:
        self.correct = 0
        self.total = 0
        self.k = k

    def update(self, out: torch.Tensor, target: torch.Tensor) -> None:
        # Computes top-k accuracy
        _, indices = torch.topk(out, self.k, dim=-1)
        target_in_top_k = torch.eq(indices, target[:, None]).bool().any(-1)
        total_correct = torch.sum(target_in_top_k, dtype=torch.int).item()
        total_samples = target.shape[0]

        self.correct += total_correct
        self.total += total_samples

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class MSEMetric:
    def __init__(self):
        self.squared_error_sum = 0.0
        self.count = 0

    def update(self, output, target):
        self.squared_error_sum += torch.sum((output - target) ** 2).item()
        self.count += target.size(0)

    def compute(self):
        return self.squared_error_sum / self.count if self.count != 0 else 0.0

    def reset(self):
        self.squared_error_sum = 0.0
        self.count = 0


class MAEMetric:
    def __init__(self):
        self.absolute_error_sum = 0.0
        self.count = 0

    def update(self, output, target):
        self.absolute_error_sum += torch.sum(torch.abs(output - target)).item()
        self.count += target.size(0)

    def compute(self):
        return self.absolute_error_sum / self.count if self.count != 0 else 0.0

    def reset(self):
        self.absolute_error_sum = 0.0
        self.count = 0


class R2Metric:
    def __init__(self):
        self.total_sum_of_squares = 0.0
        self.residual_sum_of_squares = 0.0
        self.count = 0

    def update(self, output, target):
        mean_target = torch.mean(target).item()
        self.total_sum_of_squares += torch.sum((target - mean_target) ** 2).item()
        self.residual_sum_of_squares += torch.sum((target - output) ** 2).item()
        self.count += target.size(0)

    def compute(self):
        if self.count == 0:
            return 0.0
        return 1 - (self.residual_sum_of_squares / self.total_sum_of_squares)

    def reset(self):
        self.total_sum_of_squares = 0.0
        self.residual_sum_of_squares = 0.0
        self.count = 0
        