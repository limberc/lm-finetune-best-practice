from torch.optim.lr_scheduler import _LRScheduler


class STLR(_LRScheduler):
    """
    Universal Language Model Fine-tuning for Text Classification
    https://arxiv.org/abs/1801.06146
    """
    def __init__(self, optimizer, lr_max, num_of_iters, cut_frac=0.1, ratio=32, last_epoch=-1):
        self.cut = num_of_iters * cut_frac
        self.lr_max = lr_max
        self.num_of_iters = num_of_iters
        self.cut_frac = cut_frac
        self.ratio = ratio
        super().__init__(optimizer, last_epoch)

    def _get_p(self, t):
        if t < self.cut:
            p = t / self.cut
        else:
            p = 1 - (t - self.cut) / (self.cut * (1 / self.cut_frac - 1))
        return p

    def get_lr(self):
        return [self.lr_max * (1 + self._get_p(self._step_count) * (self.ratio - 1)) / self.ratio]
