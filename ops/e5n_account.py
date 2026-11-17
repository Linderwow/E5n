import math

MODES = ('Fixed', 'Evaluation', 'Funded')


class Sizer:
    def __init__(self, mode, start, drawdown, lock_above, target, per_unit, max_units, high=0.0):
        self.mode, self.start, self.drawdown, self.lock_above, self.target, self.per_unit = mode, start, drawdown, lock_above, target, per_unit
        self.max_units = max(1, int(max_units)); self.high = 0.0 if high is None or math.isnan(high) else max(0.0, high)

    @property
    def threshold(self): return min(self.high - self.drawdown, self.lock_above)

    def cushion(self, balance): return balance - self.start - self.threshold

    def close(self, balance):
        if balance is not None and math.isfinite(balance): self.high = max(self.high, balance - self.start)

    def units(self, balance):
        if self.mode != 'Funded' or balance is None or not math.isfinite(balance) or self.per_unit <= 0: return 1
        u = math.floor(self.cushion(balance) / self.per_unit)
        return 1 if u < 1 else self.max_units if u > self.max_units else int(u)

    def funded_size(self, balance, band_only_until_lock):
        band_only = self.mode == 'Funded' and bool(band_only_until_lock) and self.threshold < self.lock_above
        return (1, True) if band_only else (self.units(balance), False)

    def target_reached(self, equity): return self.mode == 'Evaluation' and equity is not None and not math.isnan(equity) and equity - self.start >= self.target
