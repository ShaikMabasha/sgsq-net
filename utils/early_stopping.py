"""
Simple early stopping helper, shared across all training scripts.

Monitors a validation metric each epoch; stops training if it hasn't
improved for `patience` consecutive epochs. Works with metrics where higher
is better (Dice, QWK, accuracy) via mode="max", or lower is better (loss)
via mode="min".
"""


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Call once per epoch with the current validation metric. Returns True if this is a new best."""
        if self.best_score is None:
            self.best_score = score
            return True

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
            else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


if __name__ == "__main__":
    es = EarlyStopping(patience=3, mode="max")
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62, 0.90]
    stopped_at = None
    for i, s in enumerate(scores):
        is_best = es.step(s)
        if es.should_stop and stopped_at is None:
            stopped_at = i
    assert stopped_at == 5, f"expected stop at index 5, got {stopped_at}"
    print(f"Stopped at epoch index {stopped_at} (as expected, patience=3 after epoch 2's peak of 0.65)")
    print("OK")
