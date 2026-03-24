import time


class ProgressLogger:
    def __init__(self, prefix="[UNAGI-MONITOR]"):
        self.prefix = prefix

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log(self, message):
        print(f"{self.prefix} {message}", flush=True)

    def should_report(self, finished, total, max_reports=50):
        if total <= 0:
            return True
        if finished <= 1 or finished >= total:
            return True
        step = max(1, total // max_reports)
        return finished % step == 0

    def progress(self, label, finished, total, start_ts, extra=None):
        elapsed = max(0.0, time.time() - start_ts)
        if finished > 0:
            eta = elapsed * (total - finished) / finished
        else:
            eta = 0.0
        suffix = f" | {extra}" if extra else ""
        self.log(
            f"{label} finished={finished}/{total} "
            f"elapsed={self._format_duration(elapsed)} "
            f"eta={self._format_duration(eta)}{suffix}"
        )
