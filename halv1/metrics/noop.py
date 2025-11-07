# metrics/noop.py
class _No:
    def labels(self, *a, **kw): return self
    def observe(self, *a, **kw): pass
    def inc(self, *a, **kw): pass
    def dec(self, *a, **kw): pass
    def set(self, *a, **kw): pass

LLM_LATENCY = LLM_TOKENS_INFLIGHT = AB_ASSIGN = COORDINATOR_DECISION = ERRORS = _No()
