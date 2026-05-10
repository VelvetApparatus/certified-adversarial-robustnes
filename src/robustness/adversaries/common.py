class Adversary:
    def __init__(self, name, params=None, eval_mode=True):
        self.name = name
        self.params = params or {}
        self.eval_mode = eval_mode

    def __repr__(self):
        return f"<Adversary name={self.name} params={self.params}>"

    def gen(self, model, x, y):
        if not self.eval_mode:
            return self._gen(model, x, y)

        is_train = model.training
        try:
            model.eval()
            return self._gen(model, x, y)
        finally:
            model.train(is_train)

    def _gen(self, model, X, y):
        raise NotImplementedError
