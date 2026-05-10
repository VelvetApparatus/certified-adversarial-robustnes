class Adversary:
    def __init__(self, name, params=None, loss_fn=None):
        self.name = name
        self.params = params or {}
        self.loss_fn = loss_fn

    def __repr__(self):
        return f"<Adversary name={self.name} params={self.params}>"

    def __str__(self):
        return self.__repr__()

    def gen(self, model, x, y):
        is_train = model.training
        try:
            model.eval()
            return self._gen(model, x, y)
        finally:
            model.train(is_train)

    def _gen(self, model, X, y):
        raise NotImplementedError
