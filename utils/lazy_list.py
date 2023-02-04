class LazyList(object):

    def __init__(self, func, using_cache=True):
        self.func = func
        self.using_cache = using_cache
        self.cache = dict()

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        return self._evaluate(key)

    def _evaluate(self, i):
        res = self.func(i)
        if self.using_cache:
            self.cache[i] = res
        return res
