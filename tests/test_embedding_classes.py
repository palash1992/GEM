import unittest

from gem.embedding.gf import GraphFactorization
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.lle import LocallyLinearEmbedding
from gem.embedding.node2vec import node2vec
from gem.embedding.sdne import SDNE


class EmbeddingsTest(unittest.TestCase):

    def test_hope(self):
        model = HOPE()
        self._run(model)

    def test_le(self):
        model = LaplacianEigenmaps()
        self._run(model)

    def test_lle(self):
        model = LocallyLinearEmbedding()
        self._run(model)

    def test_node2vec(self):
        model = node2vec()
        self._run(model)

    def test_sdne(self):
        model = SDNE()
        self._run(model)

    def test_gf(self):
        model = GraphFactorization()
        self._run(model)

    def _run(self, model):
        try:
            model.learn_embedding()
        except ValueError:
            pass
        # check method name
        self.assertEqual(model.hyper_params['method_name'], model.get_method_name())
        # check unlearned prediction
        try:
            model.get_embedding()
        except ValueError:
            pass
