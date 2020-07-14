from base import rbm_base
from utils import *

softplus = lambda x: T.log(1 + T.exp(x))


class BernoulliRBM(rbm_base):
    """RBM with Bernoulli both visible and hidden units."""

    def __init__(self, model_path='./logs/brbm/', *args, **kwargs):
        super(BernoulliRBM, self).__init__(model_path=model_path, *args, **kwargs)

    def _h_given_v(self, v):
        h_probs = T.sigmoid(v @ self._W + self._hb)
        h_samples = T.bernoulli(h_probs)
        return h_probs, h_samples

    def _v_given_h(self, h):
        v_probs = T.sigmoid(h @ self._W.t() + self._vb)
        v_samples = T.bernoulli(v_probs)
        return v_probs, v_samples

    def _free_energy(self, v):
        h = v @ self._W + self._hb
        t1 = - v @ self._vb
        t2 = - T.sum(softplus(h), 1)
        return T.mean(t1 + t2)

    def save(self):
        self._save()

    def load(self):
        self._load(self.model_path + 'ckpt_latest.npz')


class GaussianBRBM(rbm_base):
    """RBM with Gaussian visible and Bernoulli hidden units.

    This implementation does not learn variances, but instead uses
    fixed, predetermined values. Input data should be pre-processed
    to have zero mean (or, equivalently, initialize visible biases
    to the negative mean of data). It can also be normalized to have
    unit variance. In the latter case use `sigma` equal to 1., as
    suggested in [1].
    """

    def __init__(self, model_path='./logs/gbrbm/',
                 is_fixed_sigma=True, *args, **kwargs):
        super(GaussianBRBM, self).__init__(
            model_path=model_path, *args, **kwargs)
        self._sigma = Tensor(np.ones(self.v_sz))
        self.is_fixed_sigma = is_fixed_sigma

    def _h_given_v(self, v):
        h_probs = T.sigmoid(v / self._sigma @ self._W + self._hb)
        h_samples = T.bernoulli(h_probs)
        return h_probs, h_samples

    def _v_given_h(self, h):
        v_mean = self._sigma * (h @ self._W.t()) + self._vb
        v_samples = T.normal(v_mean, std=self._sigma.clone().expand_as(v_mean))
        return v_mean, v_samples

    def _free_energy(self, v):
        return 0

    def save(self):
        self._save(_sigma=self._sigma)

    def load(self):
        _sigma = self._load(self.model_path + 'ckpt_latest.npz', '_sigma')
        self._sigma = Tensor(_sigma[0])
