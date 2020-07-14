import os

from tensorboardX import SummaryWriter

from utils import *


class rbm_base:
    def __init__(self, model_path, v_sz=784, h_sz=256, pcd=True,
                 drop_probs=0.0, n_epoch_to_save=1, im_shape=[1, 28, 28],
                 W_init=None, vb_init=None, hb_init=None, metrics_interval=200,
                 init_lr=1e-2, lr_start=2, lr_stop=8, ultimate_lr=2e-5,
                 n_gibbs_steps=1, sample_v_states=False, sample_h_states=True,
                 init_mo=0.5, ultimate_mo=0.8, mo_start=0, mo_stop=5,
                 sparsity_target=0.1, sparsity_damping=0.9, sparsity_cost=0.0,
                 max_epoch=10, batch_size=16, l2=1e-4, verbose=True):

        self.model_path = model_path
        self.im_shape = im_shape
        self.drop_probs = drop_probs
        self.pcd = pcd
        self.persistent_chains = None
        self.n_epoch_to_save = n_epoch_to_save
        self.v_sz = v_sz
        self.h_sz = h_sz

        self.W_init = W_init
        self.vb_init = vb_init
        self.hb_init = hb_init

        self.metrics_interval = metrics_interval
        self.verbose = verbose

        self.sparsity_target = sparsity_target
        self.sparsity_damping = sparsity_damping
        self.sparsity_cost = sparsity_cost

        self.init_lr = init_lr
        self.lr_start_decay = lr_start
        self.lr_stop_decay = lr_stop
        self.ultimate_lr = ultimate_lr
        self.init_mo = init_mo
        self.ultimate_mo = ultimate_mo
        self.mo_stop_decay = mo_stop
        self.mo_start_decay = mo_start

        self.sample_v_states = sample_v_states
        self.sample_h_states = sample_h_states
        self.n_gibbs_steps = n_gibbs_steps

        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.l2 = l2

        # 带下划线的成员，是变化的，是save要保存的变量
        self._step = 0
        self._lr = self.init_lr
        self._mo = self.init_mo

        self._q_mean = Tensor(np.zeros(self.h_sz))

        self._dW = Tensor(np.zeros([self.v_sz, self.h_sz]))
        self._dhb = Tensor(np.zeros(self.h_sz))
        self._dvb = Tensor(np.zeros(self.v_sz))

        self._W = T.nn.init.xavier_normal(
            Tensor(np.zeros([self.v_sz, self.h_sz]))) if self.W_init is None else self.W_init
        self._hb = Tensor(np.zeros(self.h_sz)) if self.hb_init is None else self.hb_init
        self._vb = Tensor(np.zeros(self.v_sz)) if self.vb_init is None else self.vb_init

    def _free_energy(self, v):
        raise NotImplementedError

    def _h_given_v(self, v):
        raise NotImplementedError

    def _v_given_h(self, h):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def _dropout(self, X, drop_probs):
        assert 0 <= drop_probs < 1
        X *= T.bernoulli((1 - drop_probs) * T.ones_like(X)) / (1 - drop_probs)
        return X

    def _msre_metric(self, v):
        v_ = self._gibbs_sample(self._h_given_v(v)[0], self.n_gibbs_steps)[0]
        return T.mean((v - v_) ** 2)

    def _free_energy_gap_metric(self, train, valid, batch_size):
        train_feg = self._free_energy(shuffle_batch(train, batch_size).view(-1, self.v_sz))
        valid_feg = self._free_energy(shuffle_batch(valid, batch_size).view(-1, self.v_sz))
        return train_feg - valid_feg

    def _lr_decay(self):
        lr = self._lr + linear_inc(self.init_lr, self.ultimate_lr,
                                   self.lr_start_decay, self.lr_stop_decay)
        self._lr = lr if lr > self.ultimate_lr else self.ultimate_lr

    def _mo_decay(self):
        mo = self._mo + linear_inc(self.init_mo, self.ultimate_mo,
                                   self.mo_start_decay, self.mo_stop_decay)
        self._mo = mo if mo < self.ultimate_mo else self.ultimate_mo

    def _gibbs_step(self, h0):
        v_probs, v_samples = self._v_given_h(h0)
        v = v_samples if self.sample_v_states else v_probs
        h_probs, h_samples = self._h_given_v(v)
        h = h_samples if self.sample_h_states else h_probs
        return v, h

    def _gibbs_sample(self, h0, n_gibbs_steps):
        v, h = None, h0
        for _ in range(n_gibbs_steps):
            v, h = self._gibbs_step(h)
        return v, h

    def _gibbs_chain(self, v0):
        h0 = self._h_given_v(v0)[1]
        h_gibbs = self.persistent_chains if self.pcd else h0
        vn, hn = self._gibbs_sample(h_gibbs, self.n_gibbs_steps)
        self.persistent_chains = hn
        return h0, v0, hn, vn

    def _update(self, h0, v0, hn, vn):
        N = v0.size()[0]
        # 添加稀疏化正则项
        q_means = T.mean(hn, 0)
        self._q_mean = self.sparsity_damping * self._q_mean + (1 - self.sparsity_damping) * q_means
        sparsity_penalty = self.sparsity_cost * (self._q_mean - self.sparsity_target)
        dW = (vn.t() @ hn - v0.t() @ h0) / N + self.l2 * self._W + sparsity_penalty
        dvb = T.mean(vn - v0, 0)
        dhb = T.mean(hn - h0, 0) + sparsity_penalty

        self._dW = self._mo * self._dW + self._lr * dW
        self._dvb = self._mo * self._dvb + self._lr * dvb
        self._dhb = self._mo * self._dhb + self._lr * dhb
        self._W = self._W - self._dW
        self._vb = self._vb - self._dvb
        self._hb = self._hb - self._dhb

    def fit(self, X, X_val):
        writer = SummaryWriter(self.model_path)
        self.persistent_chains = self._h_given_v(
            shuffle_batch(X, self.batch_size).view(self.batch_size, self.v_sz))[1]
        for epoch in range(self.max_epoch):
            self._lr_decay()
            self._mo_decay()
            for X_batch in next_batch(X, self.batch_size):
                X_batch = X_batch.view(-1, self.v_sz)
                X_batch = self._dropout(X_batch, self.drop_probs)
                h0, v0, hn, vn = self._gibbs_chain(X_batch)
                self._update(h0, v0, hn, vn)

                self._step += 1
                if (self._step + 1) % self.metrics_interval == 0:
                    free_energy_gap = self._free_energy_gap_metric(X, X_val, 800)
                    msre = self._msre_metric(X_batch)
                    writer.add_scalar('train_free_energy', free_energy_gap, self._step)
                    writer.add_scalar('train_msre', msre, self._step)
                    writer.add_scalar('lr', self._lr, self._step)
                    writer.add_scalar('mo', self._mo, self._step)
                    writer.add_histogram('W', self._W, self._step)
                    writer.add_histogram('v_bias', self._vb, self._step)
                    writer.add_histogram('h_bias', self._hb, self._step)
                    writer.add_histogram('dW', self._dW, self._step)
                    if self.im_shape[0] == 3:  # tensorboardX 不支持灰度图
                        writer.add_image('filters', self._filters(4))
                    else:
                        tv.utils.save_image(vn.view([-1] + self.im_shape),
                                            self.model_path + 'verbose/train_{}.png'.format(self._step))

                    if self.verbose:
                        print('epoch: [%d \ %d] global_step: [%d] train_msre: [%3f]' % (
                            epoch + 1, self.max_epoch, self._step, msre))

            # save
            if (epoch + 1) % self.n_epoch_to_save == 0:
                self.save()
        writer.close()

    def inf_from_valid(self, batch_X_val, n_gibbs_steps):
        batch_X_val = batch_X_val.view([-1, self.v_sz])
        h0_val, _ = self._h_given_v(batch_X_val)
        return self._gibbs_sample(h0_val, n_gibbs_steps)[0]

    def inf_by_stochastic(self, batch_size, n_gibbs_steps):
        h0_sto = Tensor(np.random.normal(1e-8, 0.02, size=[batch_size, self.h_sz]))
        return self._gibbs_sample(h0_sto, n_gibbs_steps)[0]

    def _filters(self, n_filter=64):
        assert n_filter < self.h_sz
        filter_ = self._W.t().contiguous()[:n_filter, :]
        return filter_.view([n_filter] + self.im_shape)

    def _save(self, **kwargs):
        np.savez(os.path.join(self.model_path + 'ckpt_{}.npz'.format(self._step)),
                 _step=self._step, _lr=self._lr, _mo=self._mo,
                 _W=self._W, _vb=self._vb, _hb=self._hb, _q_mean=self._q_mean,
                 _dW=self._dW, _dvb=self._dvb, _dhb=self._dhb, **kwargs)
        from shutil import copyfile
        copyfile(os.path.join(self.model_path + 'ckpt_{}.npz'.format(self._step)),
                 os.path.join(self.model_path + 'ckpt_latest.npz'))

    def _load(self, npz_file, *args):
        if not os.path.isfile(npz_file):
            return False
        npz = np.load(npz_file)
        self._W, self._vb, self._hb = Tensor(npz['_W']), Tensor(npz['_vb']), Tensor(npz['_hb'])
        self._step, self._mo, self._lr = int(npz['_step']), float(npz['_mo']), float(npz['_lr'])
        self._dW, self._dhb, self._dvb = Tensor(npz['_dW']), Tensor(npz['_dhb']), Tensor(npz['_dvb'])
        self._q_mean = Tensor(npz['_q_mean'])

        return [npz[t] for t in args]
