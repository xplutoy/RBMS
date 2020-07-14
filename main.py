from rbms import BernoulliRBM, GaussianBRBM
from utils import *

brbm_cd = BernoulliRBM(
    model_path='./logs/brbm_cd/',
    pcd=False,
    drop_probs=0.2,
    lr_start=2,
    sparsity_target=0.1,
    lr_stop=28,
    n_gibbs_steps=10,
    max_epoch=30,
    sample_h_states=True,
    sample_v_states=True,
    batch_size=10,
    h_sz=256,
    l2=0.001)

brbm_pcd = BernoulliRBM(
    model_path='./logs/brbm_pcd/',
    pcd=True,
    lr_start=2,
    sparsity_target=0.1,
    lr_stop=18,
    n_gibbs_steps=2,
    max_epoch=20,
    sample_h_states=False,
    sample_v_states=True,
    batch_size=32,
    h_sz=256,
    l2=0.01)

gbrbm = GaussianBRBM(
    model_path='./logs/grbm_pcd/',
    pcd=True,
    drop_probs=0.1,
    init_lr=5e-4,
    ultimate_lr=1e-6,
    lr_start=5,
    sparsity_target=0.1,
    lr_stop=595,
    metrics_interval=600,
    n_gibbs_steps=2,
    max_epoch=600,
    sample_h_states=False,
    sample_v_states=True,
    batch_size=16,
    h_sz=512,
    l2=1e-4
)


def basic_test(model, train_ds, test_ds, train=True):
    if train:
        model.fit(train_ds, test_ds)
    else:
        model.load()

    batch_size = 64
    sto_path_lst = []
    val_path_lst = []
    for _step in range(20, 1000, 20):
        samples = model.inf_by_stochastic(batch_size, _step)
        sto_filename = model.model_path + 'images/sto_{}.png'.format(_step)
        sto_path_lst.append(sto_filename)
        tv.utils.save_image(samples.view(-1, 1, 28, 28), sto_filename)

        X_val = shuffle_batch(mnist_test, batch_size)
        samples = model.inf_from_valid(X_val, _step)
        val_filename = model.model_path + 'images/val_{}.png'.format(_step)
        val_path_lst.append(val_filename)
        tv.utils.save_image(samples.view(-1, 1, 28, 28), val_filename)

    to_gif(sto_path_lst, model.model_path + 'sto.gif')
    to_gif(val_path_lst, model.model_path + 'val.gif')


if __name__ == '__main__':
    # basic_test(brbm_pcd, mnist_train, mnist_test, train=False)
    basic_test(gbrbm, mnist_train, mnist_test, train=True)
