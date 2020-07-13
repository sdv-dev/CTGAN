import numpy as np

import torch
from torch import optim
from torch.nn import functional

from .data_sampler import DataSampler
from .data_transformer import DataTransformer
from .models import Discriminator, Generator


class CTGANSynthesizer(object):
  """Conditional Table GAN Synthesizer.

  This is the core class of the CTGAN project, where the different components
  are orchestrated together.

  For more details about the process, please check the [Modeling Tabular data using
  Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

  Args:
      z_dim (int):
          Size of the random sample passed to the Generator. Defaults to 128.
      gen_dims (tuple or list of ints):
          Size of the output samples for each one of the Residuals. A Resiudal
          Layer will be created for each one of the values provided.
          Defaults to (256, 256).
      dis_dims (tuple or list of ints):
          Size of the output samples for each one of the self._dis Layers.
          A Linear Layer will be created for each one of the values provided.
          Defaults to (256, 256).
      gen_lr (float):
          Learning rate for the generator. Defaults to 2e-4.
      gen_decay (float):
          Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
      dis_lr (float):
          Learning rate for the self._dis. Defaults to 2e-4.
      dis_decay (float):
          self._dis weight decay for the Adam Optimizer. Defaults to 1e-6.
      batch_size (int):
          Number of data samples to process in each step.
  """

  def __init__(self, z_dim=128, gen_dims=(256, 256), dis_dims=(256, 256),
               gen_lr=2e-4, gen_decay=1e-6, dis_lr=2e-4, dis_decay=0,
               batch_size=500):

    assert batch_size % 2 == 0
    self._z_dim = z_dim
    self._gen_dims = gen_dims
    self._dis_dims = dis_dims

    self._gen_lr = gen_lr
    self._gen_decay = gen_decay
    self._dis_lr = dis_lr
    self._dis_decay = dis_decay
    self._batch_size = batch_size
    self._device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

  def _apply_activate(self, data):
    "Apply proper activation function to the output of the generator."
    data_t = []
    st = 0
    for column_info in self._transformer.output_info():
      for span_info in column_info:
        if span_info.activation_fn == 'tanh':
          ed = st + span_info.dim
          data_t.append(torch.tanh(data[:, st:ed]))
          st = ed
        elif span_info.activation_fn == 'softmax':
          ed = st + span_info.dim
          data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
          st = ed
        else:
          assert 0

    return torch.cat(data_t, dim=1)

  def _cond_loss(self, data, c, m):
    "Compute the cross entropy loss on the fixed discrete column."
    loss = []
    st = 0
    st_c = 0
    skip = False
    for column_info in self._transformer.output_info():
      for span_info in column_info:
        if len(column_info) != 1 or span_info.activation_fn != "softmax":
          # not discrete column
          st += span_info.dim
        else:
          ed = st + span_info.dim
          ed_c = st_c + span_info.dim
          tmp = functional.cross_entropy(
              data[:, st:ed],
              torch.argmax(c[:, st_c:ed_c], dim=1),
              reduction='none'
          )
          loss.append(tmp)
          st = ed
          st_c = ed_c

    loss = torch.stack(loss, dim=1)

    return (loss * m).sum() / data.size()[0]

  def fit(self, train_data, discrete_columns=tuple(), epochs=300,
          log_frequency=True):
    """Fit the CTGAN Synthesizer models to the training data.

    Args:
        train_data (numpy.ndarray or pandas.DataFrame):
            Training Data. It must be a 2-dimensional numpy array or a
            pandas.DataFrame.
        discrete_columns (list-like):
            List of discrete columns to be used to generate the Conditional
            Vector. If ``train_data`` is a Numpy array, this list should
            contain the integer indices of the columns. Otherwise, if it is
            a ``pandas.DataFrame``, this list should contain the column names.
        epochs (int):
            Number of training epochs. Defaults to 300.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in date
            conditional sampling. Defaults to ``True``."""

    self._transformer = DataTransformer()
    self._transformer.fit(train_data, discrete_columns)
    train_data = self._transformer.transform(train_data)

    self._data_sampler = DataSampler(
        train_data, self._transformer.output_info(), log_frequency)

    data_dim = self._transformer.output_dim()

    self._gen = Generator(
        self._z_dim + self._data_sampler.dim_cond_vec(),
        self._gen_dims,
        data_dim
    ).to(self._device)

    self._dis = Discriminator(
        data_dim + self._data_sampler.dim_cond_vec(),
        self._dis_dims
    ).to(self._device)

    optimizerG = optim.Adam(
        self._gen.parameters(), lr=self._gen_lr, betas=(0.5, 0.9),
        weight_decay=self._gen_decay
    )
    optimizerD = optim.Adam(self._dis.parameters(),
                            lr=self._dis_lr, betas=(0.5, 0.9),
                            weight_decay=self._dis_decay)

    mean = torch.zeros(self._batch_size, self._z_dim, device=self._device)
    std = mean + 1

    steps_per_epoch = max(len(train_data) // self._batch_size, 1)
    for i in range(epochs):
      for id_ in range(steps_per_epoch):
        fakez = torch.normal(mean=mean, std=std)

        condvec = self._data_sampler.sample_condvec(self._batch_size)
        if condvec is None:
          c1, m1, col, opt = None, None, None, None
          real = self._data_sampler.sample_data(self._batch_size, col, opt)
        else:
          c1, m1, col, opt = condvec
          c1 = torch.from_numpy(c1).to(self._device)
          m1 = torch.from_numpy(m1).to(self._device)
          fakez = torch.cat([fakez, c1], dim=1)

          perm = np.arange(self._batch_size)
          np.random.shuffle(perm)
          real = self._data_sampler.sample_data(
              self._batch_size, col[perm], opt[perm])
          c2 = c1[perm]

        fake = self._gen(fakez)
        fakeact = self._apply_activate(fake)

        real = torch.from_numpy(real.astype('float32')).to(self._device)

        if c1 is not None:
          fake_cat = torch.cat([fakeact, c1], dim=1)
          real_cat = torch.cat([real, c2], dim=1)
        else:
          real_cat = real
          fake_cat = fake

        y_fake = self._dis(fake_cat)
        y_real = self._dis(real_cat)

        pen = self._dis.calc_gradient_penalty(
            real_cat, fake_cat, self._device)
        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

        optimizerD.zero_grad()
        pen.backward(retain_graph=True)
        loss_d.backward()
        optimizerD.step()

        fakez = torch.normal(mean=mean, std=std)
        condvec = self._data_sampler.sample_condvec(self._batch_size)

        if condvec is None:
          c1, m1, col, opt = None, None, None, None
        else:
          c1, m1, col, opt = condvec
          c1 = torch.from_numpy(c1).to(self._device)
          m1 = torch.from_numpy(m1).to(self._device)
          fakez = torch.cat([fakez, c1], dim=1)

        fake = self._gen(fakez)
        fakeact = self._apply_activate(fake)

        if c1 is not None:
          y_fake = self._dis(torch.cat([fakeact, c1], dim=1))
        else:
          y_fake = self._dis(fakeact)

        if condvec is None:
          cross_entropy = 0
        else:
          cross_entropy = self._cond_loss(fake, c1, m1)

        loss_g = -torch.mean(y_fake) + cross_entropy

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

      print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
            (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
            flush=True)

  def sample(self, n):
    """Sample data similar to the training data.

    Args:
        n (int):
            Number of rows to sample.

    Returns:
        numpy.ndarray or pandas.DataFrame
    """

    steps = n // self._batch_size + 1
    data = []
    for i in range(steps):
      mean = torch.zeros(self._batch_size, self._z_dim)
      std = mean + 1
      fakez = torch.normal(mean=mean, std=std).to(self._device)

      condvec = self._data_sampler.sample_original_condvec(self._batch_size)
      if condvec is None:
        pass
      else:
        c1 = condvec
        c1 = torch.from_numpy(c1).to(self._device)
        fakez = torch.cat([fakez, c1], dim=1)

      fake = self._gen(fakez)
      fakeact = self._apply_activate(fake)
      data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n]

    return self._transformer.inverse_transform(data)
