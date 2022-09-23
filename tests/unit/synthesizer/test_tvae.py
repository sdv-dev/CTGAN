"""TVAE unit testing module."""

from unittest import TestCase


class TestEncoder(TestCase):

    def test___init__(self):
        """Test `__init__` for a generic case.

        Make sure 'self.seq' has same length as 2*`compress_dims`.

        Setup:
            - Create Encoder

        Input:
            - data_dim = positive integer
            - compress_dims = list of integers
            - embedding_dim = positive integer

        Output:
            - None

        Side Effects:
            - Set `self.seq`, `self.fc1` and `self.fc2`
        """

    def test_forward(self):
        """Test `test_forward` for a generic case.

        Check that the output shapes are correct and that std is positive.
        We can also test that all parameters have a gradient attached to them
        by running `encoder.parameters()`. To do that, we just need to use `loss.backward()`
        for some loss, like `loss = torch.mean(mu) + torch.mean(std) + torch.mean(logvar)`.

        Setup:
            - Create random tensor

        Input:
            - input = random tensor of shape (N, data_dim)

        Output:
            - Tuple of (mu, std, logvar):
              mu - tensor of shape (N, embedding_dim)
              std - tensor of shape (N, embedding_dim), non-negative values
              logvar - tensor of shape (N, embedding_dim)
        """


class TestDecoder(TestCase):

    def test___init__(self):
        """Test `__init__` for a generic case.

        Make sure 'self.seq' has same length as 2*`decompress_dims` + 1.

        Setup:
            - Create Decoder

        Input:
            - data_dim = positive integer
            - decompress_dims = list of integers
            - embedding_dim = positive integer

        Output:
            - None

        Side Effects:
            - Set `self.seq`, `self.sigma`
        """


class TestLossFunction(TestCase):

    def test__loss_function(self):
        """Test `_loss_function`.

        Check loss values = to specific numbers.

        Setup:
            Build all the tensors, lists, etc.

        Input:
            recon_x = tensor of shape (N, data_dims)
            x = tensor of shape (N, data_dims)
            sigmas = tensor of shape (N,)
            mu = tensor of shape (N,)
            logvar = tensor of shape (N,)
            output_info = list of SpanInfo objects from the data transformer,
                          including at least 1 continuous and 1 discrete
            factor = scalar

        Output:
            reconstruction loss = scalar = f(recon_x, x, sigmas, output_info, factor)
            kld loss = scalar = f(logvar, mu)
        """


class TestTVAE(TestCase):

    def test_set_device(self):
        """Test 'set_device' if a GPU is available.

        Check that decoder/encoder can successfully be moved to the device.
        If the machine doesn't have a GPU, this test shouldn't run.

        Setup:
            - Move decoder/encoder to device

        Input:
            - device = string

        Output:
            None

        Side Effects:
            - Set `self._device` to `device`
            - Moves `self.decoder` to `self._device`

        Note:
            - Need to be careful when checking whether the encoder is actually set
            to the right device, since it's not saved (it's only used in fit).
        """
