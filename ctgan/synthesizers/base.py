import logging

LOGGER = logging.getLogger(__name__)


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``.

    This should contain the save/load methods.
    """

    def save(self, filename):
        """Save the internal state of a copula in the specified filename.

        Args:
            filename(str): Path to save.
            
        Returns:
            None
        """
        content = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(content, f)

    @classmethod
    def load(cls, copula_path):
        """Create a new instance from a file.

        Args:
            copula_path(str): Path to file with the serialized copula.

        Returns:
            Bivariate: Instance with the parameters stored in the file.
        """
        with open(copula_path) as f:
            copula_dict = json.load(f)

        return cls.from_dict(copula_dict)