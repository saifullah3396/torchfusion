from datadings.writer import FileWriter
from datasets.utils import logging

logger = logging.get_logger(__name__)


class MsgpackWriter(FileWriter):
    """
    A msgpack writer class compatible with huggingface datasets.
    """

    def __init__(self, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_bytes = 0
        self._features = features

    def _write_data(self, key, packed):
        if key in self._keys_set:
            raise ValueError("duplicate key %r not allowed" % key)
        self._keys.append(key)
        self._keys_set.add(key)
        self._hash.update(packed)
        self._num_bytes += self._outfile.write(packed)
        self._offsets.append(self._outfile.tell())
        self.written += 1
        self._printer()

    def write(self, sample, key):
        self._write_data(key, self._packer.pack(sample))

    def finalize(self):
        return self._num_bytes, self.written
