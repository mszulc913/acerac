import csv
from abc import ABC, abstractmethod
import datetime
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Any


class Logger(ABC):
    def __init__(self, keys: List[str]):
        """Collects key-values pairs.
        This logged automatically appends additional key-value pair: timestamp of the event.

        Args:
            keys: List of keys (strings) to be logged.
        """
        self._data = defaultdict(dict)
        self._keys = keys + ['timestamp']

    def log_values(self, key_values: Dict[str, Any]):
        """Stores single data row.
        Values outside of the list defined in the constructor are omitted.

        Args:
            key_values: Dictionary with keys and values to be stored.
        """
        stored_row = OrderedDict()
        for key in self._keys:
            stored_row[key] = key_values.get(key)

        stored_row['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._store_log(stored_row)

    @abstractmethod
    def _store_log(self, key_values: Dict[str, Any]):
        ...


class CSVLogger(Logger):

    def __init__(self, file_path: Path, *args, **kwargs):
        """Logs data into csv file.
        IMPORTANT: close() method has to be called at the end of the run.

        Args:
            file_path: Path to the log file.
        """
        super().__init__(*args, **kwargs)
        self._file = open(str(file_path), 'wt')
        self._file_path = file_path
        self._writer = csv.writer(self._file, delimiter=',')
        self._writer.writerow(self._keys)

    def _store_log(self, key_values: OrderedDict):
        row = list(key_values.values())
        self._writer.writerow(row)

    def dump(self):
        """Appends collected data at the end of currently opened file."""
        self._file.close()
        self._file = open(str(self._file_path), 'a+t')
        self._writer = csv.writer(self._file, delimiter=',')

    def close(self):
        """Closes opened file."""
        self._file.close()
