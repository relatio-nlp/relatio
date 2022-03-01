# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

import logging
from pathlib import Path


class FileLogger:
    """
    File logger used to capture the warnings and write the warnings to a file.

    The default warnings can be integrated with the logging using the logging.captureWarnings(True) function.
    See https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module .
    The logs are saved in file that is created if it does not exists or the content is truncated (write mode).
    One can use at most one instance of the class.

    Parameters
    ----------
    file : pathlib.Path, default=pathlib.Path("relatio.log")
        The file path used for savings the logs.
    capture_warnings : bool, default=True
        Whether to capture the default warnings.


    Attributes
    ----------
    capture_warnings : bool
        If capture_warnings is true the warnings are logged. Otherwise they are not.


    Methods
    -------
    close()
        The file handler is properly closed.


    """

    # This class can be used only once
    _used: bool = False

    def __init__(self, file: Path = Path("relatio.log"), capture_warnings: bool = True):
        if FileLogger._used is True:
            raise RuntimeError("Only one instance is allowed.")
        else:
            FileLogger._used = True

        self._logger = logging.getLogger("py.warnings")
        # Avoid propagating the logs to the root
        self._logger.propagate = False

        self._capture_warnings: bool = capture_warnings
        if capture_warnings is True:
            logging.captureWarnings(True)

        # Create handlers
        self._handler = logging.FileHandler(filename=file, mode="w")
        self._handler.setLevel(logging.WARN)

        # Create formatters and add it to handlers
        format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._handler.setFormatter(format)

        # Add handlers to the logger
        self._logger.addHandler(self._handler)

    @property
    def capture_warnings(self):
        return self._capture_warnings

    @capture_warnings.setter
    def capture_warnings(self, value: bool):
        logging.captureWarnings(value)
        self._capture_warnings = value

    def close(self):
        self._handler.close()
