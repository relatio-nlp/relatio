import logging
from pathlib import Path


class FileLogger:
    """
    File logger used to capture the warnings and the root logger.
    The default warnings can be integrated with the logging using the logging.captureWarnings(True) function.
    See https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module .
    The logs are saved to file in write mode, and one can use the same configuration for the root logger.
    One can use at most one instance of the class and it is recommended to be called in the very begging.
    Parameters
    ----------
    file : pathlib.Path, default=pathlib.Path("relatio.log")
        The file path used for savings the logs.
    capture_warnings : bool, default=True
        Whether to capture the default warnings.
    include_root_logger : bool, default=True
        Whether to use the file also for the root logger.
    level : str, default="INFO"
        Which logging level to use (see https://docs.python.org/3/library/logging.html#logging-levels ).
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

    def __init__(
        self,
        file: Path = Path("relatio.log"),
        capture_warnings: bool = True,
        include_root_logger: bool = True,
        level: str = "INFO",
    ):
        if FileLogger._used is True:
            raise RuntimeError("Only one instance is allowed.")
        else:
            FileLogger._used = True

        self.capture_warnings = capture_warnings

        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Create handler
        self._handler = logging.FileHandler(filename=file, mode="w")
        self._handler.setLevel(level)
        self._handler.setFormatter(logging.Formatter(format))

        if include_root_logger is True:
            logging.basicConfig(handlers=[self._handler], level=level, format=format)
        else:
            self._logger = logging.getLogger("py.warnings")
            # Avoid propagating the logs to the root
            self._logger.propagate = False
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
