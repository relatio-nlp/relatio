# MIT License

# Copyright (c) 2020-2021 ETH Zurich, Andrei V. Plamada
# Copyright (c) 2020-2021 ETH Zurich, Elliott Ash
# Copyright (c) 2020-2021 University of St.Gallen, Philine Widmer
# Copyright (c) 2020-2021 Ecole Polytechnique, Germain Gauthier

import logging
from pathlib import Path


class FileLogger:
    def __init__(self, path=Path("relatio.log")):
        LOGGING_PATH = path
        self._logger = logging.getLogger("py.warnings")
        logging.captureWarnings(True)

        # Create handlers
        handler = logging.FileHandler(LOGGING_PATH, mode="w")
        handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(format)

        # Add handlers to the logger
        self._logger.addHandler(handler)

    def close(self):
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
