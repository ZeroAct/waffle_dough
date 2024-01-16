import logging
import traceback


class BaseException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        stack = traceback.extract_stack()
        logger = logging.getLogger(stack[-2].filename)
        logger.error(self.__str__(), stacklevel=2)
