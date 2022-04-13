import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(""))
logger = logging.getLogger("trainer")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
