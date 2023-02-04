# import os
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format=
#     f'[%(asctime)s][RANK={int(os.environ["RANK"]):02d}][%(filename)s][line:%(lineno)d][%(levelname)s]: %(message)s',
# )
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False


def log(*obj):
    for x in obj:
        logger.info(str(x))
