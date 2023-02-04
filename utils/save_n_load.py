import os

import torch

try:
    import moxing as mox

    # mox.file.shift('os', 'mox')
    run_on_remote = True
except:
    run_on_remote = False
from utils.config import cfg
from utils.log import log, logger


def lstrip_state_dict(state_dict, strip='module.'):
    return {(k[len(strip):] if k.startswith(strip) else k): v
            for k, v in state_dict.items()}


def save_model(model, name, cfg, logger, message=None):
    if cfg.rank == 0:
        output_path = os.path.join(cfg.OUTPUT_DIR, name)
        logger.info(f'saving model to {output_path}')
        if message is not None:
            logger.info(message)
        try:
            if output_path.startswith('s3://'):
                torch.save(model.state_dict(), '/cache/temp.pth')
                mox.file.copy('/cache/temp.pth', output_path)
            else:
                torch.save(model.state_dict(), output_path)
        except Exception as e:
            logger.error(str(e))
        else:
            logger.info('model saved.')


def load_model(model, path, device, logger):
    logger.info('loading model weight...Path: ' + str(path))
    if str(path).startswith('s3://'):
        mox.file.copy(path, '/cache/temp.pth')
        path = '/cache/temp.pth'
    state_dict = torch.load(
        path,
        map_location=device,
    )
    state_dict = lstrip_state_dict(state_dict, 'module.')
    del state_dict['head.0.weight']
    del state_dict['head.0.bias']
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )
    logger.info('missing_keys: ' + str(missing_keys))
    logger.info('unexpected_keys: ' + str(unexpected_keys))
    logger.info('done.')
