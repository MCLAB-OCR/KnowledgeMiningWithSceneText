import re


def set_parameter_requires_grad(model,
                                target_pattern,
                                requires_grad,
                                set_others=False):
    for param in model.named_parameters():
        if set_others:
            if re.search(target_pattern, param[0]) is None:
                param[1].requires_grad = requires_grad
        else:
            if re.search(target_pattern, param[0]) is not None:
                param[1].requires_grad = requires_grad
