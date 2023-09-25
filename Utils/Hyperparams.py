import optuna


def ParamsAll(trial: optuna.Trial, params: dict):
    params_instance = {}
    for k, v in params.items():
        params_instance[k] = ParamsAny(trial, k, v)
    return params_instance


def ParamsAny(trial, name, param):
    if isinstance(param, list):
        return ParamsList(trial, name, param)
    elif isinstance(param, dict):
        return ParamsDict(trial, name, param)
    else:
        return param


def ParamsList(trial: optuna.Trial, name: str, param: list):
    return [
        ParamsAny(trial, '%s%d' % (name, i), su) for i, su in enumerate(param)
    ]


def ParamsDict(trial: optuna.Trial, name: str, param: dict):
    if 'type' in param:
        return getattr(trial, param['type'])(name, **param['params'])
    else:
        res = {}
        for k, v in param.items():
            res[k] = ParamsAny(trial, name, v)
        return res
