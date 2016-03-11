from pyparsing import ParseResults


def to_flat_dict(tokens, parent_key='', sep='_'):
    """
    Creates a flattened dictionary from the named values in tokens.

    E.g. suppose tokens.dump() output is
        - isotopic_comp: ['0.000629', '(', '7', ')']
            - nominal_value: 0.000629
            - std_dev: 7e-06
    Then the new dictionary is {'isotopic_comp_nominal_value': 0.000629, 'isotopic_comp_std_dev':7e-06}

    Parameters
    ----------
    tokens: ~pyparsing.ParseResults
    parent_key: ~str -- is used in recursive calls; you don't need to pass this
    sep: ~str -- is used to concatenate keys (default: "_")

    Returns: ~dict
    -------

    """
    tokens_dict = dict()
    for key, item in tokens.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(item, ParseResults):
            tokens_dict.update(to_flat_dict(item, parent_key=new_key, sep=sep))
        else:
            tokens_dict[new_key] = item
    return tokens_dict


def to_nom_val_and_std_dev(interval):
    """
    For a given interval [mu - sigma, mu + sigma] returns (mu, sigma)
    (Here mu is nominal value and sigma is standard deviation)

    Parameters
    ----------
    interval: ~list [lwr_bnd, upr_bnd]

    Returns: ~tuple

    """
    lwr_bnd, upr_bnd = interval
    sigma = (upr_bnd - lwr_bnd)/2
    mu = lwr_bnd + sigma
    return (mu, sigma)