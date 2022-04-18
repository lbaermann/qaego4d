from absl.flags.argparse_flags import ArgumentParser
from omegaconf import DictConfig, open_dict


def apply_argparse_defaults_to_hydra_config(config: DictConfig, parser: ArgumentParser, verbose=False):
    args = parser.parse_args([])  # Parser is not allowed to have required args, otherwise this will fail!
    defaults = vars(args)

    def _apply_defaults(dest: DictConfig, source: dict, indentation=''):
        for k, v in source.items():
            if k in dest and isinstance(v, dict):
                current_value = dest[k]
                if current_value is not None:
                    assert isinstance(current_value, DictConfig)
                    _apply_defaults(current_value, v, indentation + ' ')
            elif k not in dest:
                dest[k] = v
                if verbose:
                    print(indentation, 'set default value for', k)

    with open_dict(config):
        _apply_defaults(config, defaults)
