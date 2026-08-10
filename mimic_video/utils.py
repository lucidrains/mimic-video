import importlib

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def check_import(library, min_version = None, error_message = None):
    try:
        module = importlib.import_module(library)
    except ImportError:
        raise ImportError(default(error_message, f'`{library}` must be installed'))

    if exists(min_version):
        import re

        version = getattr(module, '__version__', '0')
        version_ints = tuple(int(num) for num in re.findall(r'\d+', version))[:3]

        if version_ints < min_version:
            raise ImportError(default(error_message, f'`{library}` must be at least version {".".join(map(str, min_version))}'))
