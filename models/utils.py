def get_padding(kernel_size):
    """Return `same` padding for a given kernel size."""
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return tuple(s // 2 for s in kernel_size)
