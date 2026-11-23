"""Explicit process-bound acceleration selection; never silently fall back."""
from ctypes import c_size_t, c_void_p, c_int, c_int32, c_uint64, c_char_p

ACCELERATORS = ('cpu', 'intel-gpu')


def selected_family(base: str, accelerator: str) -> str:
    if accelerator not in ACCELERATORS:
        raise ValueError('accelerator must be cpu or intel-gpu')
    if accelerator == 'cpu':
        return base
    if base != 'mainline-support':
        raise ValueError('Intel GPU currently supports the mainline Inkling text backend only')
    return 'mainline-vulkan'


def select_intel_device(lib):
    signatures = {'ggml_backend_dev_count': ([], c_size_t),
                  'ggml_backend_dev_get': ([c_size_t], c_void_p),
                  'ggml_backend_dev_type': ([c_void_p], c_int),
                  'ggml_backend_dev_name': ([c_void_p], c_char_p),
                  'ggml_backend_dev_description': ([c_void_p], c_char_p),
                  'orbit_llama_model_gpu_layers': ([c_void_p], c_int32),
                  'orbit_llama_model_gpu_bytes': ([c_void_p], c_uint64)}
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    matches = []
    for index in range(lib.ggml_backend_dev_count()):
        dev = lib.ggml_backend_dev_get(index)
        name = (lib.ggml_backend_dev_name(dev) or b'').decode('utf-8', errors='replace')
        description = (lib.ggml_backend_dev_description(dev) or b'').decode('utf-8', errors='replace')
        # Enum 2 is IGPU in the pinned mainline ABI. Software renderers are CPU.
        if lib.ggml_backend_dev_type(dev) == 2 and name.startswith('Vulkan') and 'intel' in description.casefold():
            matches.append((dev, name, description))
    if len(matches) != 1:
        raise RuntimeError('Intel GPU requires exactly one available integrated Intel Vulkan device')
    return matches[0]


def efficient_host_cpus(allowed, frequency_by_cpu):
    """Prefer two non-boost cores when frequency groups expose a hybrid CPU."""
    pairs = {cpu: hz for cpu, hz in frequency_by_cpu.items() if cpu in allowed and hz > 0}
    if not pairs:
        return None
    ceiling = max(pairs.values()) * .85
    groups = {}
    for cpu, hz in pairs.items():
        if hz <= ceiling:
            groups.setdefault(hz, []).append(cpu)
    candidates = [(hz, sorted(cpus)) for hz, cpus in groups.items() if len(cpus) >= 2]
    return set(max(candidates)[1][:2]) if candidates else None


def configure_gpu_host():
    import os
    from pathlib import Path
    if not hasattr(os, 'sched_getaffinity'):
        return None
    allowed = os.sched_getaffinity(0)
    frequencies = {}
    for cpu in allowed:
        try:
            frequencies[cpu] = int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq').read_text())
        except (OSError, ValueError):
            continue
    selected = efficient_host_cpus(allowed, frequencies)
    if selected:
        os.sched_setaffinity(0, selected)
    return sorted(selected) if selected else None


def loaded_intel_status(lib, model, device_name, description):
    count = int(lib.orbit_llama_model_gpu_layers(model))
    if count <= 0:
        raise RuntimeError('Intel GPU did not receive model layers; CPU fallback refused')
    weight_bytes = int(lib.orbit_llama_model_gpu_bytes(model))
    if weight_bytes <= 0:
        raise RuntimeError('Intel GPU has no GPU weight buffers; CPU fallback refused')
    import os
    return {'gpu_weight_bytes': weight_bytes, 'host_cpus': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'requested': 'intel-gpu', 'active': 'intel-gpu', 'device': description,
            'backend_device': device_name, 'offloaded_layers': count, 'experimental': True}
