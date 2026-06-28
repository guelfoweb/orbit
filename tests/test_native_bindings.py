from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from orbit.native_llama.bindings import LlamaLibrary


class _Symbol:
    pass


def _stub_library() -> SimpleNamespace:
    names = [
        "ggml_backend_load_all",
        "llama_backend_free",
        "llama_log_set",
        "llama_model_default_params",
        "llama_context_default_params",
        "llama_sampler_chain_default_params",
        "llama_model_load_from_file",
        "llama_model_free",
        "llama_init_from_model",
        "llama_free",
        "llama_get_memory",
        "llama_memory_clear",
        "llama_memory_seq_cp",
        "llama_memory_seq_keep",
        "llama_memory_seq_rm",
        "llama_state_get_size",
        "llama_state_get_data",
        "llama_state_set_data",
        "llama_state_seq_get_size",
        "llama_state_seq_get_data",
        "llama_state_seq_set_data",
        "llama_model_get_vocab",
        "llama_vocab_n_tokens",
        "llama_model_chat_template",
        "llama_chat_apply_template",
        "llama_tokenize",
        "llama_token_to_piece",
        "llama_vocab_is_eog",
        "llama_batch_get_one",
        "llama_batch_init",
        "llama_batch_free",
        "llama_decode",
        "llama_synchronize",
        "llama_time_us",
        "llama_get_logits_ith",
        "llama_sampler_chain_init",
        "llama_sampler_chain_add",
        "llama_sampler_init_greedy",
        "llama_sampler_sample",
        "llama_sampler_accept",
        "llama_sampler_reset",
        "llama_sampler_free",
    ]
    return SimpleNamespace(**{name: _Symbol() for name in names})


class NativeBindingsTests(unittest.TestCase):
    def test_configure_api_binds_prefix_anchor_primitives(self) -> None:
        library = object.__new__(LlamaLibrary)
        library.build_bin = Path(".")
        library._handles = []
        library.lib = _stub_library()

        LlamaLibrary._configure_api(library)

        self.assertEqual(len(library.lib.llama_memory_seq_cp.argtypes), 5)
        self.assertEqual(len(library.lib.llama_memory_seq_keep.argtypes), 2)
        self.assertEqual(len(library.lib.llama_state_get_size.argtypes), 1)
        self.assertEqual(len(library.lib.llama_state_get_data.argtypes), 3)
        self.assertEqual(len(library.lib.llama_state_set_data.argtypes), 3)
        self.assertEqual(len(library.lib.llama_state_seq_get_size.argtypes), 2)
        self.assertEqual(len(library.lib.llama_state_seq_get_data.argtypes), 4)
        self.assertEqual(len(library.lib.llama_state_seq_set_data.argtypes), 4)
        self.assertEqual(len(library.lib.llama_vocab_n_tokens.argtypes), 1)
        self.assertEqual(len(library.lib.llama_batch_init.argtypes), 3)
        self.assertEqual(len(library.lib.llama_batch_free.argtypes), 1)
        self.assertEqual(len(library.lib.llama_synchronize.argtypes), 1)
        self.assertEqual(len(library.lib.llama_get_logits_ith.argtypes), 2)


if __name__ == "__main__":
    unittest.main()
