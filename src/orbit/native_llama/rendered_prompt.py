"""A rendered template with explicit, immutable user-input tokenization boundaries."""
from __future__ import annotations


class RenderedChatPrompt(str):
    __slots__ = ('parts',)

    def __new__(cls, text: str, parts: tuple[tuple[str, bool], ...]):
        if not isinstance(parts, tuple) or not parts or any(not isinstance(p, tuple) or len(p) != 2 for p in parts):
            raise RuntimeError('invalid native prompt parts')
        if any(not isinstance(t, str) or type(flag) is not bool for t, flag in parts):
            raise RuntimeError('invalid native prompt parts')
        if ''.join(t for t, _ in parts) != text:
            raise RuntimeError('native prompt parts do not match rendered prompt')
        value = super().__new__(cls, text)
        object.__setattr__(value, 'parts', parts)
        return value

    def __setattr__(self, name, value):
        raise AttributeError('rendered prompt is immutable')

    def __delattr__(self, name):
        raise AttributeError('rendered prompt is immutable')

    def __getitem__(self, index):
        text = super().__getitem__(index)
        if not isinstance(index, slice):
            return text
        start, stop, step = index.indices(len(self))
        if step != 1:
            raise ValueError('strided rendered prompt slicing is unsupported')
        offset = 0
        parts = []
        for part, flag in self.parts:
            left, right = max(start - offset, 0), min(stop - offset, len(part))
            if left < right:
                parts.append((part[left:right], flag))
            offset += len(part)
        return RenderedChatPrompt(text, tuple(parts) or (('', False),))

    def append_template(self, text: str):
        return RenderedChatPrompt(str(self) + text, self.parts + ((text, False),))


def prompt_from_render(render: dict) -> str:
    prompt = render.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError('Orbit chat bridge returned an empty prompt')
    if 'prompt_tokenization' not in render and 'prompt_parts' not in render:
        return prompt  # Legacy revision-bound bridge contract.
    parts = render.get('prompt_parts')
    if render.get('prompt_tokenization') != 'parts-v1' or not isinstance(parts, list):
        raise RuntimeError('unsupported native prompt tokenization contract')
    if any(not isinstance(p, dict) or set(p) != {'text', 'is_input'} for p in parts):
        raise RuntimeError('invalid native prompt part envelope')
    return RenderedChatPrompt(prompt, tuple((p['text'], p['is_input']) for p in parts))
