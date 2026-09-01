from gguf.constants import MODEL_TENSOR


def test_model_tensor_values_are_unique() -> None:
    members = MODEL_TENSOR.__members__
    by_value: dict[int, list[str]] = {}
    for name, member in members.items():
        by_value.setdefault(member.value, []).append(name)

    aliases = {value: names for value, names in by_value.items() if len(names) > 1}
    assert not aliases
