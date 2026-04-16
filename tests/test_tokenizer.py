from mini_decoder_only.dataset import CharTokenizer, TextDataset
import logging

def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer("hello world")
    encoded = tokenizer.encode("hello")
    assert tokenizer.decode(encoded) == "hello"


def test_text_dataset_batch_alignment() -> None:
    tokenizer = CharTokenizer("hello world")
    token_ids = tokenizer.encode("hello worldhello world")
    dataset = TextDataset(token_ids, block_size=4, batch_size=2)

    x, y = dataset.get_batch("train")

    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert (x[:, 1:] == y[:, :-1]).all()
