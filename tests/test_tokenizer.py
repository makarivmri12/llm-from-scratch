from src.tokenizer import Tokenizer


def test_tokenizer_encode_decode():
    tokenizer = Tokenizer()
    tokenizer.fit_texts(["hello world"])
    tokens = tokenizer.encode("hello")
    assert isinstance(tokens, list)
    text = tokenizer.decode(tokens)
    assert text == "hello"
    assert tokenizer.vocab_size > 0
