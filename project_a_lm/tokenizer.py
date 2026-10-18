"""Character-level tokenizer: the simplest possible text <-> integer mapping."""
from pathlib import Path


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))  # vocabulary = every unique character
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> id
        self.itos = {i: ch for ch, i in self.stoi.items()}  # id -> string

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    text = (Path(__file__).parent / "data" / "tiny.txt").read_text()
    tok = CharTokenizer(text)
    sample = "the cat sat"

    ids = tok.encode(sample)
    print("vocab_size:", tok.vocab_size)
    print("vocab:", repr("".join(tok.stoi)))
    print("encode:", ids)
    print("decode:", repr(tok.decode(ids)))
    print("round-trip ok:", tok.decode(ids) == sample)

    import tiktoken  # GPT-2's real subword (BPE) tokenizer, for comparison

    bpe = tiktoken.get_encoding("gpt2")
    bpe_ids = bpe.encode(sample)
    print("\nGPT-2 BPE vocab_size:", bpe.n_vocab)
    print("GPT-2 encode:", bpe_ids, "->", [bpe.decode([i]) for i in bpe_ids])
    print(f"tokens for {sample!r}: char={len(ids)}  bpe={len(bpe_ids)}")
