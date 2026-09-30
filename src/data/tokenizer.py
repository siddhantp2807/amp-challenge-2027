"""Per-residue tokenizer: 20 canonical amino acids + PAD/BOS/EOS."""

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


class Vocab:
    def __init__(self):
        self.tokens = SPECIAL_TOKENS + AMINO_ACIDS
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, sequence: str) -> list[int]:
        ids = [BOS_ID]
        for aa in sequence:
            ids.append(self.token_to_id[aa])
        ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        chars = []
        for i in ids:
            tok = self.id_to_token[int(i)]
            if tok == EOS and stop_at_eos:
                break
            if tok in (PAD, BOS, EOS):
                continue
            chars.append(tok)
        return "".join(chars)


VOCAB = Vocab()
