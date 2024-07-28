from torchtune.modules.tokenizers._tiktoken import TikTokenBaseTokenizer
from typing import Dict, Iterator, List

from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe

sound_tokens = [f'<|sound_{num:04d}|>' for num in range(1024)]

class CustomTikTokenTokenizer(TikTokenBaseTokenizer):
    def __init__(
        self,
        path: str,
        name: str,
        pattern: str,
        bos_id: int,
        eos_id: int,
        special_tokens: Dict[str, int],
    ):
        mergeable_ranks = load_tiktoken_bpe(path)
        old_vocab_size = len(mergeable_ranks) + len(special_tokens)
        self.old_vocab_size = old_vocab_size
        print(f"old_vocab_size: {old_vocab_size}")
        
        SOUND_TOKENS = {
            f"{sound_tokens[i]}".encode("utf8"): old_vocab_size + i
            for i in range(len(sound_tokens))
        }
       
        mergeable_ranks = {**mergeable_ranks, **SOUND_TOKENS}
            
        self.tt_model = Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        self.base_vocab_size = len(mergeable_ranks)
        self.vocab_size = self.tt_model.n_vocab
        self.bos_id = bos_id
        self.eos_id = eos_id
    #encode normal new sound tokens    
    def single_encode(self, token: str) -> int:
        return self.tt_model.encode_single_token(token)
    #encode 
    def encode_sound_tokens(self, token: str) -> List[int]:
        sound_tokens = token.strip().split('|><|')
        sound_tokens = sound_tokens[1:-1]
        encoded_tokens = []
        for sound_token in sound_tokens:
            encoded_token = self.single_encode(f"<|{sound_token}|>")
            encoded_tokens.append(encoded_token)
    
        return encoded_tokens