from torchtune.modules.tokenizers._tiktoken import TikTokenBaseTokenizer
from typing import Dict, Iterator, List

from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe

sound_tokens = [f'<|sound_{num:04d}|>' for num in range(513)]
duration_tokens = [f'<|duration_{num:02d}|>' for num in range(2,100)]
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
        # print(mergeable_ranks.type)
        old_vocab_size = len(mergeable_ranks) + len(special_tokens)
        self.old_vocab_size = old_vocab_size
        # print(f"old_vocab_size: {old_vocab_size}")
        # add sound tokens to the vocab. 
        SOUND_TOKENS = {
            f"{sound_tokens[i]}".encode("utf8"): old_vocab_size + i
            for i in range(len(sound_tokens))
        }
        DURATION_TOKENS = {
            f"{duration_tokens[i]}".encode("utf8"): old_vocab_size + len(sound_tokens) + i
            for i in range(len(duration_tokens))
        }
       
        mergeable_ranks = {**mergeable_ranks, **SOUND_TOKENS, **DURATION_TOKENS}
            
        self.tt_model = Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        # Vocab size without special tokens
        self.base_vocab_size = len(mergeable_ranks)
        # Vocab size with special tokens
        
        self.vocab_size = self.tt_model.n_vocab
        self.bos_id = bos_id
        self.eos_id = eos_id

    # Encode normal new sound tokens    
    def single_encode(self, token: str) -> int:
        return self.tt_model.encode_single_token(token)
    # Covert "<|sound_0000|><|sound_0001|>" -> List[ids]
    def encode_sound_tokens(self, token: str) -> List[int]:
        """
        Convert sound tokens to List[ids]
        Example: Covert "<|duration_01|><|sound_01|>" -> List[ids]
        
        Args:
            token (str): String containing sound and duration tokens
            
        Returns:
            List[int]: List of encoded token ids
        """
        sound_tokens = token.strip().split('|><|')
        sound_tokens = sound_tokens[1:-1]
        encoded_tokens = []
        for token in sound_tokens:
            if "duration" in token:
                encoded_token = self.single_encode(f"<|{token}|>")
                encoded_tokens.append(encoded_token)
            elif "sound" in token:
                encoded_token = self.single_encode(f"<|{token}|>")
                encoded_tokens.append(encoded_token)
            else:
                raise ValueError(f"Invalid token: {token}")
    
        return encoded_tokens
