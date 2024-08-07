from typing import Dict, List, Optional, Tuple
from typing_extensions import override
from torchtune.data import Message, truncate
from torchtune.modules.tokenizers import ModelTokenizer, CustomTikTokenTokenizer
from torchtune.models.llama3._tokenizer import (
    CL100K_PATTERN,
    Llama3Tokenizer,
    LLAMA3_SPECIAL_TOKENS,
    RESERVED_TOKENS,

)
special_sound_tokens = ["<|sound_start|>", "<|sound_end|>"]
SPECIAL_SOUND_TOKENS = {
    f"{special_sound_tokens[num]}": 128013 + len(RESERVED_TOKENS) + num
    for num in range(len(special_sound_tokens))
}

LLAMA3_S_SPECIAL_TOKENS = {**LLAMA3_SPECIAL_TOKENS, **SPECIAL_SOUND_TOKENS}

class Llama3STokenizer(Llama3Tokenizer):
    @override
    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        super().__init__(path, special_tokens or LLAMA3_S_SPECIAL_TOKENS)

        # Sound tokens for interleaved sound-text
        self.sound_start_id = self.special_tokens["<|sound_start|>"]
        self.sound_end_id = self.special_tokens["<|sound_end|>"]

        self.tt_model = CustomTikTokenTokenizer(
            path=path,
            name="custom_llama3_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=self.special_tokens,
        )

    @property
    def old_vocab_size(self) -> int:
        return self.tt_model.old_vocab_size

    def single_encode(self, token: str) -> int:
        return self.tt_model.single_encode(token)

    def encode_sound_tokens(self, sound_text: str, add_start: bool, add_end: bool, add_bos: bool, add_eos: bool) -> List[int]:
        token_ids = []
        if add_start:
            token_ids += [self.sound_start_id]
        token_ids += self.tt_model.encode_sound_tokens(sound_text)
        if add_end:
            token_ids += [self.sound_end_id]
        if add_bos:
            token_ids = [self.tt_model.bos_id] + token_ids
        if add_eos:
            token_ids += [self.tt_model.eos_id]
        
        return token_ids
    @override
    def _tokenize_body(self, message: Message) -> List[int]:
        tokenized_body = []
        for item in message.content:
            if item["type"] == "text":
                if "<|sound_start|>" in item["content"] and "<|sound_end|>" in item["content"]:
                    tokenized_body += [self.sound_start_id]
                    tokenized_body += self.tt_model.encode_sound_tokens(item["content"], add_start=False, add_end=False)
                    tokenized_body += [self.sound_end_id]
                else:
                    tokenized_body += self.encode(
                        item["content"].strip(), add_bos=False, add_eos=False
                    )
            elif item["type"] == "image":
                tokenized_body += [self.image_id]
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")

        if message.ipython:
            tokenized_body = [self.python_tag] + tokenized_body

        return tokenized_body

    def encode_sound_text(self, text: str) -> List[int]:
        tokens = [self.bos_id]

        user_messages = [self.start_header_id] + self.encode("user".strip(), add_bos=False, add_eos=False) + [self.end_header_id] + self.encode("\n\n", add_bos=False, add_eos=False)
        tokens += user_messages

        chunk = text.split("<|sound_start|>")[1].strip()
        sound_text = chunk.split("<|sound_end|>")[0].strip()
        sound_tokens_id = self.tt_model.encode_sound_tokens(sound_text)
        tokens += sound_tokens_id

        tokens += [self.eot_id]

        assistant_messages = [self.start_header_id] + self.encode("assistant".strip(), add_bos=False, add_eos=False) + [self.end_header_id] + self.encode("\n\n", add_bos=False, add_eos=False)
        tokens += assistant_messages

        text_inter = text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        text_tokens_id = self.encode(text_inter, add_bos=False, add_eos=False)
        tokens += text_tokens_id

        tokens += [self.eot_id]

        return tokens