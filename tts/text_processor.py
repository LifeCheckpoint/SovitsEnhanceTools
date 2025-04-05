# tts/text_processor.py
import os
import random
from typing import Dict, List

class ReferenceManager:
    def __init__(self, ref_dict: Dict[str, str]):
        self.ref_list: List[Dict] = ref_dict
        print(f"参考音频数量: {len(self.ref_list)}")
        
    def get_random_ref(self) -> tuple:
        """返回 (文件路径, 提示文本, 语种，参考 ID)"""
        ref = random.choice(self.ref_list)
        return (
            ref["file_path"],
            ref["prompt_text"],
            ref["language"],
            ref["id"]
        )

class TextSplitter:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        
    def split_text(self, text: str) -> List[str]:
        """分段逻辑"""
        chunks = []
        buffer = ""
        
        # 按句子分割
        for sentence in self._split_sentences(text):
            if len(buffer) + len(sentence) > self.chunk_size:
                chunks.append(buffer.strip())
                buffer = sentence
            else:
                buffer += sentence
                
        if buffer:
            chunks.append(buffer.strip())
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """中英文分句逻辑"""
        separators = ['。', '！', '？', '\n', '. ', '! ', '? ']
        return self._multi_split(text, separators)
    
    @staticmethod
    def _multi_split(text: str, separators: List[str]) -> List[str]:
        """多分隔符分句"""
        sentences = []
        last_index = 0
        for i in range(len(text)):
            for sep in separators:
                sep_len = len(sep)
                if text[i:i+sep_len] == sep:
                    sentences.append(text[last_index:i+sep_len])
                    last_index = i + sep_len
                    break
        sentences.append(text[last_index:])
        return [s.strip() for s in sentences if s.strip()]