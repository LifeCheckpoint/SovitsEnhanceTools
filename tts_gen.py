# main.py
import json
from typing import Dict, List
from tts.tts_client import SovitsTTS
from tts.text_processor import ReferenceManager, TextSplitter
from config import TTSConfig
from tqdm import tqdm

def load_references(ref_json: str) -> Dict[str, str]:
    with open(ref_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # 初始化组件
    tts = SovitsTTS()
    ref_manager = ReferenceManager(load_references("references.json"))
    splitter = TextSplitter(TTSConfig.CHUNK_SIZE)
    
    # 读取输入文本
    with open(TTSConfig.INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # 处理合成任务
    recent_prompt_lang = None

    chunks = splitter.split_text(full_text)
    progress_bar = tqdm(total=len(chunks), desc="合成进度", unit="段")
    for i, chunk in enumerate(chunks):
        ref_path, prompt_text, prompt_lang, ref_id = ref_manager.get_random_ref()

        try:
            # 如果语言更新，重新设置模型
            if prompt_lang != recent_prompt_lang: tts.set_model(
                sovits_path=TTSConfig.SOVITS_PATH,
                gpt_path=TTSConfig.GPT_PATH,
                prompt_lang=prompt_lang
            )
        except Exception as e:
            print(f"载入模型失败: {str(e)}")

        recent_prompt_lang = prompt_lang

        try:
            output_path = tts.synthesize_with_retry(
                ref_wav_path=ref_path,
                text=chunk,
                prompt_text=prompt_text,
                text_language="中文",
                how_to_cut="凑50字一切",
                ref_id=ref_id
            )
        except Exception as e:
            print(f"第 {i+1} 段合成失败: {str(e)}")

        progress_bar.update()

if __name__ == "__main__":
    main()