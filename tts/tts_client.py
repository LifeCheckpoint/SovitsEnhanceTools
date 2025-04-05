# tts/tts_client.py
import time, os
from gradio_client import Client, file
from typing import Optional, List, Dict, Any
from .config import TTSConfig
from .utils import ensure_dir, generate_filename, validate_audio_file

class SovitsTTS:
    def __init__(self, api_url: str = "http://localhost:9872/"):
        self.client = Client(api_url)
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        ensure_dir("output")

    def _validate_params(self, params: Dict[str, Any]):
        """参数验证"""
        # 音频文件验证
        validate_audio_file(params['ref_wav_path'])
    
        # 文本验证
        if not params['text'].strip():
            raise ValueError("合成文本不能为空")

        # 枚举值验证
        if params['prompt_language'] not in TTSConfig.VALID_LANGUAGES:
            raise ValueError(f"无效 prompt 语言: {params['prompt_language']}")
        if params['text_language'] not in TTSConfig.VALID_LANGUAGES:
            raise ValueError(f"无效 text 语言: {params['text_language']}")
        if params['how_to_cut'] not in TTSConfig.VALID_CUT_METHODS:
            raise ValueError(f"无效切分方式: {params['how_to_cut']}")
        if params['sample_steps'] not in TTSConfig.VALID_SAMPLE_STEPS:
            raise ValueError(f"无效采样步数: {params['sample_steps']}")

    def synthesize_with_retry(self, max_retries: int = TTSConfig.MAX_RETRIES, ref_id: Optional[str] = "", **kwargs) -> str:
        """带重试合成"""
        for attempt in range(max_retries):
            try:
                return self.synthesize(**kwargs, ref_id=ref_id)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(TTSConfig.RETRY_DELAY)
        raise RuntimeError("超出最大重试次数")
    
    def _save_audio(self, audio_data: bytes, name_prefix: str = "") -> str:
        """保存音频文件到输出目录"""
        filename = f"{name_prefix}_tts_{int(time.time()*1000)}.wav"
        save_path = os.path.join("output", filename)
        
        try:
            with open(save_path, "wb") as f:
                f.write(audio_data)
            return os.path.abspath(save_path)
        except IOError as e:
            raise RuntimeError(f"文件保存失败: {e}")

    def set_model(
        self,
        sovits_path: Optional[str] = None,
        gpt_path: Optional[str] = None,
        prompt_lang: str = "中文",
        text_lang: str = "中文"
    ):
        """
        设置模型参数
        :param sovits_path: SoVITS模型路径（可选）
        :param gpt_path: GPT模型路径（可选）
        """
        try:
            if sovits_path:
                self.client.predict(
                    sovits_path=sovits_path,
                    prompt_language=prompt_lang,
                    text_language=text_lang,
                    api_name="/change_sovits_weights"
                )
            if gpt_path:
                self.client.predict(
                    gpt_path=gpt_path,
                    api_name="/change_gpt_weights"
                )
        except Exception as e:
            raise RuntimeError(f"模型设置失败: {e}")

    def synthesize(
        self,
        ref_wav_path: str,
        text: str,
        prompt_text: str = "",
        prompt_language: str = "中文",
        text_language: str = "中文",
        how_to_cut: str = "凑四句一切",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed: float = 1.0,
        sample_steps: int = 32,
        pause_second: float = 0.3,
        ref_free: bool = False,
        if_freeze: bool = False,
        if_sr: bool = True,
        inp_refs: Optional[List[str]] = None,
        ref_id: Optional[str] = "",
    ) -> str:
        """
        语音合成主函数
        
        :return: 保存的本地文件绝对路径
        """
        params = locals()
        params.pop('self')
        reference_id = params.pop('ref_id')
        
        try:
            # 处理文件路径参数
            params['ref_wav_path'] = file(params['ref_wav_path'])
            if params['inp_refs']:
                params['inp_refs'] = [file(path) for path in params['inp_refs']]

            # 执行API调用
            result = self.client.predict(
                **params,
                api_name="/get_tts_wav"
            )
            
            # 获取文件内容
            if isinstance(result, (list, tuple)):
                file_path = result[0]  # 取第一个返回值
            else:
                file_path = result
            
            # 读取文件内容
            with open(file_path, "rb") as f:
                audio_data = f.read()
            
            # 保存结果
            return self._save_audio(audio_data, name_prefix=TTSConfig.TASK_NAME+(f"_refID_{reference_id}" if reference_id else ""))
    
        except FileNotFoundError:
            raise RuntimeError(f"服务端生成的文件不存在: {file_path}")
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise RuntimeError(f"合成失败: {e}")