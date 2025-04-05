from typing import List, Dict

class TTSConfig:
    VALID_LANGUAGES = ['中文', '英文', '日文', '粤语', '韩文', '中英混合', '日英混合', '粤英混合', '韩英混合', '多语种混合', '多语种混合(粤语)']
    VALID_CUT_METHODS = ['不切', '凑四句一切', '凑50字一切', '按中文句号。切', '按英文句号.切', '按标点符号切']
    VALID_SAMPLE_STEPS = [4, 8, 16, 32]

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    CHUNK_SIZE = 300  # 字符数
    
    SOVITS_PATH = "SoVITS_weights_v3/kuro_e2_s212_l64.pth"
    GPT_PATH = "GPT_weights_v3/kuro-e20.ckpt"
    INPUT_TEXT_FILE = "input/KuroV4_GPT1.txt"
    TASK_NAME = "KuroV4_GPT1"