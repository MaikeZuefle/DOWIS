TASK_LANGUAGES = {
    'ASR': ['en', 'de', 'it', 'es', 'fr', 'pt', 'nl', 'ru', 'sv', 'cs', 'hu'],  # 11 languages
    'ST': ['de', 'it', 'es', 'fr', 'pt', 'nl', 'ru', 'sv', 'cs', 'hu'],  # 10 languages
    'SQA': ['en'],  # 1 language
    'SSUM': ['en', 'de', 'it'],  # 3 languages
    'TTS': ['en'],  # 1 language
    'S2ST': ['de', 'it', 'es', 'fr', 'pt', 'nl', 'ru', 'sv', 'cs', 'hu'],  # 10 languages
    'MT': ['de', 'it', 'es', 'fr', 'pt', 'nl', 'ru', 'sv', 'cs', 'hu'],  # 10 languages
    'TSUM': ['en', 'de', 'it'],  # 3 languages
    'ACHAP': ['en']  # 1 language
}

MODEL_TASK_LANGUAGES = {
    'phi_multimodal': {
        'ASR': ['en', 'de', 'fr', 'it', 'es', 'pt'],
        'TTS': [],
        'S2ST': []
    },
    'qwen_omni': {
        'ASR': ['en', 'de', 'fr', 'it', 'es', 'pt'],
    }
}


TASK_METRICS = {
    'ASR': 'wer',
    'MT': 'CometQE',
    'S2ST': ['UTMOS', 'ASR-COMET'],  # Two metrics
    'SSUM': 'BERTScore_F1',
    'TSUM': 'BERTScore_F1',
    'ST': 'CometQE',
    'TTS': ['UTMOS', 'ASR-WER'],  # Two metrics
    'ACHAP': ['CollarF1', 'GC-BS'],
    'SQA': 'BERTScore_F1'
}


MODEL_DISPLAY_NAMES = {
    'qwen_omni': 'Qwen2.5-Omni',
    'phi_multimodal': 'Phi-4-MM'
}






# Metrics where lower is better (need to flip sign for heatmap)
LOWER_IS_BETTER = ['wer']

METRIC_DISPLAY_NAMES = {
    'BERTScore_F1': 'BERTSc.',
    'wer': 'WER',
    'CometQE': 'CometQE',
    'UTMOS': 'UTMOS',
    'ASR-COMET': 'ASR-COMET',
    'ASR-WER': 'ASR-WER',
}