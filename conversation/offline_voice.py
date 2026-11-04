"""Local speech fallback for workstations without access to the gated Phonon model."""
import io
import json
from pathlib import Path
import wave

import numpy as np


class OfflineVoice:
    def __init__(self,language):
        if language not in ('en','de'):
            raise ValueError('Supported voice languages: en, de')
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import PiperConfig
        cfg = json.loads((Path(__file__).resolve().parents[1]/'config/offline_voices.json').read_text())
        download = lambda name: hf_hub_download(cfg['repository'],name,revision=cfg['revision'])
        path = download(cfg['voices'][language])
        metadata = json.loads(Path(download(cfg['voices'][language]+'.json')).read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # Keep speech inference from starving robot control.
        self.model = PiperVoice(config=PiperConfig.from_dict(metadata),
            session=ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider']))
        self.sample_rate = self.model.config.sample_rate

    def synth(self,text):
        if not text.strip():
            return np.empty(0,dtype=np.float32)
        output = io.BytesIO()
        with wave.open(output,'wb') as wav:
            self.model.synthesize_wav(text.replace('\x00',' '),wav)
        with wave.open(io.BytesIO(output.getvalue()),'rb') as wav:
            if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getframerate()!=self.sample_rate:
                raise ValueError('Unexpected offline speech audio format')
            return np.frombuffer(wav.readframes(self.sample_rate*120),dtype='<i2').astype(np.float32)/32768
