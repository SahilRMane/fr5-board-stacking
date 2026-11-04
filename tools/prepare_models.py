"""Populate the pinned conversation model cache before starting the workcell."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'conversation'))


def main():
    print('Downloading pinned Gemma weights; the first download is several GB.',flush=True)
    subprocess.run(['bash',str(ROOT/'conversation/llm.sh'),'--help'],check=True,stdout=subprocess.DEVNULL)
    from model_server import voice
    from speech_input import ParakeetRecognizer
    print('Preparing the English and German voices...',flush=True)
    for language in ('en','de'):
        samples = voice(language).synth('Ready.' if language=='en' else 'Bereit.')
        if len(samples)==0:
            raise RuntimeError('Speech synthesis returned no audio')
    print('Preparing Parakeet speech recognition on CUDA...',flush=True)
    recognizer = ParakeetRecognizer()
    print('Conversation models cached and loaded successfully. Start with pixi run up --sim.',flush=True)


if __name__=='__main__':
    main()
