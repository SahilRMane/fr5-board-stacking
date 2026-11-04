"""Run with pixi run -e models python tests/voice_fallback_check.py; no robot."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conversation'))
import model_server
from offline_voice import OfflineVoice


class VoiceFallbackCheck(unittest.TestCase):
    def setUp(self):
        model_server.tts.clear();model_server.tts_backends.clear()

    def test_gated_voice_falls_back_explicitly(self):
        with patch.dict(os.environ,WORKCELL_TTS='auto'), \
                patch.object(model_server.ptts,'TTS',side_effect=OSError('gated')), \
                patch('offline_voice.OfflineVoice') as fallback:
            self.assertIs(model_server.voice('en'),fallback.return_value)
            self.assertEqual(model_server.tts_backends,{'en':'piper'})
            self.assertIs(model_server.voice('en'),fallback.return_value)
            fallback.assert_called_once_with('en')

    def test_strict_phonon_and_invalid_selections_report_errors(self):
        with patch.dict(os.environ,WORKCELL_TTS='phonon'), \
                patch.object(model_server.ptts,'TTS',side_effect=OSError('gated')):
            with self.assertRaises(OSError):model_server.voice('en')
        with patch.dict(os.environ,WORKCELL_TTS='unknown'):
            with self.assertRaises(ValueError):model_server.voice('en')
        with self.assertRaises(ValueError):OfflineVoice('unknown')

    def test_public_voices_generate_finite_audio_without_credentials(self):
        for language in ('en','de'):
            with self.subTest(language=language):
                voice = OfflineVoice(language)
                self.assertEqual(len(voice.synth('')),0)
                audio = voice.synth('Ready. Variant A.' if language=='en' else 'Bereit. Variante A.')
                self.assertGreater(len(audio),1000)
                self.assertEqual(audio.dtype,np.float32)
                self.assertTrue(np.isfinite(audio).all())
                self.assertLessEqual(float(np.abs(audio).max()),1.)


if __name__=='__main__':unittest.main()
