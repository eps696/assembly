
import os
import sys
import importlib.util

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore')

def make_mock(name, version="2.5.8"):
    def mock_function(*args, **kwargs):
        raise ImportError(f"Flash Attention not available: {name}")
    spec = importlib.util.spec_from_loader(name, loader=None, origin="mock")
    module = importlib.util.module_from_spec(spec)
    module.__name__ = name
    module.__package__ = name.split('.')[0] if '.' in name else name
    module.__version__ = version
    module.__file__ = f"<mock {name}>"
    module.__spec__ = spec
    module.flash_attn_func = mock_function
    module.flash_attn_varlen_func = mock_function
    module.index_first_axis = mock_function
    module.pad_input = mock_function
    module.unpad_input = mock_function
    return module

def disable_flash_attention():
    modules_to_remove = [name for name in sys.modules.keys() if 'flash_attn' in name]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    mock_modules = {
        'flash_attn': make_mock('flash_attn'),
        'flash_attn.flash_attn_interface': make_mock('flash_attn.flash_attn_interface'),
        'flash_attn.bert_padding': make_mock('flash_attn.bert_padding'),
        'flash_attn_cuda': make_mock('flash_attn_cuda'),
    }
    for name, module in mock_modules.items():
        sys.modules[name] = module

disable_flash_attention()

import torch
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS

from util import basename

device = "cuda" if torch.cuda.is_available() else "cpu"

class ChatterBox:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        try:
            self.model = ChatterboxTTS.from_local(model_dir, device=device)
        except:
            print(f'ChatterboxTTS models not found in {model_dir}, getting from HuggingFace...')
            self.model = ChatterboxTTS.from_pretrained(device=device)

    def gen(self, text, filename=None, **kwargs):
        wav = self.model.generate(text, **kwargs)
        if filename is not None:
            ta.save(filename, wav, self.model.sr)
        return wav

    def tts_to_file(self, text, speaker, file_path):
        self.get_voice(os.path.join(self.model_dir, 'vctk-%s.pt' % speaker))
        self.gen(text, file_path, exaggeration=.69)

    def get_voice(self, filename=None):
        voice_emb = torch.load(filename, map_location='cpu', weights_only=False)
        self.model.conds = voice_emb.to(device)

    def set_voice(self, wav_file, filename=None):
        self.model.prepare_conditionals(wav_file)
        voice_emb = self.model.conds
        if filename is not None:
            torch.save(voice_emb, filename)
        return voice_emb

def main():
    tts = ChatterBox()
    text = "Jack stopped and looked into the sky. He was not ready to what he saw there."
    tts.get_voice('../models/chatter/vctk-f-227.pt')
    tts.gen(text, "test.wav") # audio_prompt_path=in.wav, exaggeration=0.5, cfg_weight=0.5
    

if __name__ == '__main__':
    main()
