
import os
import time
import torch
import torchaudio

from mmaudio.eval_utils import ModelConfig, model_cfg, generate, load_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tts_sr = 24000  # Sample rate for TTS

class MMAud:
    def __init__(self, model_dir='models', model_name='large_44k_v2', seed=696, cfg_strength=4.5, num_steps=25):
        """Initialize MMAudio sound generation model
        Args:
            model_dir: Directory containing model files
            model_name: Model variant (small_16k, small_44k, medium_44k, large_44k, large_44k_v2)
            seed: Random seed for generation
            cfg_strength: Classifier-free guidance strength
            num_steps: Number of diffusion steps
        """
        if not os.path.isdir(model_dir):
            print('!! Not found models', model_dir)
            self.ok = False
            return
        try:
            self.model_dir = model_dir
            self.cfg_strength = cfg_strength
            self.num_steps = num_steps
            
            model_cfg_obj = model_cfg(model_name, model_dir)
            self.seq_cfg = model_cfg_obj.seq_cfg
            
            dtype = torch.bfloat16
            
            self.net = get_my_mmaudio(model_cfg_obj.model_name).to(device, dtype).eval()
            self.net.load_weights(torch.load(model_cfg_obj.model_path, map_location=device, weights_only=True))
            
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(seed)
            self.fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)
            
            self.feature_utils = FeaturesUtils(
                tod_vae_ckpt=model_cfg_obj.vae_path,
                synchformer_ckpt=model_cfg_obj.synch_path,
                enable_conditions=True,
                mode=model_cfg_obj.mode,
                bigvgan_vocoder_ckpt=model_cfg_obj.bigvgan_16k_path,
                need_vae_encoder=False
            ).to(device, dtype).eval()
            self.ok = True
        except:
            self.ok = False
        
    @torch.inference_mode()
    def gen(self, video_path, prompt='', unprompt='', volume=0.5):
        """Generate sound for video and mix it into the video file
        
        Args:
            video_path: Path to input video file (will be modified in-place)
            prompt: Text prompt describing desired sound
            unprompt: Negative prompt (sounds to avoid)
            volume: Volume of generated sound relative to existing audio (0.0-1.0)
            
        Returns:
            Path to the video file (same as input, now with mixed audio)
        """
        sound_path = os.path.splitext(video_path)[0] + '_sound_temp.wav'
        
        video_info = load_video(video_path)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        duration = video_info.duration_sec
        
        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(
            self.seq_cfg.latent_seq_len,
            self.seq_cfg.clip_seq_len,
            self.seq_cfg.sync_seq_len
        )
        
        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[unprompt],
            feature_utils=self.feature_utils,
            net=self.net,
            fm=self.fm,
            rng=self.rng,
            cfg_strength=self.cfg_strength
        )
        
        audio = audios.float().cpu()[0]
        torchaudio.save(sound_path, audio, self.seq_cfg.sampling_rate)
        video_path = self.mix_with_video(video_path, sound_path, volume)
        if os.path.exists(sound_path):
            os.remove(sound_path)
        return video_path

    def mix_with_video(self, video_path, sound_path, volume=0.5):
        """Mix generated sound with video's existing audio track
        
        Args:
            video_path: Path to video file (will be replaced with mixed version)
            sound_path: Path to generated sound file
            volume: Volume of sound relative to existing audio (0.0-1.0)
        """
        temp_video = os.path.splitext(video_path)[0] + '_temp.mp4'
        
        command = f'ffmpeg -y -v quiet -i "{video_path}" -i "{sound_path}" -filter_complex "[1:a]loudnorm=I=-15:TP=-1.5:LRA=11[norm];[norm]volume={volume}[s];[0:a][s]amix=inputs=2:duration=first:weights={1-volume} {volume}" -c:v copy -ar {tts_sr} -ac 1 "{temp_video}"'
        os.system(command)

        if os.path.exists(temp_video):
            try:
                os.replace(temp_video, video_path)
            except:
                print("!! cannot replace video !!")
                video_path = temp_video
        return video_path

    @torch.inference_mode()
    def gen_from_prompt(self, prompt='', duration=8.0, output_path='output.wav', unprompt=''):
        """Generate sound from text prompt only (no video)
        
        Args:
            prompt: Text prompt describing desired sound
            duration: Duration in seconds
            output_path: Path for output audio file
            unprompt: Negative prompt (sounds to avoid)
            
        Returns:
            Path to generated audio file
        """
        self.seq_cfg.duration = duration
        self.net.update_seq_lengths(
            self.seq_cfg.latent_seq_len,
            self.seq_cfg.clip_seq_len,
            self.seq_cfg.sync_seq_len
        )
        audios = generate(
            None,  # no clip frames
            None,  # no sync frames
            [prompt],
            negative_text=[unprompt],
            feature_utils=self.feature_utils,
            net=self.net,
            fm=self.fm,
            rng=self.rng,
            cfg_strength=self.cfg_strength
        )
        audio = audios.float().cpu()[0]
        torchaudio.save(output_path, audio, self.seq_cfg.sampling_rate)
        return output_path


def main():
    """Test the MMA class"""
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-t', '--prompt', default='', help='Sound prompt')
    parser.add_argument('-v', '--volume', type=float, default=0.5, help='Sound volume (0.0-1.0)')
    parser.add_argument('-m', '--model', default='large_44k_v2', help='Model name')
    parser.add_argument('-md', '--model_dir', default='models', help='Models directory')
    args = parser.parse_args()
    
    mma = MMAud(model_dir=args.model_dir, model_name=args.model)
    mma.gen(args.input, prompt=args.prompt, volume=args.volume)
    print(f'Sound mixed into: {args.input}')


if __name__ == '__main__':
    main()
