
import os
import base64
import asyncio
import concurrent.futures
import requests
import math
from PIL import Image

from runware import Runware, IImageInference, IVideoInference
from runware.types import IFrameImage

from eps import basename

# Keywords indicating content moderation rejection - don't retry these
MODERATION_KEYWORDS = [
    'invalid content', 'content detected', 'moderation', 'content policy',
    'violates', 'flagged', 'safety', 'inappropriate', 'not allowed',
    'prohibited', 'blocked', 'filtered', 'rejected'
]

class Censored(Exception):
    """Raised when content is rejected by the provider's moderation system"""
    pass

def is_moderation_error(error_msg):
    """Check if an error message indicates content moderation rejection"""
    return any(kw in str(error_msg).lower() for kw in MODERATION_KEYWORDS)

SIZES = {
    'flux': {(1280,720), (1344,768)},
    'imagen': {(1280,896), (1408,768)},
    'nano': {(1024,1024),(2048,2048),(4096,4096),(1264,848),(2528,1696),(5096,3392),(5056,3392),(848,1264),(1696,2528),(3392,5096),(3392,5056),
             (1200,896),(2400,1792),(4800,3584),(896,1200),(1792,2400),(3584,4800),(928,1152),(1856,2304),(3712,4608),(1152,928),(2304,1856),
             (4608,3712),(768,1376),(1536,2752),(3072,5504),(1376,768),(2752,1536),(5504,3072),(1548,672),(1584,672),(3168,1344),(6336,2688)},
    'gpt': {(1024,1024), (1536,1024), (1024,1536)},
    'mj': {(1456,816), (816,1456), (1024,1024), (1232,928), (928,1232), (1344,896), (896,1344), (1680,720)},
    'bytedance': {(864,480), (736,544), (640,640), (544,736), (480,864), (960,416), (1248,704), (1120,832), (960,960), (832,1120), 
                  (704,1248), (1504,640), (1920,1088), (1664,1248), (1440,1440), (1248,1664), (1088,1920), (2176,928)},
    'seedpro': {(864,496),(752,560),(640,640),(560,752),(496,864),(992,432),(1280,720),(1112,834),(960,960),(834,1112),(720,1280),(1470,630)},
    'klingai': {(1920,1080), (1080,1920), (1080,1080)},
    'pruna': {(1280,720),(720,1280),(960,720),(720,960),(1080,720),(720,1080),(720,720),(1440,1080),(1080,1440),(1620,1080),(1080,1620),(1080,1080)},
    'veo': {(1920,1080), (1280,720), (1080,1920)}
}

IMAGE_CONFIG = {
    'flux': {'id': 'bfl:5@1',
        'desc': 'FLUX.2 [pro], 3-5¢, good',
        'refs': 8,
        'size': (1280, 720), 'sizes': SIZES['flux']
    },
    'ima': {'id': 'google:2@2',
        'desc': 'Imagen 4.0 Ultra, 6¢, good',
        'size': (1280, 896), 'sizes': SIZES['imagen']
    },
    'nano': {'id': 'google:4@2',
        'desc': 'Nano Banana 2 Pro, 14¢, best',
        'refs': 14,
        'size': (1376, 768), 'sizes': SIZES['nano']
    },
    'gpt': {'id': 'openai:4@1',
        'desc': 'GPT Image 1.5, 20¢, best',
        'size': (1536,1024), 'sizes': SIZES['gpt']
    },
    'hidream': {'id': 'runware:97@1',
        'desc': 'HiDream-I1-Full, 1¢, ok real',
        'size': (1344, 768), 'sizes': []
    },

    'seedream': {'id': 'bytedance:seedream@4.5',
        'desc': 'Seedream 4.5, 4¢',
        'size': (2560, 1440), 'sizes': []
    },
    'runway': {'id': 'runway:4@1',
        'desc': 'Runway Gen-4 Image, 8¢',
        'size': (1280, 720), 'sizes': []
    },
    'mj7': {'id': 'midjourney:3@1',
        'desc': 'Midjourney V7, 3¢ 4pcs',
        'num': 4,
        'size': (1456, 816), 'sizes': SIZES['mj'] # (1248, 704)
    },
    'mj6': {'id': 'midjourney:2@1',
        'desc': 'Midjourney V6.1, 3¢ 4pcs',
        'num': 4,
        'size': (1456, 816), 'sizes': SIZES['mj']
    },
    'imafast': {'id': 'google:2@3',
        'desc': 'Imagen 4.0 Fast, 2¢',
        'size': (1280, 896), 'sizes': SIZES['imagen']
    },
    'hidrefast': {'id': 'runware:97@3',
        'desc': 'HiDream-I1-Fast, 0.4¢, BAD',
        'size': (1344, 768), 'sizes': []
    }
}

VIDEO_CONFIG = {
    'pruna': {'id': 'prunaai:p-video@0',
        'desc': 'P-Video, 5¢ 10sec 720p-draft, 20¢ 10sec 720p, 10¢ 10sec 1080p-draft, 40¢ 10sec 1080p',
        'duration': [2,10], 'fps': 24, 'size': (1280, 720), 'sizes': SIZES['pruna']
    },
    'seedprof': {'id': 'bytedance:2@2',
        'desc': 'Seedance 1.0 Pro Fast, 8¢ 10sec 864x480, 16¢ 12sec 1248x704',
        'duration': [2,12], 'fps': 24, 'size': (1248, 704), 'sizes': SIZES['bytedance']
    },
    'hail': {'id': 'minimax:4@1', # good
        'desc': 'MiniMax Hailuo 2.3, 28¢ i2v 5sec',
        'duration': [3,6], 'fps': 25, 'size': (1366, 768), 'sizes': {(1366,768)}
    },
    'kling25': {'id': 'klingai:6@1', # good slow
        'desc': 'KlingAI 2.5 Turbo PRO, 70¢ 10sec',
        'duration': [3,10], 'fps': 24, 'size': (1920, 1080), 'sizes': SIZES['klingai']
    },
    'veofast': {'id': 'google:3@3', # best $$
        'desc': 'Veo 3.1 Fast, $1.2 8sec',
        'duration': [2,8], 'fps': 25, 'size': (1280, 720), 'sizes': SIZES['veo']
    },
    'seedpro': {'id': 'bytedance:seedance@1.5-pro', # crazy!!
        'desc': 'Seedance 1.5 Pro, 60¢ 12sec',
        'duration': [2,12], 'fps': 24, 'size': (1280, 720), 'sizes': SIZES['seedpro']
    },

    'kling26': {'id': 'klingai:kling-video@2.6-pro', # good text2vid ONLY
        'desc': 'KlingAI 2.6 Pro, 70¢ 10sec',
        'duration': [3,10], 'fps': 24, 'size': (1920, 1080), 'sizes': SIZES['klingai']
    },
    'runway': {'id': 'runway:1@1',
        'desc': 'Runway Gen-4 Turbo, 50¢ 10sec, missing parameter inputs.frameImages',
        'duration': [2,10], 'fps': 25, 'size': (1280, 720), 'sizes': {(1280,720)}
    },
    'veo': {'id': 'google:3@2', # best $$$
        'desc': 'Veo 3.1, $3.2 8sec',
        'duration': [2,8], 'fps': 25, 'size': (1280, 720), 'sizes': SIZES['veo']
    },
    'sora': {'id': 'openai:3@1',
        'desc': 'Sora 2, 10¢/sec',
        'duration': [4,12], 'fps': 30, 'size': (1280, 720), 'sizes': {(1280,720)}
    },
    'hailfast': {'id': 'minimax:4@2',
        'desc': 'MiniMax Hailuo 2.3 Fast, 20¢, img2vid only',
        'duration': [3,6], 'fps': 25, 'size': (1366, 768), 'sizes': {(1366,768)}
    },
    'seedlite': {'id': 'bytedance:1@1',
        'desc': 'Seedance 1.0 Lite, 8-15¢ 10sec 864x480',
        'duration': [2,12], 'fps': 24, 'size': (1248, 704), 'sizes': SIZES['bytedance']
    },
    'default': {
        'duration': [4,10], 'size': (1280, 720), 'sizes': {(1920,1080), (1280,720), (1080,1920), (1080,1080)}
    }
}

class RunwareGen:
    def __init__(self, a):
        self.a = a
        self.api_key = getattr(a, 'runware_api_key', None) or os.getenv('RUNWARE_API_KEY')
        if not self.api_key:
            raise RuntimeError('Runware API key missing. Provide --runware_api_key or set RUNWARE_API_KEY env var')

        self.timeout = getattr(a, 'poll_timeout', 300)
        self.max_retries = getattr(a, 'max_retries', 3)

        self.set_models(a.img_model, a.vid_model)

    def set_models(self, img_model=None, vid_model=None):
        if img_model:
            try:
                self.img_cfg = IMAGE_CONFIG[img_model]
                self.img_model_id = IMAGE_CONFIG[img_model]['id']
            except:
                raise RuntimeError(f'!! Runware Image model not found: {img_model} !!')
        if vid_model:
            try:
                self.vid_cfg = VIDEO_CONFIG[vid_model]
                self.vid_model_id = VIDEO_CONFIG[vid_model]['id']
            except:
                raise RuntimeError(f'!! Runware Video model not found: {vid_model} !!')

    # ── Async/Sync Bridge ──────────────────────────────────────────────────────

    def _run_async(self, coro):
        """Run async coroutine from sync context, handling existing event loops"""
        try:
            asyncio.get_running_loop()
            # In async context - use thread with fresh loop
            with concurrent.futures.ThreadPoolExecutor() as ex:
                return ex.submit(self._run_in_new_loop, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    def _run_in_new_loop(self, coro):
        """Run coroutine in a fresh event loop (for ThreadPoolExecutor)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # ── Image Helpers ──────────────────────────────────────────────────────────

    def _to_base64(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _crop_ar(self, image_path, target_w, target_h, verbose=False):
        """Crop image to match target aspect ratio"""
        target_ar = target_w / target_h
        with Image.open(image_path) as img:
            w, h = img.size
            if abs(w/h - target_ar) < 0.01:
                return image_path  # Already correct AR

            if w/h > target_ar:  # Too wide
                new_w, new_h = int(h * target_ar), h
                left = (w - new_w) // 2
                box = (left, 0, left + new_w, h)
            else:  # Too tall
                new_w, new_h = w, int(w / target_ar)
                top = (h - new_h) // 2
                box = (0, top, w, top + new_h)

            cropped = img.crop(box)
            out_path = f"{os.path.splitext(image_path)[0]}_cropped{os.path.splitext(image_path)[1]}"
            cropped.save(out_path, quality=95)
            if verbose: print(f"  Cropped {w}x{h} -> {new_w}x{new_h}")
            return out_path

    def _prepare_images(self, items, mode='ref', target_size=None, verbose=False):
        """ Prepare images for API.
        mode='ref': reference images (returns list of base64/uuid strings)
        mode='frame': frame images for I2V (returns list of IFrameImage, with cropping)
        """
        if not items or None in items: return []

        result = []
        for idx, item in enumerate(items):
            # Extract path or uuid
            if isinstance(item, dict):
                path = item.get('path')
                uuid = item.get('uuid')
            elif isinstance(item, str) and os.path.isfile(item):
                path, uuid = item, None
            else:
                path, uuid = None, str(item) if item else None

            # Convert to base64 or use uuid
            if path:
                if mode == 'frame' and target_size:
                    path = self._crop_ar(path, *target_size, verbose)
                img_data = self._to_base64(path)
            elif uuid:
                img_data = uuid
            else:
                continue

            # Wrap in IFrameImage for frame mode
            if mode == 'frame':
                pos = 'first' if idx == 0 else ('last' if idx == len(items) - 1 else idx)
                result.append(IFrameImage(inputImage=img_data, frame=pos))
            else:
                result.append(img_data)

        if verbose and result:
            print(f" got {len(result)} {'frame' if mode == 'frame' else 'reference'} images")
        return result

    # ── Download ───────────────────────────────────────────────────────────────

    def _download(self, url, out_path, verbose=False):
        """Download file with retries"""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        for attempt in range(5):
            try:
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk: f.write(chunk)
                return out_path
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < 4:
                    wait = (attempt + 1) * 2
                    if verbose: print(f"!! Download failed, retry in {wait}s: {e}")
                    import time; time.sleep(wait)
                else:
                    raise

    # ── Public API ─────────────────────────────────────────────────────────────

    def gen_img(self, fname, prompt, wdir='tmp', ref_images=None, verbose=False):
        """Text-to-image generation (sync interface)"""
        return self._run_async(self._gen_img(fname, prompt, wdir, ref_images, verbose))

    def gen_vid(self, fname, prompt, duration, images, fps=24, ref_images=None, wdir='tmp', verbose=False):
        """Image-to-video generation (sync interface)"""
        return self._run_async(self._gen_vid(fname, prompt, duration, images, fps, ref_images, wdir, verbose))

    # ── Async Implementations ──────────────────────────────────────────────────

    async def _gen_img(self, fname, prompt, wdir, ref_images, verbose):
        """Async T2I with retry"""
        out_dir = os.path.join(self.a.out_dir, wdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, basename(fname) + "-rw.png")
        if os.path.isfile(out_path): return out_path # already exists

        try:
            w, h = map(int, self.a.img_size.split('-'))
        except:
            w, h = None, None
        if not (w, h) in self.img_cfg['sizes']: w, h = self.img_cfg['size']
        if verbose: print(f" Using size {w}x{h} for {self.img_model_id}")

        params = {'positivePrompt': prompt, 'model': self.img_model_id, 'width': w, 'height': h}
        params['numberResults'] = max(self.img_cfg.get('num', 0), 1)

        refs = self._prepare_images(ref_images, mode='ref', verbose=verbose)
        if refs: params['referenceImages'] = refs

        for attempt in range(self.max_retries):
            client = None
            try:
                client = Runware(api_key=self.api_key)
                await asyncio.wait_for(client.connect(), timeout=30.0)
                images = await asyncio.wait_for(client.imageInference(requestImage=IImageInference(**params)), timeout=self.timeout)
                if not images: raise RuntimeError("No images generated")

                for i, image in enumerate(images):
                    out_path_ = out_path.replace('.png',f'-{i}.png') if len(images) > 1 else out_path
                    downloaded = self._download(image.imageURL, out_path_, verbose)
                return downloaded

            except Exception as e:
                if is_moderation_error(str(e)):
                    print(f"!! T2I CENSORED")
                    raise Censored(str(e)) from None
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    if attempt > 0: print(f"!! T2I attempt {attempt+1}/{self.max_retries} failed: {e}, retry in {wait}s ..")
                    await asyncio.sleep(wait)
                else:
                    print(f"!! T2I failed after {self.max_retries} attempts: {e}")
                    raise
            finally:
                if client:
                    try: await client.disconnect()
                    except: pass

    async def _gen_vid(self, fname, prompt, duration, images, fps, ref_images, wdir, verbose):
        """Async I2V with retry"""
        duration = min(self.vid_cfg['duration'][1], max(self.vid_cfg['duration'][0], math.ceil(duration)))
        try:
            w, h = map(int, self.a.vid_size.split('-'))
        except:
            w, h = None, None
        if not (w, h) in self.vid_cfg['sizes']: w, h = self.vid_cfg['size']
        if verbose: print(f" Using size {w}x{h} for {self.vid_model_id}")

        params = {'positivePrompt': prompt, 'model': self.vid_model_id, 'duration': duration, 'width': w, 'height': h}
        frames = self._prepare_images(images, mode='frame', target_size=(w, h), verbose=verbose)
        if frames: params['frameImages'] = frames
        refs = self._prepare_images(ref_images, mode='ref', verbose=verbose)
        if refs: params['referenceImages'] = refs

        for attempt in range(self.max_retries):
            client = None
            try:
                client = Runware(api_key=self.api_key)
                await asyncio.wait_for(client.connect(), timeout=30.0)
                videos = await asyncio.wait_for(client.videoInference(requestVideo=IVideoInference(**params)), timeout=self.timeout)
                if not videos: raise RuntimeError("No videos generated")

                result = videos[0]
                out_dir = os.path.join(self.a.out_dir, wdir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, basename(fname) + "-rw.mp4")

                try:
                    return self._download(result.videoURL, out_path, verbose)
                except Exception as dl_err:
                    # Download failed but video generated - save info
                    print(f"!! Download failed: {dl_err}")
                    info_path = out_path.replace('.mp4', '.info.txt')
                    with open(info_path, 'w') as f:
                        f.write(f"UUID: {getattr(result, 'videoUUID', 'N/A')}\nURL: {result.videoURL}\nError: {dl_err}\n")
                    return info_path

            except Exception as e:
                if is_moderation_error(str(e)):
                    print(f"!! I2V CENSORED")
                    raise Censored(str(e)) from None
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    if attempt > 0: print(f"!! I2V attempt {attempt+1}/{self.max_retries} failed: {e}, retry in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"!! I2V failed after {self.max_retries} attempts: {e}")
                    raise
            finally:
                if client:
                    try: await client.disconnect()
                    except: pass
