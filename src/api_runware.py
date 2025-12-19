
import os
import base64
import asyncio
import concurrent.futures
import requests
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

# Video configuration per model family: sizes + duration limits
VIDEO_CONFIG = {
    'bytedance': {
        'sizes': {(864,480), (736,544), (640,640), (544,736), (480,864), (960,416),
                  (1248,704), (1120,832), (960,960), (832,1120), (704,1248), (1504,640),
                  (1920,1088), (1664,1248), (1440,1440), (1248,1664), (1088,1920), (2176,928)},
        'default_size': (1248, 704),
        'duration': [2,12]
    },
    'klingai': {
        'sizes': {(1920,1080), (1080,1920), (1080,1080)},
        'default_size': (1920, 1080),
        'duration': [3,10]
    },
    'minimax': {
        'sizes': {(1366,768)},
        'default_size': (1366, 768),
        'duration': [3,6]
    },
    'google': {  # veo
        'sizes': {(1920,1080), (1280,720), (1080,1920)},
        'default_size': (1280, 720),
        'duration': [2,8]
    },
    'runway': {
        'sizes': {(1280,720)},
        'default_size': (1280, 720),
        'duration': [2,10]
    },
    'openai': {  # sora
        'sizes': {(1280,720)},
        'default_size': (1280, 720),
        'duration': [4,12]
    },
    'default': {
        'sizes': {(1920,1080), (1280,720), (1080,1920), (1080,1080)},
        'default_size': (1280, 720),
        'duration': [4,10]
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

        self.img_model = a.img_model
        # self.img_model = 'bfl:5@1' # flux2 pro, 8 refs, 1344-768, 3¢
        # self.img_model = 'runware:400@1' # flux2 dev, 1344-768, 1.5¢
        # self.img_model = 'google:2@3' # imagen 4 fast, 1280-896 1408-768, 2¢
        # self.img_model = 'bfl:3@1' # flux kontext, 2 refs, 1392-752, 4¢
        self.vid_model = a.vid_model
        # self.vid_model = 'bytedance:2@2' # seedance pro fast 24fps, 3-12 sec, 5-8¢
        # self.vid_model = 'bytedance:1@1' # seedance lite 24fps, 3-12 sec, 8-15¢

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

    def gen_t2i(self, fname, prompt, wdir='tmp', ref_images=None, verbose=False):
        """Text-to-image generation (sync interface)"""
        return self._run_async(self._gen_t2i(fname, prompt, wdir, ref_images, verbose))

    def gen_i2v(self, fname, prompt, duration, images, fps=24, ref_images=None, wdir='tmp', verbose=False):
        """Image-to-video generation (sync interface)"""
        return self._run_async(self._gen_i2v(fname, prompt, duration, images, fps, ref_images, wdir, verbose))

    async def _gen_t2i(self, fname, prompt, wdir, ref_images, verbose):
        """Async T2I with retry"""
        width, height = map(int, self.a.img_size.split('-'))
        refs = self._prepare_images(ref_images, mode='ref', verbose=verbose)

        for attempt in range(self.max_retries):
            client = None
            try:
                client = Runware(api_key=self.api_key)
                await asyncio.wait_for(client.connect(), timeout=30.0)

                params = {'positivePrompt': prompt, 'model': self.img_model, 'width': width, 'height': height}
                if refs: params['referenceImages'] = refs

                images = await asyncio.wait_for(client.imageInference(requestImage=IImageInference(**params)), timeout=self.timeout)
                if not images:
                    raise RuntimeError("No images generated")

                result = images[0]
                out_dir = os.path.join(self.a.out_dir, wdir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, basename(fname) + "-rw.png")
                downloaded = self._download(result.imageURL, out_path, verbose)

                uuid = getattr(result, 'imageUUID', None)
                return {'path': downloaded, 'uuid': uuid} if uuid else downloaded

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

    async def _gen_i2v(self, fname, prompt, duration, images, fps, ref_images, wdir, verbose):
        """Async I2V with retry"""
        cfg = self._get_vid_config()
        duration = min(cfg['duration'][1], max(cfg['duration'][0], int(duration)))
        try:
            w, h = map(int, self.a.vid_size.split('-'))
        except:
            w, h = 1080, 1920
        w, h = self._resolve_vid_size(w, h)
        if verbose: print(f" Using size {w}x{h} for {self.vid_model}")

        frames = self._prepare_images(images, mode='frame', target_size=(w, h), verbose=verbose)
        refs = self._prepare_images(ref_images, mode='ref', verbose=verbose)

        for attempt in range(self.max_retries):
            client = None
            try:
                client = Runware(api_key=self.api_key)
                await asyncio.wait_for(client.connect(), timeout=30.0)

                params = {'positivePrompt': prompt, 'model': self.vid_model, 'duration': duration, 'width': w, 'height': h}
                if frames: params['frameImages'] = frames
                if refs: params['referenceImages'] = refs

                if verbose: print(f" requesting I2V... {w}x{h}")
                videos = await asyncio.wait_for(client.videoInference(requestVideo=IVideoInference(**params)), timeout=self.timeout)
                if verbose: print(f" I2V done!")

                if not videos:
                    raise RuntimeError("No videos generated")

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
        if not items: return []

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

    def _get_vid_config(self):
        """Get video configuration (size, duration limits) for current model"""
        model = self.vid_model.lower()
        for key, cfg in VIDEO_CONFIG.items():
            if key in model:
                return cfg
        return VIDEO_CONFIG['default']

    def _resolve_vid_size(self, w, h):
        """Resolve video size to supported dimensions for current model"""
        cfg = self._get_vid_config()
        return (w, h) if (w, h) in cfg['sizes'] else cfg['default_size']

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

