#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# Suppress tqdm progress bars from funasr before importing
import os
os.environ['TQDM_DISABLE'] = '1'

import re
import logging
import argparse
import numpy as np
import soundfile as sf
import subprocess
import shutil
import copy
import librosa

# Fix PIL.Image.ANTIALIAS compatibility for Pillow 10.0+
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except ImportError:
    pass

from moviepy.editor import *
import moviepy.editor as mpy
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import SubtitlesClip, TextClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from utils.subtitle_utils import generate_srt, generate_srt_clip
from utils.argparse_tools import ArgumentParser, get_commandline_args
from utils.trans_utils import pre_proc, proc, write_state, load_state, proc_spk, convert_pcm_to_float

# Configure MoviePy to use ImageMagick
try:
    # Try to find magick in PATH
    magick_path = shutil.which('magick')
    if magick_path:
        change_settings({"IMAGEMAGICK_BINARY": magick_path})
        logging.info(f"[CONFIG] ImageMagick configured at: {magick_path}")
    else:
        logging.warning("[CONFIG] ImageMagick not found in PATH. Subtitles will be skipped.")
except Exception as e:
    logging.warning(f"[CONFIG] Could not configure ImageMagick: {e}")


class VideoClipper():
    def __init__(self, funasr_model):
        logging.warning("Initializing VideoClipper.")
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0
        # Set font path relative to project root (parent of funclip directory)
        self.font_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "font", "STHeitiMedium.ttc")

    def _crop_center(self, video_clip, target_aspect):
        """Default center crop fallback"""
        w, h = video_clip.w, video_clip.h
        current_aspect = w / h

        if current_aspect > target_aspect:
            # Video is wider than target - crop width
            new_w = int(h * target_aspect)
            new_h = h
            crop_x = (w - new_w) // 2
            crop_y = 0
        else:
            # Video is taller than target - crop height
            new_w = w
            new_h = int(w / target_aspect)
            crop_x = 0
            crop_y = (h - new_h) // 2

        video_clip = video_clip.crop(x1=crop_x, y1=crop_y, x2=crop_x+new_w, y2=crop_y+new_h)
        return video_clip

    def _crop_face_aware(self, video_clip, target_aspect):
        """Intelligently crop video to TikTok format, favoring face/content areas.
        Uses multi-frame sampling to ensure faces stay centered throughout the video."""
        try:
            try:
                import cv2
            except ImportError as ie:
                logging.warning(f"[CROP] OpenCV not available: {ie}, falling back to center crop")
                return self._crop_center(video_clip, target_aspect)

            w, h = video_clip.w, video_clip.h
            current_aspect = w / h

            # Multi-frame sampling approach: Sample 5 frames throughout the video
            sample_times = [
                video_clip.duration * 0.10,   # 10%
                video_clip.duration * 0.30,   # 30%
                video_clip.duration * 0.50,   # 50%
                video_clip.duration * 0.70,   # 70%
                video_clip.duration * 0.90,   # 90%
            ]

            all_faces = []
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Detect faces in all sampled frames with relaxed parameters for better detection
            for sample_time in sample_times:
                try:
                    frame = video_clip.get_frame(sample_time)
                    frame_cv = cv2.cvtColor((frame * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
                    # More relaxed detection: scaleFactor=1.05 is more sensitive, minNeighbors=3 catches more faces
                    faces = face_cascade.detectMultiScale(frame_cv, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
                    if len(faces) > 0:
                        all_faces.extend(faces)
                except Exception as frame_err:
                    logging.debug(f"[CROP] Could not process frame at {sample_time}: {frame_err}")
                    continue

            if len(all_faces) > 0:
                # Found faces - use the largest face as reference (usually the main subject)
                all_faces = np.array(all_faces)
                face_areas = all_faces[:, 2] * all_faces[:, 3]  # width * height
                largest_face_idx = np.argmax(face_areas)
                fx, fy, fw, fh = all_faces[largest_face_idx]

                # Calculate the region we want to keep (face + margins)
                face_center_x = int(fx + fw / 2)
                face_center_y = int(fy + fh / 2)
                face_left = int(fx)
                face_right = int(fx + fw)
                face_top = int(fy)
                face_bottom = int(fy + fh)

                logging.info(f"[CROP] Main face detected at x={face_center_x}, y={face_center_y}, size={fw}x{fh}")

                # Determine crop area ensuring face is well-positioned
                if current_aspect > target_aspect:
                    # Wider than target - crop width
                    new_w = int(h * target_aspect)
                    new_h = h

                    # Position crop: keep face visible with good margins
                    # Try to place face at 35% from left edge (not dead center, gives room to frame)
                    crop_x = face_center_x - int(new_w * 0.35)
                    # Ensure face is fully visible: left edge must be at least 50px from crop boundary
                    min_crop_x = face_left - int(new_w * 0.25)
                    max_crop_x = face_right + int(new_w * 0.25) - new_w
                    crop_x = max(min_crop_x, min(crop_x, max_crop_x))
                    crop_x = max(0, min(crop_x, w - new_w))
                    crop_y = 0
                else:
                    # Taller than target - crop height
                    new_w = w
                    new_h = int(w / target_aspect)
                    crop_x = 0

                    # Position crop: keep face visible with good margins
                    # Try to place face at 35% from top (upper-middle for better framing)
                    crop_y = face_center_y - int(new_h * 0.35)
                    # Ensure face is fully visible: top edge must have margin
                    min_crop_y = face_top - int(new_h * 0.25)
                    max_crop_y = face_bottom + int(new_h * 0.25) - new_h
                    crop_y = max(min_crop_y, min(crop_y, max_crop_y))
                    crop_y = max(0, min(crop_y, h - new_h))

                logging.info(f"[CROP] Face-aware crop: x={crop_x}, y={crop_y}, size={new_w}x{new_h}")
                video_clip = video_clip.crop(x1=crop_x, y1=crop_y, x2=crop_x+new_w, y2=crop_y+new_h)
                return video_clip
            else:
                # No faces found - use upward bias (where content usually is)
                logging.info("[CROP] No faces detected across samples, using content-aware crop (upper bias)")
                if current_aspect > target_aspect:
                    new_w = int(h * target_aspect)
                    new_h = h
                    crop_x = (w - new_w) // 2
                    crop_y = 0
                else:
                    new_w = w
                    new_h = int(w / target_aspect)
                    crop_x = 0
                    crop_y = max(0, int(h * 0.1))  # Shift up 10% - capture upper content

                video_clip = video_clip.crop(x1=crop_x, y1=crop_y, x2=crop_x+new_w, y2=crop_y+new_h)
                return video_clip

        except Exception as e:
            logging.warning(f"[CROP] Face-aware crop failed ({type(e).__name__}), falling back to center crop")
            return self._crop_center(video_clip, target_aspect)

    def resize_to_tiktok(self, video_clip):
        """Crop and resize video to TikTok format (9:16 aspect ratio, 1080x1920)"""
        try:
            target_w, target_h = 1080, 1920
            target_aspect = target_w / target_h  # 0.5625

            # Try face-aware cropping first, fallback to center crop
            video_clip = self._crop_face_aware(video_clip, target_aspect)

            # Resize to final TikTok dimensions (1080x1920)
            video_clip = video_clip.resize((target_w, target_h))

            logging.info("Video cropped and resized to TikTok format (1080x1920)")
            return video_clip
        except Exception as e:
            logging.error(f"Error resizing to TikTok format: {str(e)}")
            return video_clip  # Return original if resize fails


    def recog(self, audio_input, sd_switch='no', state=None, hotwords="", output_dir=None):
        if state is None:
            state = {}
        sr, data = audio_input

        # Convert to float64 consistently (includes data type checking)
        data = convert_pcm_to_float(data)

        # assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
        if sr != 16000: # resample with librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        if len(data.shape) == 2:  # multi-channel wav input
            logging.warning("Input wav shape: {}, only first channel reserved.".format(data.shape))
            data = data[:,0]
        state['audio_input'] = (sr, data)
        if sd_switch == 'Yes':
            rec_result = self.funasr_model.generate(data,
                                                    return_spk_res=True,
                                                    return_raw_text=True,
                                                    is_final=True,
                                                    output_dir=output_dir,
                                                    hotword=hotwords,
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_srt = generate_srt(rec_result[0]['sentence_info'])
            state['sd_sentences'] = rec_result[0]['sentence_info']
        else:
            rec_result = self.funasr_model.generate(data,
                                                    return_spk_res=False,
                                                    sentence_timestamp=True,
                                                    return_raw_text=True,
                                                    is_final=True,
                                                    hotword=hotwords,
                                                    output_dir=output_dir,
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_srt = generate_srt(rec_result[0]['sentence_info'])
        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        res_text = rec_result[0]['text']
        return res_text, res_srt, state

    def clip(self, dest_text, start_ost, end_ost, state, dest_spk=None, output_dir=None, timestamp_list=None):
        # get from state
        audio_input = state['audio_input']
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        sr, data = audio_input
        data = data.astype(np.float64)

        if timestamp_list is None:
            all_ts = []
            if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
                for _dest_text in dest_text.split('#'):
                    if '[' in _dest_text:
                        match = re.search(r'\[(\d+),\s*(\d+)\]', _dest_text)
                        if match:
                            offset_b, offset_e = map(int, match.groups())
                            log_append = ""
                        else:
                            offset_b, offset_e = 0, 0
                            log_append = "(Bracket detected in dest_text but offset time matching failed)"
                        _dest_text = _dest_text[:_dest_text.find('[')]
                    else:
                        log_append = ""
                        offset_b, offset_e = 0, 0
                    _dest_text = pre_proc(_dest_text)
                    ts = proc(recog_res_raw, timestamp, _dest_text)
                    for _ts in ts: all_ts.append([_ts[0]+offset_b*16, _ts[1]+offset_e*16])
                    if len(ts) > 1 and match:
                        log_append += '(offsets detected but No.{} sub-sentence matched to {} periods in audio, \
                            offsets are applied to all periods)'
            else:
                for _dest_spk in dest_spk.split('#'):
                    ts = proc_spk(_dest_spk, state['sd_sentences'])
                    for _ts in ts: all_ts.append(_ts)
                log_append = ""
        else:
            all_ts = timestamp_list
        ts = all_ts
        # ts.sort()
        srt_index = 0
        clip_srt = ""
        if len(ts):
            start, end = ts[0]
            start = min(max(0, start+start_ost*16), len(data))
            end = min(max(0, end+end_ost*16), len(data))
            res_audio = data[start:end]
            start_end_info = "from {} to {}".format(start/16000, end/16000)
            srt_clip, _, srt_index = generate_srt_clip(sentences, start/16000.0, end/16000.0, begin_index=srt_index)
            clip_srt += srt_clip
            for _ts in ts[1:]:  # multiple sentence input or multiple output matched
                start, end = _ts
                start = min(max(0, start+start_ost*16), len(data))
                end = min(max(0, end+end_ost*16), len(data))
                start_end_info += ", from {} to {}".format(start, end)
                res_audio = np.concatenate([res_audio, data[start+start_ost*16:end+end_ost*16]], -1)
                srt_clip, _, srt_index = generate_srt_clip(sentences, start/16000.0, end/16000.0, begin_index=srt_index-1)
                clip_srt += srt_clip
        if len(ts):
            message = "{} periods found in the speech: ".format(len(ts)) + start_end_info + log_append
        else:
            message = "No period found in the speech, return raw speech. You may check the recognition result and try other destination text."
            res_audio = data
        return (sr, res_audio), message, clip_srt

    def video_recog(self, video_filename, sd_switch='no', hotwords="", output_dir=None):
        video = mpy.VideoFileClip(video_filename)
        # Extract the base name, add '_clip.mp4', and 'wav'
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            clip_video_file = base_name + '_clip.mp4'
            audio_file = base_name + '.wav'
            audio_file = os.path.join(output_dir, audio_file)
        else:
            base_name, _ = os.path.splitext(video_filename)
            clip_video_file = base_name + '_clip.mp4'
            audio_file = base_name + '.wav'

        if video.audio is None:
            logging.error("No audio information found.")
            sys.exit(1)

        video.audio.write_audiofile(audio_file)
        wav = librosa.load(audio_file, sr=16000)[0]
        # delete the audio file after processing
        if os.path.exists(audio_file):
            os.remove(audio_file)
        state = {
            'video_filename': video_filename,
            'clip_video_file': clip_video_file,
            'video': video,
        }
        # res_text, res_srt = self.recog((16000, wav), state)
        return self.recog((16000, wav), sd_switch, state, hotwords, output_dir)

    def video_clip(self,
                   dest_text,
                   start_ost,
                   end_ost,
                   state,
                   font_size=32,
                   font_color='white',
                   add_sub=False,
                   dest_spk=None,
                   output_dir=None,
                   timestamp_list=None):
        logging.info(f"[DEBUG_VIDEO_CLIP_START] add_sub={add_sub}, has_ts_list={timestamp_list is not None}")
        # get from state
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        video = state['video']
        clip_video_file = state['clip_video_file']
        video_filename = state['video_filename']

        if timestamp_list is None:
            all_ts = []
            if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
                for _dest_text in dest_text.split('#'):
                    if '[' in _dest_text:
                        match = re.search(r'\[(\d+),\s*(\d+)\]', _dest_text)
                        if match:
                            offset_b, offset_e = map(int, match.groups())
                            log_append = ""
                        else:
                            offset_b, offset_e = 0, 0
                            log_append = "(Bracket detected in dest_text but offset time matching failed)"
                        _dest_text = _dest_text[:_dest_text.find('[')]
                    else:
                        offset_b, offset_e = 0, 0
                        log_append = ""
                    # import pdb; pdb.set_trace()
                    _dest_text = pre_proc(_dest_text)
                    ts = proc(recog_res_raw, timestamp, _dest_text.lower())
                    for _ts in ts: all_ts.append([_ts[0]+offset_b*16, _ts[1]+offset_e*16])
                    if len(ts) > 1 and match:
                        log_append += '(offsets detected but No.{} sub-sentence matched to {} periods in audio, \
                            offsets are applied to all periods)'
            else:
                for _dest_spk in dest_spk.split('#'):
                    ts = proc_spk(_dest_spk, state['sd_sentences'])
                    for _ts in ts: all_ts.append(_ts)
        else:  # AI clip pass timestamp as input directly
            logging.info(f"[DEBUG_VIDEO_CLIP_TS_CONVERT] Converting {len(timestamp_list)} timestamps")
            all_ts = [[i[0]*16.0, i[1]*16.0] for i in timestamp_list]
            logging.info(f"[DEBUG_VIDEO_CLIP_ALL_TS] Result: {all_ts}")

        srt_index = 0
        time_acc_ost = 0.0
        ts = all_ts
        # ts.sort()
        clip_srt = ""
        logging.info(f"[DEBUG_VIDEO_CLIP_FINAL_TS] len(ts)={len(ts)}")
        if len(ts):
            if self.lang == 'en' and isinstance(sentences, str):
                sentences = sentences.split()
            start, end = ts[0][0] / 16000, ts[0][1] / 16000
            srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index, time_acc_ost=time_acc_ost)
            start, end = start+start_ost/1000.0, end+end_ost/1000.0
            video_clip = video.subclip(start, end)
            start_end_info = "from {} to {}".format(start, end)
            clip_srt += srt_clip
            # Store subtitle info for later (after cropping)
            video_clip._subs_to_add = (subs, add_sub)
            concate_clip = [video_clip]
            time_acc_ost += end+end_ost/1000.0 - (start+start_ost/1000.0)
            for _ts in ts[1:]:
                start, end = _ts[0] / 16000, _ts[1] / 16000
                srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index-1, time_acc_ost=time_acc_ost)
                if not len(subs):
                    continue
                chi_subs = []
                sub_starts = subs[0][0][0]
                for sub in subs:
                    chi_subs.append(((sub[0][0]-sub_starts, sub[0][1]-sub_starts), sub[1]))
                start, end = start+start_ost/1000.0, end+end_ost/1000.0
                _video_clip = video.subclip(start, end)
                start_end_info += ", from {} to {}".format(str(start)[:5], str(end)[:5])
                clip_srt += srt_clip
                # Store subtitle info for later (after cropping)
                _video_clip._subs_to_add = (chi_subs, add_sub)
                concate_clip.append(copy.copy(_video_clip))
                time_acc_ost += end+end_ost/1000.0 - (start+start_ost/1000.0)
            message = "{} periods found in the audio: ".format(len(ts)) + start_end_info
            logging.warning("Saving {} separate clips...".format(len(concate_clip)))

            # Default output folder: ./clips (in FunClip directory)
            if output_dir is None or not output_dir.strip():
                output_dir = os.path.join(os.getcwd(), "clips")

            os.makedirs(output_dir, exist_ok=True)
            _, file_with_extension = os.path.split(clip_video_file)
            clip_video_file_name, _ = os.path.splitext(file_with_extension)

            # Save each clip segment separately
            clip_video_files = []
            for seg_idx, video_segment in enumerate(concate_clip, 1):
                # Resize to TikTok format for each segment
                resized_clip = self.resize_to_tiktok(video_segment)

                # Add subtitles AFTER cropping so they're sized for the final frame
                if hasattr(video_segment, '_subs_to_add'):
                    subs_data, should_add_sub = video_segment._subs_to_add
                    if should_add_sub and subs_data:
                        try:
                            # Create TextClip with center alignment, using responsive width based on cropped dimensions
                            # Width is 90% of cropped video width with 5% padding on each side
                            text_width = int(resized_clip.w * 0.90)
                            generator = lambda txt: TextClip(txt, font=self.font_path, fontsize=font_size, color=font_color, align='center', method='caption', size=(text_width, None))
                            subtitles = SubtitlesClip(subs_data, generator)
                            # Position subtitles in middle of bottom half: 65% from top of cropped frame
                            resized_clip = CompositeVideoClip([resized_clip, subtitles.set_pos(('center', 0.65), relative=True)])
                        except Exception as e:
                            logging.warning(f"Subtitles skipped for segment {seg_idx} (ImageMagick may not be installed): {type(e).__name__}")

                output_file = os.path.join(output_dir, "{}_segment_{}.mp4".format(clip_video_file_name, seg_idx))
                temp_audio_file = os.path.join(output_dir, "{}_tempaudio_{}.mp4".format(clip_video_file_name, seg_idx))
                logging.info(f"Saving clip segment {seg_idx}/{len(concate_clip)} to: {output_file}")
                resized_clip.write_videofile(output_file, audio_codec="aac", temp_audiofile=temp_audio_file, verbose=False, logger=None)
                clip_video_files.append(output_file)
                self.GLOBAL_COUNT += 1

            clip_video_file = clip_video_files[0] if clip_video_files else video_filename
            message = "{} separate clips saved to {}".format(len(clip_video_files), output_dir)
        else:
            clip_video_file = video_filename
            message = "No period found in the audio, return raw speech. You may check the recognition result and try other destination text."
            srt_clip = ''
        return clip_video_file, message, clip_srt


def get_parser():
    parser = ArgumentParser(
        description="ClipVideo Argument",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        help="Stage, 0 for recognizing and 1 for clipping",
        required=True
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file path",
        required=True
    )
    parser.add_argument(
        "--sd_switch",
        type=str,
        choices=("no", "yes"),
        default="no",
        help="Turn on the speaker diarization or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Output files path",
    )
    parser.add_argument(
        "--dest_text",
        type=str,
        default=None,
        help="Destination text string for clipping",
    )
    parser.add_argument(
        "--dest_spk",
        type=str,
        default=None,
        help="Destination spk id for clipping",
    )
    parser.add_argument(
        "--start_ost",
        type=int,
        default=0,
        help="Offset time in ms at beginning for clipping"
    )
    parser.add_argument(
        "--end_ost",
        type=int,
        default=0,
        help="Offset time in ms at ending for clipping"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='zh',
        help="language"
    )
    return parser


def runner(stage, file, sd_switch, output_dir, dest_text, dest_spk, start_ost, end_ost, output_file, config=None, lang='zh'):
    audio_suffixs = ['.wav','.mp3','.aac','.m4a','.flac']
    video_suffixs = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    _,ext = os.path.splitext(file)
    if ext.lower() in audio_suffixs:
        mode = 'audio'
    elif ext.lower() in video_suffixs:
        mode = 'video'
    else:
        logging.error("Unsupported file format: {}\n\nplease choise one of the following: {}".format(file),audio_suffixs+video_suffixs)
        sys.exit(1) # exit if the file is not supported
    while output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if stage == 1:
        from funasr import AutoModel
        # initialize funasr automodel
        logging.warning("Initializing modelscope asr pipeline.")
        if lang == 'zh':
            funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                    spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                    disable_update=True)
            audio_clipper = VideoClipper(funasr_model)
            audio_clipper.lang = 'zh'
        elif lang == 'en':
            funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                disable_update=True)
            audio_clipper = VideoClipper(funasr_model)
            audio_clipper.lang = 'en'
        if mode == 'audio':
            logging.warning("Recognizing audio file: {}".format(file))
            wav, sr = librosa.load(file, sr=16000)
            res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch)
        if mode == 'video':
            logging.warning("Recognizing video file: {}".format(file))
            res_text, res_srt, state = audio_clipper.video_recog(file, sd_switch)
        total_srt_file = output_dir + '/total.srt'
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
            logging.warning("Write total subtitle to {}".format(total_srt_file))
        write_state(output_dir, state)
        logging.warning("Recognition successed. You can copy the text segment from below and use stage 2.")
        print(res_text)
    if stage == 2:
        audio_clipper = VideoClipper(None)
        if mode == 'audio':
            state = load_state(output_dir)
            wav, sr = librosa.load(file, sr=16000)
            state['audio_input'] = (sr, wav)
            (sr, audio), message, srt_clip = audio_clipper.clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            if output_file is None:
                output_file = output_dir + '/result.wav'
            clip_srt_file = output_file[:-3] + 'srt'
            logging.warning(message)
            sf.write(output_file, audio, 16000)
            assert output_file.endswith('.wav'), "output_file must ends with '.wav'"
            logging.warning("Save clipped wav file to {}".format(output_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitle to {}".format(clip_srt_file))
        if mode == 'video':
            state = load_state(output_dir)
            state['video_filename'] = file
            if output_file is None:
                state['clip_video_file'] = file[:-4] + '_clip.mp4'
            else:
                state['clip_video_file'] = output_file
            clip_srt_file = state['clip_video_file'][:-3] + 'srt'
            state['video'] = mpy.VideoFileClip(file)
            clip_video_file, message, srt_clip = audio_clipper.video_clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
            logging.warning("Clipping Log: {}".format(message))
            logging.warning("Save clipped mp4 file to {}".format(clip_video_file))
            with open(clip_srt_file, 'w') as fout:
                fout.write(srt_clip)
                logging.warning("Write clipped subtitle to {}".format(clip_srt_file))


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    runner(**kwargs)


if __name__ == '__main__':
    main()
