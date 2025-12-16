import json
import os
import sys
from io import BytesIO

import requests

# isort: off
import torch
import numpy as np
import tensorrt as trt
# isort: on
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from ocr import OCRSystem

from tensorrt_llm import profiler
from tensorrt_llm._utils import (mpi_rank, str_dtype_to_torch, str_dtype_to_trt,
                      trt_dtype_to_torch)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.enc_dec_model_runner import EncDecModelRunner
from tensorrt_llm.runtime.model_runner import ModelRunner
from tensorrt_llm.runtime.session import Session, TensorInfo


class MultimodalModelRunner:

    def __init__(self, args):
        self.args = args

        self.runtime_rank = mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # parse model type from visual engine config
        with open(os.path.join(self.args.visual_engine_dir, "config.json"),
                  "r") as f:
            config = json.load(f)
        self.model_type = config['builder_config']['model_type']
        self.vision_precision = config['builder_config']['precision']
        self.decoder_llm = not (
            't5' in self.model_type
        )  # BLIP2-T5 is using encoder-decoder models as LLMs

        self.init_image_encoder()
        self.init_tokenizer()
        self.init_llm()

        # Initialize OCR only if requested
        self.ocr_system = None
        if getattr(self.args, 'ocr', False):
            try:
                self.ocr_system = OCRSystem('/home/xsf/fl/edge/models/ONNX', '/home/xsf/fl/edge/models/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt')
                logger.info('OCR system initialized')
            except Exception as e:
                logger.warning(f'Failed to initialize OCR system: {e}')

    def init_tokenizer(self):
        use_fast = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.hf_model_dir, use_fast=use_fast, use_legacy=False)
        if isinstance(self.tokenizer, bool):
            logger.warning(f"AutoTokenizer.from_pretrained returned {self.tokenizer}, falling back to LlamaTokenizerFast")
            from transformers import LlamaTokenizerFast
            self.tokenizer = LlamaTokenizerFast.from_pretrained(self.args.hf_model_dir)

        try:
            # Some tokenizer implementations or misconfigurations may result in
            # `self.tokenizer` being an unexpected type (e.g. bool). Guard
            # against that and attempt to reload a standard HF tokenizer if
            # possible.
            self.tokenizer.padding_side = "right"
        except Exception:
            logger.warning("tokenizer object is invalid or does not support padding_side; attempting to reload AutoTokenizer")
            try:
                # Try to reload a standard tokenizer from the HF model dir.
                # Prefer use_fast=False for compatibility.
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_dir, use_fast=False, use_legacy=False)
                self.tokenizer.padding_side = "right"
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer properly: {e}")
                # Re-raise the original exception to preserve behavior
                raise

    def init_image_encoder(self):
        vision_encoder_path = os.path.join(self.args.visual_engine_dir,
                                           self.args.visual_engine_name)
        logger.info(f'Loading engine from {vision_encoder_path}')
        with open(vision_encoder_path, 'rb') as f:
            engine_buffer = f.read()
        logger.info(f'Creating session from engine {vision_encoder_path}')
        self.visual_encoder_session = Session.from_serialized_engine(
            engine_buffer)
        if self.model_type in ["phi-3-vision", "llava_next"]:
            self.image_newlines = {}
            image_newlines_path = os.path.join(self.args.visual_engine_dir,
                                               'image_newlines.safetensors')
            with safe_open(image_newlines_path,
                           framework="pt",
                           device=self.device) as f:
                for k in f.keys():
                    self.image_newlines[k] = f.get_tensor(k)

    def init_llm(self):
        if self.decoder_llm:
            self.model = ModelRunner.from_dir(self.args.llm_engine_dir,
                                              rank=mpi_rank(),
                                              debug_mode=False,
                                              stream=self.stream,
                                              enable_context_fmha_fp32_acc=self.
                                              args.enable_context_fmha_fp32_acc)
            self.model_config = self.model.session._model_config
            self.runtime_mapping = self.model.session.mapping
        else:
            self.model = EncDecModelRunner.from_engine(
                os.path.basename(self.args.hf_model_dir),
                self.args.llm_engine_dir,
                skip_encoder=self.model_type in ['nougat', 'pix2struct'],
                debug_mode=False,
                stream=self.stream,
                enable_context_fmha_fp32_acc=self.args.
                enable_context_fmha_fp32_acc)
            if self.model_type in ['nougat', 'pix2struct']:
                self.model_config = self.model.decoder_model_config
                self.runtime_mapping = self.model.decoder_runtime_mapping
            else:
                self.model_config = self.model.encoder_model_config
                self.runtime_mapping = self.model.encoder_runtime_mapping


    def preprocess(self, warmup, pre_prompt, post_prompt, image,
                   attention_mask):
        # Simplified preprocess: keep only the LLaVA-style flow.
        # This runner is trimmed to support the 'llava' model type only.
        # Steps: get visual features, tokenize pre/post prompts, assemble
        # fake prompt ids and return inputs for LLM generation.
        if not warmup:
            profiler.start("Vision")

        visual_features, visual_atts = self.get_visual_features(image,
                                                               attention_mask)

        if not warmup:
            profiler.stop("Vision")

        # Tokenize prompts
        pre_input_ids = self.tokenizer(pre_prompt,
                                       return_tensors="pt",
                                       padding=True).input_ids
        if post_prompt[0] is not None:
            post_input_ids = self.tokenizer(post_prompt,
                                            return_tensors="pt",
                                            padding=True).input_ids
            length = pre_input_ids.shape[1] + post_input_ids.shape[1] + visual_atts.shape[1]
        else:
            post_input_ids = None
            length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_lengths = torch.IntTensor([length] * self.args.batch_size).to(
            torch.int32)

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths)

        return input_ids, input_lengths, ptuning_args, visual_features

    def generate(self,
                 pre_prompt,
                 post_prompt,
                 image,
                 decoder_input_ids,
                 max_new_tokens,
                 attention_mask,
                 warmup=False):
        if not warmup:
            profiler.start("Generate")

        input_ids, input_lengths, ptuning_args, visual_features = self.preprocess(
            warmup, pre_prompt, post_prompt, image, attention_mask)
        if warmup: return None

        profiler.start("LLM")
        if self.decoder_llm:
            end_id = self.tokenizer.eos_token_id
            if 'opt' in self.model_type and 'blip2' in self.model_type:
                # For BLIP2-OPT, model outputs a "\n" at the end.
                # we avoid it by using newline as the end token
                end_id = self.tokenizer.encode("\n",
                                               add_special_tokens=False)[0]

            ptuning_args[0] = torch.stack([ptuning_args[0]])
            output_ids = self.model.generate(
                input_ids,
                sampling_config=None,
                prompt_table=ptuning_args[0],
                max_new_tokens=max_new_tokens,
                end_id=end_id,
                pad_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None else
                self.tokenizer.all_special_ids[0],
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                temperature=self.args.temperature,
                repetition_penalty=self.args.repetition_penalty,
                num_beams=self.args.num_beams,
                output_sequence_lengths=False,
                return_dict=False)
        else:
            if self.model_type in ['nougat', 'pix2struct']:
                # Trim encoder input_ids to match visual features shape
                ids_shape = (self.args.batch_size, visual_features.shape[1])
                if self.model_type == 'nougat':
                    input_ids = torch.zeros(ids_shape, dtype=torch.int32)
                elif self.model_type == 'pix2struct':
                    input_ids = torch.ones(ids_shape, dtype=torch.int32)

            output_ids = self.model.generate(
                input_ids,
                decoder_input_ids,
                max_new_tokens,
                num_beams=self.args.num_beams,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                debug_mode=False,
                prompt_embedding_table=ptuning_args[0],
                prompt_tasks=ptuning_args[1],
                prompt_vocab_size=ptuning_args[2],
                attention_mask=attention_mask)

            # Reset input_lengths to match decoder_input_ids
            input_lengths = torch.ones(input_lengths.shape,
                                       dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True)
                for batch_idx in range(self.args.batch_size)
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(self.args.num_beams)
            ] for batch_idx in range(self.args.batch_size)]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image, attention_mask):
        visual_features = {
            'input': image.to(str_dtype_to_torch(self.vision_precision))
        }
        if attention_mask is not None:
            visual_features['attention_mask'] = attention_mask
        tensor_info = [
            TensorInfo('input', str_dtype_to_trt(self.vision_precision),
                       image.shape)
        ]
        if attention_mask is not None:
            tensor_info.append(
                TensorInfo('attention_mask', trt.DataType.INT32,
                           attention_mask.shape))

        visual_output_info = self.visual_encoder_session.infer_shapes(
            tensor_info)

        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs,
                                             self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        return image_embeds, image_atts

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids,
                           input_lengths):
        # Assemble fake prompts which points to image embedding actually
        if hasattr(self, 'num_frames') and (visual_features.shape[1]
                                            == self.num_frames):
            visual_features = visual_features.view(visual_features.shape[0], -1,
                                                   visual_features.shape[-1])

        fake_prompt_id = torch.arange(
            self.model_config.vocab_size, self.model_config.vocab_size +
            visual_features.shape[0] * visual_features.shape[1])
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0],
                                                visual_features.shape[1])

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))

            assert prompt_table.shape[
                1] == hidden_size, "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=str_dtype_to_torch(self.model_config.dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)],
                                dtype=torch.int32).cuda()
            if self.decoder_llm: tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def load_test_image(self):
        # Simplified: return a single PIL Image for use with LLaVA flow.
        img_url = self.args.image_path
        if img_url is None:
            img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'

        if img_url.startswith("http") or img_url.startswith("https"):
            image = Image.open(
                requests.get(img_url, stream=True, timeout=5).raw).convert('RGB')
        else:
            image = Image.open(img_url).convert("RGB")

        return image

    def setup_inputs(self, input_text, raw_image):
        # LLaVA-only setup: simple processor path and prompt templates.
        attention_mask = None
        # Default prompt for LLaVA
        if input_text is None:
            input_text = "Question: which city is this? Answer:"

        # OCR Integration
        try:
            # raw_image is PIL Image (RGB), convert to numpy
            img_np = np.array(raw_image)
            ocr_text = self.ocr_system.extract_text(img_np)
            if ocr_text:
                logger.info(f"OCR Text detected: {ocr_text}")
                input_text = f"The image contains the following text: {ocr_text}\n" + input_text
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        # print(f"OCR Text integrated into prompt: {input_text}")

        pre_prompt = "USER:\n"
        post_prompt = input_text + " ASSISTANT:"

        # Use AutoProcessor to prepare the image
        processor = AutoProcessor.from_pretrained(self.args.hf_model_dir)
        image = processor(text=pre_prompt + post_prompt,
                          images=raw_image,
                          return_tensors="pt")['pixel_values']

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * self.args.batch_size
        post_prompt = [post_prompt] * self.args.batch_size

        if image.dim() == 5:
            image = image.expand(self.args.batch_size, -1, -1, -1, -1).contiguous()
        else:
            image = image.expand(self.args.batch_size, -1, -1, -1).contiguous()
        image = image.to(self.device)

        # Generate decoder_input_ids for enc-dec models (if any)
        if self.decoder_llm:
            decoder_input_ids = None
        else:
            config = AutoConfig.from_pretrained(self.args.hf_model_dir)
            decoder_start_id = config.decoder_start_token_id
            decoder_input_ids = torch.IntTensor([[decoder_start_id]])
            decoder_input_ids = decoder_input_ids.repeat((self.args.batch_size, 1))

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids, attention_mask

    def run(self, input_text, input_image, max_new_tokens):
        input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids, attention_mask = self.setup_inputs(
            input_text, input_image)

        output_text = self.generate(pre_prompt,
                                    post_prompt,
                                    processed_image,
                                    decoder_input_ids,
                                    max_new_tokens,
                                    attention_mask=attention_mask,
                                    warmup=False)

        return input_text, output_text
