import argparse
import re
import os
from typing import List
from collections import defaultdict
from difflib import SequenceMatcher
from datasets import load_dataset
from tqdm import tqdm
from multimodal_model_runner import MultimodalModelRunner
from tensorrt_llm import logger
from PIL import Image

def normalize_ocr_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s

def build_prompt(dataset_name: str, question_type: str, question: str) -> str:
    prompt = (
        "You are an OCR assistant. "
        "Please read the text in the image and answer the question as accurately as possible.\n\n"
        # f"Dataset: {dataset_name}\n"
        f"Task type: {question_type}\n\n"
        f"Question: {question}\n\n"
        "Please answer with only the recognized text (no extra words)."
    )
    return prompt

def run_single(model: MultimodalModelRunner, prompt_text: str, image, max_new_tokens: int) -> str:
    input_text, output_text = model.run(prompt_text, image, max_new_tokens)
    return output_text[0][0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name", type=str, default="LIME-DATA/ocrbench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--llm_engine_dir", type=str, required=True)
    parser.add_argument("--visual_engine_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--ocr", action='store_true', help="Enable OCR integration")

    args = parser.parse_args()

    logger.info("========== Initializing MultimodalModelRunner ==========")

    class _TRTArgs:
        pass

    trt_args = _TRTArgs()
    trt_args.visual_engine_dir = args.visual_engine_dir
    trt_args.visual_engine_name = "model.engine"
    trt_args.llm_engine_dir = args.llm_engine_dir
    trt_args.llm_engine_name = "rank0.engine"
    trt_args.hf_model_dir = args.hf_model_dir
    trt_args.max_new_tokens = args.max_new_tokens
    trt_args.batch_size = 1
    trt_args.num_beams = args.num_beams
    trt_args.top_k = args.top_k
    trt_args.top_p = args.top_p
    trt_args.temperature = args.temperature
    trt_args.repetition_penalty = args.repetition_penalty
    trt_args.run_profiling = False
    trt_args.profiling_iterations = 1
    trt_args.check_accuracy = False
    trt_args.video_path = None
    trt_args.image_path = None
    trt_args.path_sep = ","
    trt_args.enable_context_fmha_fp32_acc = False
    trt_args.log_level = "error"
    trt_args.ocr = args.ocr # Pass OCR flag

    logger.set_level(trt_args.log_level)
    model = MultimodalModelRunner(trt_args)

    logger.info("========== Loading OCRBench Data ==========")
    # Load dataset from local path if it exists, otherwise from HF
    local_data_path = "/home/xsf/fl/edge/images/OCRBench/data"
    if os.path.exists(local_data_path):
        # Assuming the local path contains the dataset structure or images
        # If it's a HF dataset saved locally, load_dataset can load from disk
        try:
            ds = load_dataset(local_data_path, split=args.split)
        except:
             # Fallback: if it's just images, we might need a custom loader, 
             # but user said "dataset (Value)..." which implies a structured dataset.
             # Let's try loading from HF name but using cache dir or if user meant the folder IS the dataset
             ds = load_dataset(args.hf_dataset_name, split=args.split)
    else:
        ds = load_dataset(args.hf_dataset_name, split=args.split)
        
    logger.info(f"Original samples: {len(ds)}")

    if args.num_samples > 0 and len(ds) > args.num_samples:
        ds = ds.select(range(args.num_samples))
        logger.info(f"Using first {args.num_samples} samples")

    logger.info("========== Starting Evaluation ==========")

    total = 0
    correct_word = 0
    correct_char_count = 0
    total_char_count = 0
    
    # Stats per question type
    type_stats = defaultdict(lambda: {"total": 0, "correct_word": 0, "correct_char": 0, "total_char": 0})

    for ex in tqdm(ds, desc="Evaluating"):
        dataset_name = ex.get("dataset", "")
        question = ex["question"]
        question_type = ex.get("question_type", "")
        image = ex["image"]
        answers: List[str] = ex["answer"]

        prompt_text = build_prompt(dataset_name, question_type, question)

        try:
            model_answer = run_single(model, prompt_text, image, args.max_new_tokens)
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            continue

        total += 1
        type_stats[question_type]["total"] += 1

        norm_pred = normalize_ocr_text(model_answer)
        norm_gts = [normalize_ocr_text(a) for a in answers if a is not None]

        # Word-level accuracy (Exact Match)
        is_correct = any(norm_pred == g for g in norm_gts)
        if is_correct:
            correct_word += 1
            type_stats[question_type]["correct_word"] += 1

        # Character-level accuracy (Edit Distance based)
        if norm_gts:
            # Find best matching GT
            best_ratio = 0
            best_gt = ""
            for gt in norm_gts:
                ratio = SequenceMatcher(None, norm_pred, gt).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_gt = gt
            
            # Calculate correct chars roughly as ratio * total_len (approximation)
            # Or better: use Levenshtein distance to find errors.
            # Simple approach requested: "single letter reading accuracy".
            # Usually this means 1 - NED (Normalized Edit Distance) or similar.
            # Let's use SequenceMatcher ratio as a proxy for character accuracy per sample.
            # Ratio = 2*M / T, where M is matches, T is total length.
            # Let's accumulate raw lengths and matches if possible, or just average ratio.
            # User asked for "single letter reading accuracy", let's report Average Character Accuracy (1 - CER) or similar.
            # We will use the best ratio as the sample's char accuracy.
            
            char_acc = best_ratio
            correct_char_count += char_acc
            type_stats[question_type]["correct_char"] += char_acc
            
            # For total char count, we just count samples for averaging later
            total_char_count += 1
            type_stats[question_type]["total_char"] += 1

        if total <= 10:
            print("-" * 40)
            print(f"Type: {question_type}")
            print(f"GT: {answers}")
            print(f"Pred: {model_answer}")
            print(f"Norm Pred: {norm_pred}")

    if total == 0:
        print("No samples evaluated.")
        return

    word_acc = correct_word / total * 100.0
    avg_char_acc = (correct_char_count / total_char_count * 100.0) if total_char_count > 0 else 0.0

    print("\n====== Evaluation Results ======")
    print(f"Total Samples: {total}")
    print(f"Word Accuracy: {word_acc:.2f}%")
    print(f"Char Accuracy: {avg_char_acc:.2f}%")

    print("\nPer Question Type:")
    for qtype, st in type_stats.items():
        t = st["total"]
        if t == 0: continue
        w_acc = st["correct_word"] / t * 100.0
        c_acc = (st["correct_char"] / st["total_char"] * 100.0) if st["total_char"] > 0 else 0.0
        print(f"  - {qtype}: Word Acc={w_acc:.2f}%, Char Acc={c_acc:.2f}% ({t} samples)")

if __name__ == "__main__":
    main()
