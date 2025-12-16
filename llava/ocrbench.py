import argparse
import re
from typing import List

from collections import defaultdict
from difflib import SequenceMatcher

from datasets import load_dataset
from tqdm import tqdm

from tensorrt_llm.runtime import MultimodalModelRunner
from tensorrt_llm import logger


def normalize_ocr_text(s: str) -> str:
    """
    对 OCR 文本做一个简单归一化：
      - 转小写
      - 去掉首尾空格
      - 去掉所有非字母数字字符（包括空格、标点）
    这样 "Centre", "CENTRE ", "cen-tre" -> "centre"
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    # 保留字母数字（如果有中文，可以酌情再加 Unicode 范围）
    s = re.sub(r"[^0-9a-z]+", "", s)
    return s


def build_prompt(dataset_name: str, question_type: str, question: str) -> str:
    """
    构造喂给 LLaVA 的文本 prompt。
    dataset/question_type 可以作为一点上下文信息给模型。
    """
    # 你可以根据需要改成中文提示；这里只给一个通用英文版本
    prompt = (
        "You are an OCR assistant. "
        "Please read the text in the image and answer the question as accurately as possible.\n\n"
        f"Dataset: {dataset_name}\n"
        f"Task type: {question_type}\n\n"
        f"Question: {question}\n\n"
        "Please answer with only the recognized text (no extra words)."
    )
    return prompt


def run_single(
    model: MultimodalModelRunner,
    prompt_text: str,
    image,
    max_new_tokens: int,
) -> str:
    """
    对单条 (prompt + image) 调用 LLaVA (TensorRT-LLM) 推理，返回字符串输出。

    假设 MultimodalModelRunner.run 接口与官方 examples/multimodal/run.py 一致：
        input_text, output_text = model.run(prompt, image, max_new_tokens)
    其中 output_text 是形如 [[generated_str]] 的结构。
    """
    input_text, output_text = model.run(prompt_text, image, max_new_tokens)
    return output_text[0][0]


def main():
    parser = argparse.ArgumentParser()

    # ===== OCRBench 数据集相关 =====
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="LIME-DATA/ocrbench",
        help="HuggingFace 上 OCRBench 数据集名称，默认 LIME-DATA/ocrbench，"
             "如果你用的是别的（比如 Nayana-OCRBench-in-1k-v2-arxiv），在这里改。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="使用哪个 split（比如 train / test 等，视数据集而定）。",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="最多评测多少条样本（0 或负数表示用全部）。",
    )

    # ===== LLaVA / TensorRT-LLM 引擎相关 =====
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="LLaVA HF 权重路径（只用于 tokenizer 等）。",
    )
    parser.add_argument(
        "--llm_engine_dir",
        type=str,
        required=True,
        help="TRT-LLM LLM 引擎目录。",
    )
    parser.add_argument(
        "--visual_engine_dir",
        type=str,
        required=True,
        help="TRT-LLM vision 引擎目录。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="每个样本最大生成 token 数。",
    )

    # ===== 解码相关参数 =====
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.0)
    # 直接用 1.0，避免 temperature=0 被 TRT-LLM 每次改写并打 warning
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()

    logger.info("========== 加载 OCRBench 数据 ==========")
    ds = load_dataset(args.hf_dataset_name, split=args.split)
    logger.info(f"原始样本数: {len(ds)}")

    if args.num_samples > 0 and len(ds) > args.num_samples:
        ds = ds.select(range(args.num_samples))
        logger.info(f"仅使用前 {args.num_samples} 条样本进行评测")

    logger.info("========== 初始化 MultimodalModelRunner ==========")

    class _TRTArgs:
        pass

    trt_args = _TRTArgs()
    trt_args.visual_engine_dir = args.visual_engine_dir
    trt_args.visual_engine_name = "model.engine"
    trt_args.llm_engine_dir = args.llm_engine_dir
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
    trt_args.log_level = "error"   # 关掉 warning，避免刷屏

    logger.set_level(trt_args.log_level)
    model = MultimodalModelRunner(trt_args)

    logger.info("========== 开始评测 OCRBench ==========")

    total = 0
    correct = 0
    sim_sum = 0.0

    # 按 question_type 做一个细分统计
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "sim_sum": 0.0})

    for ex in tqdm(ds, desc="Evaluating OCRBench"):
        dataset_name = ex.get("dataset", "")
        question = ex["question"]
        question_type = ex.get("question_type", "")
        image = ex["image"]
        answers: List[str] = ex["answer"]  # list[str]

        prompt_text = build_prompt(dataset_name, question_type, question)

        try:
            model_answer = run_single(model, prompt_text, image, args.max_new_tokens)
        except Exception as e:
            logger.warning(f"推理失败，跳过此样本，err={e}")
            continue

        total += 1
        type_stats[question_type]["total"] += 1

        # 归一化预测与所有 GT
        norm_pred = normalize_ocr_text(model_answer)
        norm_gts = [normalize_ocr_text(a) for a in answers if a is not None]

        # exact match：任意一个 GT 匹配就算对
        is_correct = any(norm_pred == g for g in norm_gts)
        if is_correct:
            correct += 1
            type_stats[question_type]["correct"] += 1

        # 字符级相似度：和所有 GT 取最大值
        if norm_gts:
            sims = [
                SequenceMatcher(None, norm_pred, g).ratio()
                for g in norm_gts
            ]
            best_sim = max(sims)
        else:
            best_sim = 0.0

        sim_sum += best_sim
        type_stats[question_type]["sim_sum"] += best_sim

        #如果想看前几条输出，可以解注释：
        if total <= 5:
            print("=" * 80)
            print("Dataset:", dataset_name)
            print("Type:", question_type)
            print("Q:", question)
            print("GT:", answers)
            print("Model:", model_answer)
            print("Norm pred:", norm_pred)
            print("Norm gts:", norm_gts)
            print("Best sim:", best_sim)

    logger.info("========== 评测结束 ==========")

    if total == 0:
        logger.info("有效样本数为 0，无法计算指标。")
        return

    acc = correct / total * 100.0
    avg_sim = sim_sum / total * 100.0

    print("====== OCRBench Evaluation Result ======")
    print(f"Total samples   : {total}")
    print(f"Exact match     : {acc:.2f}%  (correct={correct})")
    print(f"Avg char-sim (%) : {avg_sim:.2f}%")

    # 按 question_type 打印细分结果
    print("\nPer question_type:")
    for qtype, st in type_stats.items():
        t = st["total"]
        if t == 0:
            continue
        c = st["correct"]
        s = st["sim_sum"]
        acc_t = c / t * 100.0
        sim_t = s / t * 100.0
        print(f"  - {qtype}: "
              f"acc={acc_t:.2f}% (correct={c}/{t}), "
              f"avg_sim={sim_t:.2f}%")

if __name__ == "__main__":
    main()
