import codecs
import torch
from argparse import ArgumentParser
from collections import OrderedDict

from onmt.translate.translator import build_translator
from onmt.opts import translate_opts
from subword_nmt.apply_bpe import BPE


# ===================== PATHS =====================
BPE_CODES_PATH = "/home/aics/yogi/MT/multi_16k_1225/OpenNMT-py/newdata/train.codes"
MODEL_PATH = "/home/aics/yogi/MT/multi_16k_1225/OpenNMT-py/en_hi_model/train1/model_finetuned_step_120000.pt"

# ===================== GLOBAL STORAGE =====================
ml_models = {}


# ===================== LOAD MODELS =====================
def load_models():
    # Prevent re-loading
    if ml_models:
        return

    print(f"Loading BPE codes from: {BPE_CODES_PATH}")
    with codecs.open(BPE_CODES_PATH, "r", "utf-8") as f:
        ml_models["bpe_processor"] = BPE(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------- Build OpenNMT translate options (CORRECT WAY) --------
    parser = ArgumentParser()
    translate_opts(parser)
    opt = parser.parse_args(
    args=["--model", MODEL_PATH, "--src", "dummy"]
    )


    # -------- Override required options --------
    opt.models = [MODEL_PATH]
    opt.data_type = "text"
    opt.src = ""
    opt.output = "/dev/null"

    opt.gpu = 0 if device == "cuda" else -1
    opt.fp32 = False

    opt.beam_size = 5
    opt.n_best = 1
    opt.max_length = 100
    opt.replace_unk = False

    opt.alpha = 0.6
    opt.beta = 0.0
    opt.length_penalty = "wu"
    opt.coverage_penalty = "none"

    # -------- Load translator --------
    ml_models["translator"] = build_translator(opt, report_score=False)
    print("✅ Models loaded successfully")


# ===================== INFERENCE PIPELINE =====================
def process_pipeline(text: str) -> str:
    load_models()

    tokenized = ml_models["bpe_processor"].process_line(text)

    _, output = ml_models["translator"].translate(
        src=[tokenized],
        batch_size=1
    )

    return output[0][0].replace("@@ ", "")
