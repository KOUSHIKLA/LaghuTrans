# -*- coding: utf-8 -*-
"""
LaghuTrans Inference Core
English to Hindi Neural Machine Translation
"""

import os
import codecs
import torch
from argparse import ArgumentParser
from collections import OrderedDict
from onmt.translate.translator import build_translator
from onmt.opts import translate_opts
from subword_nmt.apply_bpe import BPE

# ===================== CONFIGURATION =====================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default paths (users should update these or use environment variables)
BPE_CODES_PATH = os.environ.get(
    "LAGHUTRANS_BPE_CODES",
    os.path.join(SCRIPT_DIR, "models", "train.codes")
)

MODEL_PATH = os.environ.get(
    "LAGHUTRANS_MODEL",
    os.path.join(SCRIPT_DIR, "models", "model_finetuned_step_120000.pt")
)

# ===================== GLOBAL STORAGE =====================
ml_models = {}

# ===================== LOAD MODELS =====================
def load_models():
    """
    Load the BPE processor and OpenNMT translator model.
    Models are loaded only once and cached in ml_models dict.
    
    Raises:
        FileNotFoundError: If model files are not found at specified paths
    """
    # Prevent re-loading
    if ml_models:
        return
    
    # Check if files exist
    if not os.path.exists(BPE_CODES_PATH):
        raise FileNotFoundError(
            f"BPE codes file not found at: {BPE_CODES_PATH}\n"
            f"Please download the model files or set LAGHUTRANS_BPE_CODES environment variable."
        )
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            f"Please download the model files or set LAGHUTRANS_MODEL environment variable."
        )
    
    print(f"Loading BPE codes from: {BPE_CODES_PATH}")
    with codecs.open(BPE_CODES_PATH, "r", "utf-8") as f:
        ml_models["bpe_processor"] = BPE(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # -------- Build OpenNMT translate options --------
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
    """
    Translate English text to Hindi.
    
    Args:
        text (str): English text to translate
        
    Returns:
        str: Translated Hindi text
        
    Example:
        >>> hindi_output = process_pipeline("The farmer planted rice.")
        >>> print(hindi_output)
    """
    load_models()
    
    # Apply BPE tokenization
    tokenized = ml_models["bpe_processor"].process_line(text)
    
    # Translate
    _, output = ml_models["translator"].translate(
        src=[tokenized],
        batch_size=1
    )
    
    # Remove BPE markers and return
    return output[0][0].replace("@@ ", "")


# ===================== UTILITY FUNCTIONS =====================
def set_model_paths(bpe_codes_path: str, model_path: str):
    """
    Manually set model paths (useful for custom configurations).
    Must be called before load_models().
    
    Args:
        bpe_codes_path (str): Path to BPE codes file
        model_path (str): Path to trained model checkpoint
    """
    global BPE_CODES_PATH, MODEL_PATH
    BPE_CODES_PATH = bpe_codes_path
    MODEL_PATH = model_path


if __name__ == "__main__":
    # Test the pipeline
    test_text = "The farmer planted rice in the paddy fields."
    print(f"Input: {test_text}")
    print(f"Output: {process_pipeline(test_text)}")
