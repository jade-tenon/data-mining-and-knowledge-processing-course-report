import streamlit as st

import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """从本地目录或远程加载生成模型。若 model_name 指向本地目录，则强制离线加载。"""
    st.write(f"Loading generation model: {model_name} ...")
    try:
        # 判定是否为本地模型目录或压缩包
        is_local = os.path.exists(model_name) and (
            os.path.isdir(model_name) or model_name.endswith((".tar.gz", ".zip"))
        )
        local_files_only = True if is_local else False
        st.write(f"local_files_only={local_files_only}")

        # 选择设备与精度
        has_cuda = torch.cuda.is_available()
        device_map = "auto" if has_cuda else {"": "cpu"}
        torch_dtype = torch.float16 if has_cuda else torch.float32

        # 加载 tokenizer（离线模式如果为本地）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only
        )

        # 加载模型（离线模式如果为本地）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            device_map=device_map,
            torch_dtype=torch_dtype
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        st.success("Generation model and tokenizer loaded.")
        return model, tokenizer

    except Exception as e:
        st.error(f"Failed to load generation model: {e}")
        return None, None