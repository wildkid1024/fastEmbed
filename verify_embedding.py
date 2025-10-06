from transformers import BertModel, BertTokenizer, XLMRobertaModel, XLMRobertaTokenizer
import torch
from sentence_transformers import SentenceTransformer

def main():
    """
    model_path = "/home/wildkid1024/Public/Models/bge-small-zh-v1.5"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(
        model_path,
        output_hidden_states=True,
        output_attentions=True,
        torchscript=False
    )
    model.eval()
    """
    model_path = "/home/wildkid1024/Models/BAAI/bge-m3/"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaModel.from_pretrained(
        model_path,
        output_hidden_states=True,
        output_attentions=True,
        torchscript=False
    )
    model.eval()
    print(model)

    # ==== 1. 输入文本（与 C++ 保持一致，避免 [UNK]）====
    text = "This is a test sentence."  # 使用中文确保分词一致
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True  # 自动添加 [CLS] (101) 和 [SEP] (102)
    )
    input_ids = inputs["input_ids"]
    print(f"Python Token IDs: {input_ids[0].tolist()}")  # 确认与 C++ 分词结果一致

    mask = inputs["attention_mask"]
    inputs.pop("attention_mask")
    # ==== 2. 获取嵌入层输出（MHA 的输入）====
    with torch.no_grad():
        embedding_output = model.embeddings(**inputs)  # 直接调用嵌入层，跳过 encoder

    print("===== Python 嵌入层输出 =====")
    print(f"形状: {embedding_output.shape}")  # 应为 (batch_size, seq_len, hidden_size)，如 (1, 5, 384)
    print(" ".join(f"{x:.6f}" for x in embedding_output.squeeze().cpu().numpy().flatten()[:10] ))  # 展平为一维向量，保留6位小数

    # ==== 3. 提取第一层 MHA 原始输出（未经过输出 Dense 层）====
    first_encoder_layer = model.encoder.layer[0]  # 第一层编码器
    mha_module = first_encoder_layer.attention.self  # MHA 核心模块

    # 调用 MHA 前向传播（输入为嵌入层输出，对应 C++ 的 embedded 变量）
    mha_outputs = mha_module(
        hidden_states=embedding_output,
        attention_mask=mask,  # 注意力掩码
        output_attentions=True
    )
    # mha_outputs[0] 即为多头注意力拼接后的原始输出（形状：[batch_size, seq_len, embedding_dim]）
    python_mha_output = mha_outputs[0].detach().squeeze().cpu().numpy()  # 移除 batch 维度并转为 numpy

    # ==== 4. 按 C++ 格式打印 MHA 输出 ====
    print("\n===== Python 第一层 MHA 原始输出 =====")
    print(f"形状: {python_mha_output.shape}")  # 应为 (seq_len, embedding_dim)，如 (5, 384)
    print(" ".join(f"{x:.6f}" for x in python_mha_output.flatten()[:10]))  # 展平为一维向量，保留6位小数


    with torch.no_grad():
        model.eval()
        output = model(**inputs)

    print("\n===== Python 模型输出 =====")
    print(f"形状: {output.pooler_output.shape}")  # 应为 (batch_size, seq_len, hidden_size)，如 (1, 5, 384)
    print(output.pooler_output)  # 展平为一维向量，保留6位小数

def test_sentence_transformers():
    from sentence_transformers.models.Pooling import Pooling
    from sentence_transformers.models.Normalize import Normalize
    model_path = "/home/wildkid1024/Models/BAAI/bge-m3/"
    model = SentenceTransformer(model_path)
    print(model)
    texts = ["This is a test sentence.","Another example sentence."]  # 使用中文确保分词一致
    embedding = model.encode(texts)
    print(embedding)

def trans_torch_to_safetensors():
    from safetensors.torch import save_file
    import torch

    # model_path = '/home/wildkid1024/Public/Models/bge-large-zh-v1.5'
    # model_path = '/home/wildkid1024/Models/models-hf/sentence_models/bge-large-zh-v1.5'
    model_path = '/home/wildkid1024/Models/BAAI/bge-m3/'
    state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
    save_file(state_dict, f"{model_path}/model.safetensors")
    print("Safetensors 文件已保存")

if __name__ == "__main__":
    # main()
    test_sentence_transformers()
    # trans_torch_to_safetensors()
