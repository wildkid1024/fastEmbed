import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
# from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3Tokenizer


model_path = "/public/Models/Qwen/Qwen3-Embedding-0___6B"

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def test_with_transformers():
    # Each query must come with a one-sentence instruction that describes the task
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]

    input_texts = queries

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    # model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda()

    model.eval()
    print(model)
    
    print("=> original sentence")
    print(input_texts)

    max_length = 8192
    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    print("=> batch_dict")
    print(batch_dict)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    print(embeddings)

def test_with_sentences_transformer():
    # Requires transformers>=4.51.0
    # Requires sentence-transformers>=2.7.0

    from sentence_transformers import SentenceTransformer

    # Load the model
    model = SentenceTransformer(model_path)

    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # together with setting `padding_side` to "left":
    # model = SentenceTransformer(
    #     "Qwen/Qwen3-Embedding-0.6B",
    #     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
    #     tokenizer_kwargs={"padding_side": "left"},
    # )

    # The queries and documents to embed
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    # Encode the queries and documents. Note that queries benefit from using a prompt
    # Here we use the prompt called "query" stored under `model.prompts`, but you can
    # also pass your own prompt via the `prompt` argument
    query_embeddings = model.encode(queries, prompt_name="query")
    print(query_embeddings)


if __name__ == "__main__":
    test_with_transformers()
    # test_with_sentences_transformer()