import os

from fastapi import FastAPI
from vllm import LLM, SamplingParams

app = FastAPI()
model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")

# Инициализируем движок vLLM (одна строчка – за это vllm любят):
llm = LLM(
    model=model_id,
    dtype="bfloat16",  # float16, если GPU не поддерживает bfloat16
    trust_remote_code=True,
)


def build_prompt(text: str, labels) -> str:
    labels_str = ", ".join(labels)
    return (
        "Присвой тексту одну из категорий ниже.\n"
        f"Возможные категории: {labels_str}\n\n"
        f"Текст:\n{text}\n\n"
        "Ответ (только категория):"
    )


sampler = SamplingParams(
    temperature=0.0,  # делаем вывод максимально детерминированным
    top_p=1.0,
    max_tokens=3,
    stop=["\n"],  # модель останавливается после первой строки ответа
)


@app.post("/classify")
def classify(payload: dict):
    text = payload["text"]
    labels = payload["labels"]  # список меток для классификации
    # Формируем подсказку (prompt) для LLM, чтобы он выбрал метку
    prompt = build_prompt(text, labels)
    out = llm.generate([prompt], sampling_params=sampler)[0]
    return out.outputs[0].text.strip()
