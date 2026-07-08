"""
Minimal end-to-end test of the local HuggingFace engine (Qwen3-8B) with the
REVOLVE (v2) optimizer.

Both roles use the SAME local Qwen3-8B model, loaded ONCE and shared:
  * the "task" model whose system prompt we are optimizing (forward pass), and
  * the optimization / backward engine that computes textual gradients and
    proposes the improved system prompt (REVOLVE, optimizer_v2).

Requirements
------------
    pip install "transformers>=4.51.0" torch accelerate
    # ~16GB VRAM for Qwen3-8B in bf16; a CUDA GPU is strongly recommended.

Run
---
    python my_test_code/test_qwen3_revolve.py
"""

import textgrad as tg
from textgrad.engine.huggingface import ChatHuggingFace
from textgrad import TextualGradientDescent_v2 as TGD_v2  # REVOLVE optimizer

MODEL_STRING = "Qwen/Qwen3-8B"


def build_engine():
    """Load Qwen3-8B once and reuse the same instance everywhere.

    enable_thinking=False keeps outputs short, clean and easy for the optimizer
    to parse (no <think>...</think> blocks). Set it to True if you want the model
    to reason before answering — the reasoning is stripped from the returned text.
    """
    print(f"Loading {MODEL_STRING} (this may take a while the first time)...")
    engine = ChatHuggingFace(
        model_string=MODEL_STRING,
        enable_thinking=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_new_tokens=1024,
    )
    print("Model loaded.\n")
    return engine


def main():
    # --- one shared engine for both the task model and the optimizer ---------
    engine = build_engine()
    tg.set_backward_engine(engine, override=True)

    # --- system prompt we want to improve ------------------------------------
    system_prompt = tg.Variable(
        "You are a helpful assistant. Answer the question.",
        requires_grad=True,
        role_description="system prompt for a math question-answering assistant",
    )

    # the task model uses the SAME local Qwen3 engine
    model = tg.BlackboxLLM(engine, system_prompt=system_prompt)

    # REVOLVE optimizer, also driven by the SAME engine
    optimizer = TGD_v2(engine=engine, parameters=[system_prompt])

    # a small "training set" of math word problems ----------------------------
    training_examples = [
        ("Natalia sold clips to 48 friends in April, and then she sold half as "
         "many clips in May. How many clips did she sell altogether?", "72"),
        ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
         "How many bolts in total does it take?", "3"),
        ("If a train travels 60 km in 1.5 hours, what is its average speed in "
         "km/h?", "40"),
    ]

    # loss: an evaluator (also Qwen3) critiques each answer -> textual gradient
    evaluation_instruction = (
        "Below is a math word problem, the correct final answer, and an answer "
        "produced by an assistant. Judge whether the assistant's final answer is "
        "correct and whether the reasoning is clear and concise. Point out any "
        "mistakes and give concrete, actionable feedback on how the assistant's "
        "system prompt could be changed so that future answers are more accurate "
        "and better explained. Be critical and specific."
    )

    print("=" * 80)
    print("INITIAL SYSTEM PROMPT:")
    print(system_prompt.value)
    print("=" * 80)

    NUM_EPOCHS = 2
    for epoch in range(NUM_EPOCHS):
        print(f"\n############## EPOCH {epoch + 1}/{NUM_EPOCHS} ##############")
        for i, (question_text, gold) in enumerate(training_examples):
            question = tg.Variable(
                question_text,
                requires_grad=False,
                role_description="a math word problem posed to the assistant",
            )

            # forward pass: task model answers with the current system prompt
            answer = model(question)
            print(f"\n[Example {i + 1}] Q: {question_text}")
            print(f"  Gold answer: {gold}")
            print(f"  Model answer: {answer.value}")

            # build the loss variable (evaluator feedback on this answer)
            loss_input = tg.Variable(
                f"{evaluation_instruction}\n\n"
                f"Problem: {question_text}\n"
                f"Correct final answer: {gold}\n"
                f"Assistant's answer: {answer.value}",
                requires_grad=False,
                role_description="evaluation instruction for the assistant's answer",
            )
            loss_fn = tg.TextLoss(loss_input, engine=engine)
            loss = loss_fn(answer)

            # backward + optimizer step (REVOLVE updates the system prompt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("  --> Updated system prompt:")
            print(f"      {system_prompt.value}")

    print("\n" + "=" * 80)
    print("FINAL OPTIMIZED SYSTEM PROMPT:")
    print(system_prompt.value)
    print("=" * 80)


if __name__ == "__main__":
    main()