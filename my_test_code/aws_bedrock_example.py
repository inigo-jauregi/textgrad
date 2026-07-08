import os
import textgrad as tg


def main():
    # Verify AWS Profile is set
    print(f"Using AWS Profile: {os.environ.get('AWS_PROFILE', 'Not set')}")

    llm_engine = tg.get_engine(
        "experimental:bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        cache=False  # Optional: disable caching
    )

    # Set as backward engine
    tg.set_backward_engine(
        "experimental:bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        cache=False,
        override=True
    )

    # Define a system prompt to optimize
    system_prompt = tg.Variable(
        "You are a helpful translator of movie subtitles. "
        "You translate from English to Spanish taking into account the context in the movie.",
        requires_grad=True,
        role_description="system prompt to guide the model"
    )

    print("\nInitial system prompt:", system_prompt.value)

    # Create a model with the prompt
    model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)

    # Define your question
    question = tg.Variable(
        "Frankly, my dear, I don't give a damn",
        requires_grad=False,
        role_description="user question"
    )

    # Get initial response
    response = model(question)

    print("\nSource (English):", question.value)
    print("\nTarget (Spanish):", response.value)

    # Set up optimization
    optimizer = tg.TGD(parameters=[system_prompt])

    # Define evaluation criteria
    loss_fn = tg.TextLoss(
        "Evaluate if the response is clear, accurate, and well-structured. Be critical."
    )

    # Optimize
    loss = loss_fn(response)

    print("\nLoss (feedback):", loss)

    loss.backward()
    optimizer.step()

    print("New Optimized system prompt:", system_prompt.value)

    # Get final response
    final_response = model(question)
    print("confirming new system prompt:")
    print(model.system_prompt)
    print("confirmed")
    print("\nTarget (Spanish) (new optimized system prompt):", final_response)


if __name__ == "__main__":
    main()
