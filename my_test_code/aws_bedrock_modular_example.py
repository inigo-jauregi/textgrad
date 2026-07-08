import os
import textgrad as tg


def main():

    # Verify AWS Profile is set
    print(f"Using AWS Profile: {os.environ.get('AWS_PROFILE', 'Not set')}")
    # Setup - use your Bedrock models
    model_string = "experimental:bedrock/anthropic.claude-3-sonnet-20240229-v1:0"

    # Initialize engine (Is this the backend for the loss??)
    llm_engine = tg.get_engine(model_string,
                               cache=False, # Optional: disable cachin
                               )
    # Set as backward engine
    tg.set_backward_engine(model_string,
                           cache=False, # Optional: disable cachin
                           override=True)

    print("=" * 80)
    print("Two-Stage Translation System with TextGrad")
    print("=" * 80)

    # ============================================================================
    # Define Constraint Extraction Model
    # ============================================================================
    # Define a system prompt to optimize
    constraint_system_prompt = tg.Variable(
        "You work on movie subtitles translation. "
        "Given the <METADATA>, the <SOURCE_SENTENCE> and the <TARGET_LANGUAGE> "
        "you need to identify the main lexical and idiomatic context such as style, formality-level, gender pronouns, idioms, etc."
        " that need to be taken into account when translating the sentence from the movie. ",
        requires_grad=True,
        role_description="system prompt for context extraction model"
    )

    constraint_model = tg.BlackboxLLM(
        llm_engine,
        system_prompt=constraint_system_prompt
    )

    # ============================================================================
    # STAGE 2: Translation Model
    # ============================================================================
    translation_system_prompt = tg.Variable(
        """You are a professional translator for movie subtitles.""",
        requires_grad=True,
        role_description="system prompt for translation model"
    )

    translation_model = tg.BlackboxLLM(
        llm_engine,
        system_prompt=translation_system_prompt
    )

    # ============================================================================
    # Example Translation Task
    # ============================================================================
    input_context_and_sentence = tg.Variable(
        """
        <TARGET_LANGUAGE>Spanish</TARGET_LANGUAGE>
        
        <METADATA> 
            - Country of production: Belgium, Luxembourg, France
            - Genre: Action, Comedy, Crime
            - Plot: Jean-Claude Van Damme gets involved in a bank robbery with hostages situation and reflects about his life during it.
            - Rated: PG rating R
            - Writers: Written by: Mabrouk El Mechri, Frédéric Benudis, Frédéric Taddeï
            - Year: Released in 2008
        </METADATA>
        
        <SOURCE_SENTENCE>Unless... the path you've set for me is full of hurdles where the answer comes before the question.</SOURCE_SENTENCE>
        """,
        requires_grad=False,
        role_description="source sentence to translate"
    )

    # ============================================================================
    # FORWARD PASS: Chain the two models
    # ============================================================================
    print("\n" + "=" * 80)
    print("INITIAL SYSTEM (Before Optimization)")
    print("=" * 80)

    context_detection_input = tg.Variable(
        f"""
                {input_context_and_sentence.value}

                "Provide all relevant context as a list of items between <CONTEXT> </CONTEXT> tags so that the translator can use them later."
            """,
        requires_grad=False,
        role_description="input to translation model"
    )


    constraints = constraint_model(context_detection_input)

    # Post-process constraints (only extract text between <CONSTRAINTS> </CONSTRAINTS>), if no tags, a text, NO CONSTRAINTS IDENTIFIED
    # First check the tags are present
    if "<CONTEXT>" not in constraints.value or "</CONTEXT>" not in constraints.value:
        prcsd_constraints = "NO CONTEXT IDENTIFIED"
        print(f"WARNING: No constraints identified in constraints: {constraints.value}")
    else:
        prcsd_constraints = constraints.value.split("<CONTEXT>")[1].split("</CONTEXT>")[0]
        print(f"CONTEXT identified: {prcsd_constraints}")

    # Stage 2: Translate with constraints
    translation_input = tg.Variable(
        f"""
            {input_context_and_sentence.value}
            
            <CONTEXT>
            {prcsd_constraints}
            </CONTEXT>
            
            "Provide the translation between <TRANSLATION> </TRANSLATION> tags so that the translator can use them later."
        """,
        requires_grad=False,
        role_description="input to translation model"
    )

    translation = translation_model(translation_input)
    print(f"[Stage 2] Translation:")
    print(f"{translation.value}\n")

    # ============================================================================
    # EVALUATION & OPTIMIZATION
    # ============================================================================
    # Define what makes a good translation
    evaluation_instruction = tg.Variable(
        """Evaluate this translation considering:
        1. Accuracy: Does it preserve the original meaning?
        2. Context appropriateness: Are idioms, cultural references, gender pronouns and tone properly handled?
        3. Naturalness: Does it sound natural in the target language?

        Be critical and specific about what's wrong or missing. Respond in English.""",
        requires_grad=False,
        role_description="evaluation criteria"
    )

    # Create loss function
    loss_fn = tg.TextLoss(evaluation_instruction.value)

    # Setup optimizer for BOTH prompts
    optimizer = tg.TGD(
        parameters=[constraint_system_prompt, translation_system_prompt]
    )

    print("=" * 80)
    print("OPTIMIZING SYSTEM...")
    print("=" * 80)

    # Optimization loop
    num_iterations = 2
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        context_detection_input = tg.Variable(
            f"""
                        {input_context_and_sentence.value}

                        "Provide all relevant context as a list of items between <CONTEXT> </CONTEXT> tags so that the translator can use them later."
                    """,
            requires_grad=False,
            role_description="input to translation model"
        )

        # Forward pass (recalculate with current prompts)
        constraints = constraint_model(context_detection_input)

        # Post-process constraints (only extract text between <CONSTRAINTS> </CONSTRAINTS>), if no tags, a text, NO CONSTRAINTS IDENTIFIED
        # First check the tags are present
        if "<CONTEXT>" not in constraints.value or "</CONTEXT>" not in constraints.value:
            prcsd_constraints = "NO CONTEXT IDENTIFIED"
            print(f"WARNING: No constraints identified in constraints: {constraints.value}")
        else:
            prcsd_constraints = constraints.value.split("<CONTEXT>")[1].split("</CONTEXT>")[0]
            print(f"CONTEXT identified: {prcsd_constraints}")

        # Update translation input with new constraints
        translation_input = tg.Variable(
            f"""
                    {input_context_and_sentence.value}
            
                    <CONTEXT>
                    {prcsd_constraints}
                    </CONTEXT>
                    
                    Provide the translation between <TRANSLATION> </TRANSLATION> tags so that the translator can use them later.
                """,
            requires_grad=False,
            role_description="input to translation model"
        )

        translation = translation_model(translation_input)

        # Compute loss
        loss = loss_fn(translation)
        print("*******************************")
        print(f"Loss feedback: {loss.value}...")
        print("*******************************")

        # Backward pass - computes gradients for both prompts
        loss.backward()

        # Update both prompts
        optimizer.step()
        optimizer.zero_grad()

    # ============================================================================
    # TEST OPTIMIZED SYSTEM
    # ============================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZED SYSTEM (After Optimization)")
    print("=" * 80)

    print("\n[Optimized] Constraint Model Prompt:")
    print(constraint_system_prompt.value)
    print("\n" + "-" * 80)

    print("\n[Optimized] Translation Model Prompt:")
    print(translation_system_prompt.value)
    print("\n" + "-" * 80)

    # Run final translation
    context_detection_input = tg.Variable(
        f"""
                            {input_context_and_sentence.value}

                            "Provide all relevant context as a list of items between <CONTEXT> </CONTEXT> tags so that the translator can use them later."
                        """,
        requires_grad=False,
        role_description="input to translation model"
    )

    # Forward pass (recalculate with current prompts)
    constraints = constraint_model(context_detection_input)

    # Post-process constraints (only extract text between <CONSTRAINTS> </CONSTRAINTS>), if no tags, a text, NO CONSTRAINTS IDENTIFIED
    # First check the tags are present
    if "<CONTEXT>" not in constraints.value or "</CONTEXT>" not in constraints.value:
        prcsd_constraints = "NO CONTEXT IDENTIFIED"
        print(f"WARNING: No constraints identified in constraints: {constraints.value}")
    else:
        prcsd_constraints = constraints.value.split("<CONTEXT>")[1].split("</CONTEXT>")[0]
        print(f"CONTEXT identified: {prcsd_constraints}")

    print(f"\n[Stage 1] Final Extracted Constraints:")
    print(f"{prcsd_constraints}\n")

    translation_input_final = tg.Variable(
        f"""
                    {input_context_and_sentence.value}
            
                    <CONTEXT>
                    {prcsd_constraints}
                    </CONTEXT>
                    
                    "Provide the translation between <TRANSLATION> </TRANSLATION> tags so that the translator can use them later."
                    """,
        requires_grad=False,
        role_description="input to translation model"
    )

    translation_final = translation_model(translation_input_final)
    print(f"[Stage 2] Final Translation:")
    print(f"{translation_final.value}\n")


if __name__ == "__main__":
    main()
