"""
Prompt template construction functions for building modular prompts.
"""

from typing import Union, List, Optional, Dict, Any


def lowercase_first_char(text: str) -> str:
    """Lowercases the first character of a string.

    Args:
        text: Input string.

    Returns:
        The input string with the first character lowercased.
    """
    return text[0].lower() + text[1:] if text else text


def format_prompt_section(lead_in: str, value: Union[str, List[str]]) -> str:
    """Formats a prompt section by joining a lead-in with content.

    Args:
        lead_in: Introduction sentence for the section.
        value: Section content, as a string or list of strings.

    Returns:
        A formatted string with the lead-in followed by the content.
    """
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"


def build_prompt_from_config(
    config: Dict[str, Any],
    input_data: str = "",
    app_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a complete prompt string for Ethiopian History for Kids chatbot, with strict formatting and naming."""
    prompt_parts = []

    # Persona/role
    if role := config.get("role"):
        prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    # Main instruction
    instruction = config.get("instruction")
    if not instruction:
        raise ValueError("Missing required field: 'instruction'")
    prompt_parts.append(format_prompt_section("Your task is as follows:", instruction))

    # Context/background
    if context := config.get("context"):
        prompt_parts.append(f"Background information that may help you:\n{context}")

    # Output constraints (rules)
    if constraints := config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                "Strictly follow these formatting and content rules:", constraints
            )
        )

    # Style/tone
    if tone := config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Use the following style and tone:", tone
            )
        )

    # Output format
    if format_ := config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Format your response as follows:", format_)
        )

    # Examples
    if examples := config.get("examples"):
        prompt_parts.append("Here are some examples to guide your response:")
        if isinstance(examples, list):
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:\n{example}")
        else:
            prompt_parts.append(str(examples))

    # Goal
    if goal := config.get("goal"):
        prompt_parts.append(f"Your goal is:\n{goal}")

    # Input data (the retrieved document content)
    if input_data:
        prompt_parts.append(
            "Here is the content you need to work with:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    # Reasoning strategy
    reasoning_strategy = config.get("reasoning_strategy")
    if reasoning_strategy and reasoning_strategy != "None" and app_config:
        strategies = app_config.get("reasoning_strategies", {})
        if strategy_text := strategies.get(reasoning_strategy):
            prompt_parts.append(strategy_text.strip())

    prompt_parts.append("Now perform the task as instructed above.")
    return "\n\n".join(prompt_parts)


def print_prompt_preview(prompt: str, max_length: int = 500) -> None:
    """Prints a preview of the constructed prompt for debugging purposes.

    Args:
        prompt: The constructed prompt string.
        max_length: Maximum number of characters to show.
    """
    print("=" * 60)
    print("CONSTRUCTED PROMPT:")
    print("=" * 60)
    if len(prompt) > max_length:
        print(prompt[:max_length] + "...")
        print(f"\n[Truncated - Full prompt is {len(prompt)} characters]")
    else:
        print(prompt)
    print("=" * 60)


def build_system_prompt_from_config(
    config: Dict[str, Any],
    background_content: str = "",
) -> str:
    """Builds a system prompt string for Ethiopian History for Kids chatbot, with strict formatting and naming."""
    prompt_parts = []

    # Role is required for system prompts
    role = config.get("role")
    if not role:
        raise ValueError("Missing required field: 'role'")
    prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    # Add behavioral constraints
    if constraints := config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                "Strictly follow these important guidelines:", constraints
            )
        )

    # Add style and tone guidelines
    if tone := config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Communication style:", tone
            )
        )

    # Add output format requirements
    if format_ := config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Response formatting:", format_)
        )

    # Add goal if specified
    if goal := config.get("goal"):
        prompt_parts.append(f"Your primary objective: {goal}")

    # Include background content if provided
    if background_content:
        prompt_parts.append(
            "Base your responses on this background content:\n\n"
            "=== BACKGROUND CONTENT ===\n"
            f"{background_content.strip()}\n"
            "=== END BACKGROUND CONTENT ==="
        )

    return "\n\n".join(prompt_parts)