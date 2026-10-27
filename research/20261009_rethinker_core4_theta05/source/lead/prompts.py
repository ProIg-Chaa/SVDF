"""
提示模板管理。

为不同题型（MCQ / Open-Ended）和不同基准提供标准化的提示格式。
"""

from typing import Dict, Optional


# ============================== 基础模板 ==============================

MCQ_TEMPLATE = (
    "{question}\n\nOptions:\n{options}"
)

MCQ_COT_TEMPLATE = (
    "{question}\n\nOptions:\n{options}\n\n"
    "Please think step by step and then provide your answer."
)

OPEN_ENDED_TEMPLATE = (
    "{question}\n\n"
    "Please provide a detailed solution."
)

OPEN_ENDED_COT_TEMPLATE = (
    "{question}\n\n"
    "Please think step by step and provide your detailed solution."
)


# ============================== 格式化函数 ==============================

def format_mcq_prompt(
    question: str,
    options: str,
    use_cot: bool = False,
) -> str:
    """
    格式化选择题提示。

    Args:
        question: 题目文本。
        options: 选项文本（如 "A. xxx\\nB. yyy"）。
        use_cot: 是否添加 Chain-of-Thought 引导语。

    Returns:
        str: 格式化后的完整提示。
    """
    template = MCQ_COT_TEMPLATE if use_cot else MCQ_TEMPLATE
    return template.format(question=question, options=options)


def format_open_ended_prompt(
    question: str,
    use_cot: bool = False,
) -> str:
    """
    格式化开放式问答提示。

    Args:
        question: 题目文本。
        use_cot: 是否添加 Chain-of-Thought 引导语。

    Returns:
        str: 格式化后的完整提示。
    """
    template = OPEN_ENDED_COT_TEMPLATE if use_cot else OPEN_ENDED_TEMPLATE
    return template.format(question=question)


def format_prompt_from_sample(
    sample: Dict,
    use_cot: bool = False,
    question_key: str = "question",
    options_key: str = "options",
) -> str:
    """
    根据样本字典自动选择 MCQ 或 Open-Ended 模板进行格式化。

    Args:
        sample: 样本字典。
        use_cot: 是否添加 CoT 引导语。
        question_key: 题目对应的键名。
        options_key: 选项对应的键名。

    Returns:
        str: 格式化后的完整提示。
    """
    question = sample.get(question_key, "")
    options = sample.get(options_key, "")

    if options:
        return format_mcq_prompt(question, options, use_cot=use_cot)
    return format_open_ended_prompt(question, use_cot=use_cot)


def build_chat_messages(
    image_path: str,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> list:
    """
    构建 Qwen2.5-VL 格式的对话消息列表。

    Args:
        image_path: 图片路径或 URL。
        prompt: 用户提示文本。
        system_prompt: 可选的系统提示。

    Returns:
        list[dict]: 对话消息列表。
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]
    messages.append({"role": "user", "content": user_content})

    return messages
