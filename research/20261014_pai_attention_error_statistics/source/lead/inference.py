"""
推理流程：输入构建、单样本推理。

此文件从 run_qwen25vl_example.py 提取而来，
核心推理逻辑（LEAD/CoT 调用、输入处理、输出解码）与原始实现完全一致。
"""

import argparse
from typing import Dict, Any

import torch
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

from .generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_cot,
    generate_cot_visual_reanchor,
    generate_pure_soft,
    generate_lead,
    generate_lead_attenachor,
)


def prepare_inputs(
    processor: AutoProcessor,
    messages,
    device: torch.device,
) -> Dict[str, Any]:
    """
    将对话消息编码为模型可接受的输入张量。

    Args:
        processor: Qwen2.5-VL 处理器。
        messages: 对话消息列表（含 image/text 内容）。
        device: 目标设备。

    Returns:
        dict: 包含 input_ids、attention_mask、pixel_values 等张量。
    """
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<|image_pad|>" not in chat_text:
        chat_text = chat_text.replace(
            "<|im_start|>user\n",
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>",
            1,
        )
    image_inputs, video_inputs = process_vision_info(messages)
    encoded = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    encoded = encoded.to(device)
    return dict(encoded)


def run_single_inference(model, processor, tokenizer, args: argparse.Namespace) -> str:
    """
    对单个样本执行推理，返回模型生成文本。

    核心逻辑与原始 run_qwen25vl_example.py 中的 main() 完全一致，
    仅将 model/processor/tokenizer 从全局变量改为显式参数传入。

    Args:
        model: 已加载的 Qwen2.5-VL 模型。
        processor: 对应的处理器。
        tokenizer: 对应的分词器。
        args: 命令行参数（包含 image, prompt, method, alpha 等）。

    Returns:
        str: 模型生成的文本（已去除首尾空白）。
    """
    set_seed(args.seed)
    compute_device = args.device
    if compute_device == "auto":
        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(compute_device)

    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    model_inputs = prepare_inputs(processor, messages, device)
    prompt_len = model_inputs["input_ids"].shape[1]
    args.prompt_tokens = int(prompt_len)
    args.token_entropy_trace = (
        [] if getattr(args, "save_token_entropy", False) else None
    )

    print("input_ids len:", prompt_len)
    if "image_grid_thw" in model_inputs:
        thw = model_inputs["image_grid_thw"]
        print("image_grid_thw:", thw.tolist())

    for key, value in model_inputs.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)

    math_ids_set = get_math_symbols_ids(tokenizer)
    math_ids_tensor = (
        torch.tensor(list(math_ids_set), device=device) if math_ids_set else None
    )

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    forced_prefix_ids = getattr(args, "forced_prefix_ids", None)
    if forced_prefix_ids is not None:
        gen_kwargs["forced_prefix_ids"] = forced_prefix_ids
    soft_vector_collector = getattr(args, "trace_soft_vector_collector", None)
    if soft_vector_collector is not None:
        gen_kwargs["trace_soft_vector_collector"] = soft_vector_collector
        gen_kwargs["trace_capture_soft_vector_step"] = getattr(
            args, "trace_capture_soft_vector_step", -1
        )
    external_route_vector = getattr(args, "trace_external_route_vector", None)
    if external_route_vector is not None:
        gen_kwargs["trace_external_route_vector"] = external_route_vector
        gen_kwargs["trace_external_route_source"] = getattr(
            args, "trace_external_route_source", None
        )
    if args.token_entropy_trace is not None:
        gen_kwargs["token_trace"] = args.token_entropy_trace
        gen_kwargs["trace_topk"] = getattr(args, "trace_topk", 0)
        gen_kwargs["trace_event_geometry"] = getattr(args, "trace_event_geometry", False)
        gen_kwargs["trace_event_steps"] = getattr(args, "trace_event_steps", [0, 1, 2, 4, 8, 16, 32])
        gen_kwargs["trace_route_override_step"] = getattr(args, "trace_route_override_step", -1)
        gen_kwargs["trace_route_override_kind"] = getattr(args, "trace_route_override_kind", "none")
        gen_kwargs["trace_route_override_mix_lambda"] = getattr(
            args, "trace_route_override_mix_lambda", 0.95
        )
        gen_kwargs["trace_forced_answer_probe"] = getattr(args, "trace_forced_answer_probe", False)
        gen_kwargs["trace_probe_gold_choice"] = getattr(args, "trace_probe_gold_choice", None)
        gen_kwargs["trace_probe_choice_case"] = getattr(args, "trace_probe_choice_case", "upper")
    if getattr(args, "save_visual_attn_summary", False):
        gen_kwargs["log_visual_attn_summary"] = True
        gen_kwargs["visual_attn_summary_last_k"] = args.visual_attn_summary_last_k
    if getattr(args, "sidecar_attn_on_entropy", False):
        gen_kwargs["sidecar_attn_on_entropy"] = True
        gen_kwargs["sidecar_attn_entropy_threshold"] = args.sidecar_attn_entropy_threshold
        gen_kwargs["sidecar_attn_last_k"] = args.sidecar_attn_last_k
    if args.method == "pure_soft":
        gen_kwargs["image_pad_bias"] = getattr(args, "pure_soft_image_pad_bias", False)
        gen_kwargs["image_pad_bias_lambda"] = getattr(args, "image_pad_bias_lambda", 0.0)
        gen_kwargs["image_pad_bias_min_step"] = getattr(args, "image_pad_bias_min_step", 0)
        gen_kwargs["image_pad_bias_max_step"] = getattr(args, "image_pad_bias_max_step", None)
        gen_kwargs["image_pad_bias_entropy_min"] = getattr(args, "image_pad_bias_entropy_min", None)
    if args.method == "cot_visual_reanchor":
        gen_kwargs["reanchor_entropy_threshold"] = args.reanchor_entropy_threshold
        gen_kwargs["reanchor_visual_attn_threshold"] = args.reanchor_visual_attn_threshold
        gen_kwargs["reanchor_lambda"] = args.reanchor_lambda
        gen_kwargs["reanchor_top_m"] = args.reanchor_top_m
        gen_kwargs["reanchor_attn_last_k"] = args.reanchor_attn_last_k
        gen_kwargs["reanchor_max_trigger_count"] = args.reanchor_max_trigger_count
        gen_kwargs["reanchor_cooldown"] = args.reanchor_cooldown
        gen_kwargs["reanchor_min_step"] = args.reanchor_min_step
        gen_kwargs["reanchor_max_step"] = args.reanchor_max_step
        gen_kwargs["reanchor_anchor_mode"] = args.reanchor_anchor_mode
        gen_kwargs["reanchor_trigger_mode"] = args.reanchor_trigger_mode
        gen_kwargs["reanchor_rolling_window"] = args.reanchor_rolling_window
        gen_kwargs["reanchor_min_history"] = args.reanchor_min_history
        gen_kwargs["reanchor_entropy_delta_threshold"] = args.reanchor_entropy_delta_threshold
        gen_kwargs["reanchor_visual_drop_threshold"] = args.reanchor_visual_drop_threshold

    if args.method == "cot_greedy":
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        if args.method == "lead":
            if math_ids_tensor is not None:
                model_inputs["math_ids_tensor"] = math_ids_tensor
            model_inputs["alpha_0"] = args.alpha
            model_inputs["max_switch_count"] = args.max_switch_count
            model_inputs["window_size"] = args.window_size
            model_inputs["lead_disable_simple_visual_anchor"] = args.lead_disable_simple_visual_anchor
            model_inputs["lead_force_normal"] = args.lead_force_normal
            model_inputs["lead_initial_soft_only"] = args.lead_initial_soft_only
            model_inputs["lead_initial_transition_only"] = args.lead_initial_transition_only
            model_inputs["lead_initial_transition_with_refinement"] = args.lead_initial_transition_with_refinement
            model_inputs["lead_force_initial_transition_step1"] = args.lead_force_initial_transition_step1
            model_inputs["lead_transition_source"] = args.lead_transition_source
            model_inputs["lead_transition_anchor"] = args.lead_transition_anchor
            model_inputs["beta_0"] = args.lead_transition_beta0
            model_inputs["lead_transition_norm_match"] = args.lead_transition_norm_match
            model_inputs["lead_transition_random_seed"] = args.lead_transition_random_seed
            model_inputs["lead_initial_transition_cache_rebuild_after_step"] = args.lead_initial_transition_cache_rebuild_after_step
            model_inputs["lead_initial_transition_cache_rebuild_prefix_len"] = args.lead_initial_transition_cache_rebuild_prefix_len
            model_inputs["lead_initial_transition_hard_boundary_only"] = args.lead_initial_transition_hard_boundary_only
            model_inputs["lead_early_visual_anchor"] = args.lead_early_visual_anchor
            model_inputs["lead_early_visual_anchor_source"] = args.lead_early_visual_anchor_source
            model_inputs["lead_early_visual_anchor_top_m"] = args.lead_early_visual_anchor_top_m
            model_inputs["lead_early_visual_anchor_lambda"] = args.lead_early_visual_anchor_lambda
            model_inputs["lead_early_visual_anchor_temperature"] = args.lead_early_visual_anchor_temperature
            model_inputs["lead_initial_transition_delay_steps"] = args.lead_initial_transition_delay_steps
            model_inputs["lead_transition_dynamic_entropy_window"] = args.lead_transition_dynamic_entropy_window
            model_inputs["lead_transition_dynamic_entropy_ratio"] = args.lead_transition_dynamic_entropy_ratio
            model_inputs["lead_transition_dynamic_min_history"] = args.lead_transition_dynamic_min_history
            model_inputs["lead_transition_dynamic_max_step"] = args.lead_transition_dynamic_max_step
            model_inputs["lead_transition_semantic_adaptive"] = args.lead_transition_semantic_adaptive
            model_inputs["lead_transition_semantic_entropy_threshold"] = args.lead_transition_semantic_entropy_threshold
            model_inputs["lead_transition_semantic_max_extra_steps"] = args.lead_transition_semantic_max_extra_steps
            model_inputs["lead_disable_step0_linebreak_mix"] = args.lead_disable_step0_linebreak_mix
            model_inputs["lead_disable_to_normal_transition"] = args.lead_disable_to_normal_transition
            model_inputs["lead_soft_quota_ratio"] = args.lead_soft_quota_ratio
            model_inputs["lead_refinement_window"] = args.lead_refinement_window
            model_inputs["lead_refinement_soft_cap"] = args.lead_refinement_soft_cap
            model_inputs["lead_refinement_entropy_threshold"] = (
                args.lead_refinement_entropy_threshold
            )
            model_inputs["lead_refinement_threshold_mode"] = (
                args.lead_refinement_threshold_mode
            )
            model_inputs["lead_refinement_online_quantile"] = (
                args.lead_refinement_online_quantile
            )
            model_inputs["lead_refinement_reference_ratio"] = (
                args.lead_refinement_reference_ratio
            )
            model_inputs["lead_refinement_soft_mix_lambda"] = (
                args.lead_refinement_soft_mix_lambda
            )
            model_inputs["lead_guard_candidate_only"] = args.lead_guard_candidate_only
            model_inputs["lead_disable_answer_zone_lock"] = args.lead_disable_answer_zone_lock
            model_inputs["convergence_words"] = "</think>"
            model_inputs["lead_soft_veto_on_diffuse"] = args.lead_soft_veto_on_diffuse
            model_inputs["lead_veto_entropy_window"] = args.lead_veto_entropy_window
            model_inputs["lead_veto_entropy_alpha"] = args.lead_veto_entropy_alpha
            model_inputs["lead_veto_min_history"] = args.lead_veto_min_history
            model_inputs["lead_veto_min_entropy"] = args.lead_veto_min_entropy
            model_inputs["lead_veto_low_conf_tau"] = args.lead_veto_low_conf_tau
            model_inputs["lead_veto_low_margin_tau"] = args.lead_veto_low_margin_tau
            model_inputs["lead_veto_min_step"] = args.lead_veto_min_step
            model_inputs["lead_veto_require_repeat_degen"] = args.lead_veto_require_repeat_degen
            model_inputs["lead_veto_repeat_ngram"] = args.lead_veto_repeat_ngram
            model_inputs["lead_veto_recent_repeat_window"] = args.lead_veto_recent_repeat_window
            model_inputs["lead_veto_recent_repeat_tau"] = args.lead_veto_recent_repeat_tau
            model_inputs["lead_format_cooldown"] = args.lead_format_cooldown
            model_inputs["format_cooldown_steps"] = args.format_cooldown_steps
            model_inputs["format_cooldown_min_step"] = args.format_cooldown_min_step
            model_inputs["format_cooldown_highrisk_only"] = args.format_cooldown_highrisk_only
            model_inputs["format_cooldown_normal_steps"] = args.format_cooldown_normal_steps
            model_inputs["format_cooldown_highrisk_steps"] = args.format_cooldown_highrisk_steps
            model_inputs["format_cooldown_max_active"] = args.format_cooldown_max_active
            model_inputs["format_cooldown_entropy_min"] = args.format_cooldown_entropy_min
            model_inputs["format_cooldown_top1_max"] = args.format_cooldown_top1_max
            model_inputs["format_cooldown_margin_max"] = args.format_cooldown_margin_max
            outputs = generate_lead(
                model,
                tokenizer,
                **model_inputs,
                **gen_kwargs,
            )
        elif args.method in {"lead_attenachor", "lead_attenanchor"}:
            if math_ids_tensor is not None:
                model_inputs["math_ids_tensor"] = math_ids_tensor
            model_inputs["alpha_0"] = args.alpha
            model_inputs["max_switch_count"] = args.max_switch_count
            model_inputs["window_size"] = args.window_size
            model_inputs["convergence_words"] = "</think>"
            model_inputs["visual_anchor_top_m"] = args.visual_anchor_top_m
            model_inputs["visual_anchor_attn_last_k"] = args.visual_anchor_attn_last_k
            model_inputs["visual_anchor_lambda_scale"] = args.visual_anchor_lambda_scale
            model_inputs["visual_anchor_entropy_upper"] = args.visual_anchor_entropy_upper
            model_inputs["visual_anchor_skip_nonword"] = args.visual_anchor_skip_nonword
            model_inputs["visual_anchor_single_use"] = args.visual_anchor_single_use
            model_inputs["soft_trigger_mode"] = args.soft_trigger_mode
            model_inputs["soft_warning_margin"] = args.soft_warning_margin
            model_inputs["soft_confirm_margin"] = args.soft_confirm_margin
            model_inputs["soft_delta2_threshold"] = args.soft_delta2_threshold
            model_inputs["soft_repeat_warning_boost"] = args.soft_repeat_warning_boost
            model_inputs["soft_repeat_confirm_boost"] = args.soft_repeat_confirm_boost
            model_inputs["soft_repeat_delta2_boost"] = args.soft_repeat_delta2_boost
            model_inputs["soft_repeat_cooldown"] = args.soft_repeat_cooldown
            model_inputs["soft_post_reset_ref_margin"] = args.soft_post_reset_ref_margin
            model_inputs["soft_post_reset_cooldown"] = args.soft_post_reset_cooldown
            outputs = generate_lead_attenachor(
                model,
                tokenizer,
                **model_inputs,
                **gen_kwargs,
            )
        elif args.method == "pure_soft":
            model_inputs["collapse_on_diffuse"] = getattr(args, "pure_soft_collapse_on_diffuse", False)
            model_inputs["collapse_entropy_window"] = args.collapse_entropy_window
            model_inputs["collapse_entropy_alpha"] = args.collapse_entropy_alpha
            model_inputs["collapse_min_history"] = args.collapse_min_history
            model_inputs["collapse_min_entropy"] = args.collapse_min_entropy
            model_inputs["collapse_low_conf_tau"] = args.collapse_low_conf_tau
            model_inputs["collapse_low_margin_tau"] = args.collapse_low_margin_tau
            model_inputs["collapse_min_step"] = args.collapse_min_step
            model_inputs["collapse_patience"] = args.collapse_patience
            model_inputs["collapse_patience_window"] = args.collapse_patience_window
            model_inputs["collapse_require_repeat_degen"] = args.collapse_require_repeat_degen
            model_inputs["collapse_repeat_ngram"] = args.collapse_repeat_ngram
            model_inputs["collapse_recent_repeat_window"] = args.collapse_recent_repeat_window
            model_inputs["collapse_recent_repeat_tau"] = args.collapse_recent_repeat_tau
            model_inputs["format_cooldown"] = args.pure_soft_format_cooldown
            model_inputs["format_cooldown_steps"] = args.format_cooldown_steps
            model_inputs["format_cooldown_min_step"] = args.format_cooldown_min_step
            model_inputs["format_cooldown_highrisk_only"] = args.format_cooldown_highrisk_only
            model_inputs["format_cooldown_normal_steps"] = args.format_cooldown_normal_steps
            model_inputs["format_cooldown_highrisk_steps"] = args.format_cooldown_highrisk_steps
            model_inputs["format_cooldown_mix_lambda"] = args.format_cooldown_mix_lambda
            model_inputs["format_cooldown_max_active"] = args.format_cooldown_max_active
            model_inputs["format_cooldown_entropy_min"] = args.format_cooldown_entropy_min
            model_inputs["format_cooldown_top1_max"] = args.format_cooldown_top1_max
            model_inputs["format_cooldown_margin_max"] = args.format_cooldown_margin_max
            model_inputs["answer_zone_discrete"] = args.pure_soft_answer_zone_discrete
            outputs = generate_pure_soft(
                model,
                tokenizer,
                **model_inputs,
                **gen_kwargs,
            )
        else:
            if args.method == "cot_visual_reanchor":
                outputs = generate_cot_visual_reanchor(
                    model,
                    tokenizer,
                    **model_inputs,
                    **gen_kwargs,
                )
            else:
                outputs = generate_cot(
                    model,
                    tokenizer,
                    **model_inputs,
                    **gen_kwargs,
                )

    generated_text = tokenizer.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("=== Prompt ===")
    print(args.prompt)
    print("=== Model Output ===")
    print(generated_text.strip())

    return generated_text.strip()
