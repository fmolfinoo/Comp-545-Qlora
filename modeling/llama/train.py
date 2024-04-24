# Ignore the warning for flash attention
import warnings

warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
# End of ignore warning
from functools import partial
import json
import logging
from pathlib import Path

from accelerate import Accelerator, DataLoaderConfiguration
import datasets
from omegaconf import OmegaConf
import hydra
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig,
)
from trl import SFTTrainer

import weblinx as wl
from weblinx.processing import load_candidate_elements
from weblinx.processing.prompt import (
    build_input_records_from_selected_turns,
    select_turns_and_candidates_for_prompts,
)
from weblinx.utils.hydra import save_path_to_hydra_logs
from weblinx.utils import set_seed

from processing import (
    build_formatter_for_multichoice,
    build_prompt_records_for_llama_truncated,
    insert_formatted_chat_into_records,
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    set_seed(cfg.seed)
    split_path = Path(cfg.data.split_path).expanduser()
    model_save_dir = Path(cfg.model.save_dir).expanduser()
    model_save_dir.mkdir(exist_ok=True, parents=True)
    logging.info(OmegaConf.to_yaml(cfg))

    demo_names = wl.utils.load_demo_names_in_split(split_path, split=cfg.train.split)
    demos = [wl.Demonstration(demo_name, base_dir=cfg.data.base_dir) for demo_name in demo_names]
    #candidates = load_candidate_elements(path=cfg.candidates.train_path)
    # We load the tokenizer from the model directory
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.save_dir, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    # Modified accelerator to remove warning
    if cfg.train.use_accelerator_device_map:
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=None,
            split_batches=False,
            even_batches=True,
            use_seedable_sampler=True
        )
        accelerator = Accelerator(data_loader_config=dataloader_config)
        model_kwargs["device_map"] = {"": accelerator.process_index}

    elif cfg.train.use_auto_device_map:
        model_kwargs["device_map"] = "auto"

    if cfg.model.use_flash_attention_2:
        model_kwargs["use_flash_attention_2"] = True

    # Qlora Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.model.save_dir, quantization_config=bnb_config)
    # Prepare model gradient for training
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    #Lora configuration
    config = LoraConfig(
        r=264,
        lora_alpha=32,
        bias="all",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]  # Target modules for llama model
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    # End of model gradient preparation
    format_intent = build_formatter_for_multichoice()
    input_records_fname = "input_records_trunc.json"
    # We load the resulting dictionary directly from the input_records_trunc.json file to save 1 hour before starting training
    """
    build_prompt_records_fn = partial(
        build_prompt_records_for_llama_truncated,
        format_intent=format_intent,
        tokenizer=tokenizer,
    )

    selected_turns = select_turns_and_candidates_for_prompts(
        demos=demos,
        candidates=candidates,
        num_candidates=cfg.candidates.k,
    )
    input_records = build_input_records_from_selected_turns(
        selected_turns=selected_turns,
        format_intent=format_intent,
        build_prompt_records_fn=build_prompt_records_fn,
        format_prompt_records_fn=None,
    )

    template_tokenizer = AutoTokenizer.from_pretrained(cfg.model.save_dir)
    insert_formatted_chat_into_records(
        input_records, template_tokenizer, include_output_target=True
    )

    #with open(model_save_dir.joinpath(input_records_fname), "w") as f:
    #    json.dump(input_records, f, indent=2)
    """

    #Load the input records from the file
    with open(f"./{input_records_fname}", "r", encoding='utf-8') as f:
        input_records = json.load(f)

    input_records_texts = [{"text": record["text"]} for record in input_records]

    training_args = TrainingArguments(
        num_train_epochs=3,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        lr_scheduler_type=cfg.train.scheduler,
        learning_rate=cfg.train.learning_rate,
        save_strategy="no",
        evaluation_strategy="no",
        logging_first_step=True,
        prediction_loss_only=True,
        dataloader_num_workers=24,
        # Modified the following arguments
        output_dir="./outputModel",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_strategy="steps",
        optim="paged_adamw_8bit",
        # Added the following arguments
        fp16=True,
        logging_steps=500,
        # Disabled the following arguments
        # bf16=True,
        # bf16_full_eval=True,
        # warmup_ratio=cfg.train.warmup_ratio,
    )
    # Added dataset size Limit

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets.Dataset.from_list(input_records_texts),
        max_seq_length=model.config.max_position_embeddings,
        dataset_text_field="text",
    )

    trainer.train()

    # Save model, tokenizer, trainer state, and path to hydra logs
    trainer.model.save_pretrained("./outputModel/llama-7b-finetuned")
    tokenizer.save_pretrained("./outputModel/tokenizer")
    trainer.state.save_to_json("./outputModel/trainer_state.json")
    save_path_to_hydra_logs(save_dir="./outputModel")


if __name__ == "__main__":
    main()
