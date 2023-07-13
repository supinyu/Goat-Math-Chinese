MODEL_PATH=/disk/model_checkpoint/Baichuan-13B-Chat

deepspeed --include="localhost:2" train_lora.py \
  --train_file /disk/nlp_info/LLM_dataset/school_math_0.25M_goat_train.json \
  --model_name_or_path $MODEL_PATH \
  --output_dir baichuan-13B-chat-lora-output \
  --prompt_column instruction \
  --response_column output \
  --cache_dir goat_cache \
  --overwrite_cache \
  --overwrite_output_dir \
  --max_source_length 512 \
  --max_target_length 512 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --logging_steps 50 \
  --save_steps 500 \
  --lora_rank 8 \
  --fp16 \
  --torch_dtype float16 \
  --preprocessing_num_workers 30 \
  --deepspeed ds_config.json