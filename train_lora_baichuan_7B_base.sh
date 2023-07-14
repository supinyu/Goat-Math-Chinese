MODEL_PATH=/disk/model_checkpoint/baichuan-7B

deepspeed --include="localhost:1,3" train_lora.py \
  --train_file /disk/nlp_info/LLM_dataset/school_math_0.5M_goat_train.json \
  --model_name_or_path $MODEL_PATH \
  --output_dir lora_goat_output \
  --prompt_column instruction \
  --response_column output \
  --cache_dir goat_cache \
  --overwrite_cache \
  --overwrite_output_dir \
  --max_source_length 512 \
  --max_target_length 512 \
  --num_train_epochs 2 \
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
  --preprocessing_num_workers 30 \
  --deepspeed ds_config.json
