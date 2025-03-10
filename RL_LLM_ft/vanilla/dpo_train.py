# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import torch
import os

# 1. 환경 설정 - 멀티 GPU 활용
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 토크나이저 병렬화 활성화
device_map = "auto"  # 자동 장치 매핑으로 여러 GPU 활용

# 2. 데이터 전처리 (MATH 데이터셋용)
def generate_math_preference_data(examples):
    """
    GSM8K 데이터셋의 문제와 해설을 프리퍼런스 데이터로 변환합니다.
    """
    return {
        "prompt": [f"문제: {p}\nAnswer:" for p in examples["question"]],
        "chosen": examples["answer"],  # 리스트 형태로 직접 반환
        "rejected": [""] * len(examples["question"])
    }

# 3. 데이터셋 로드 및 전처리
dataset = load_dataset("gsm8k", "main", split="train[:10%]")

# 데이터셋 전처리
processed_dataset = dataset.map(
    generate_math_preference_data, 
    batched=True, 
    remove_columns=dataset.column_names,
    num_proc=4  # 병렬 처리 활용
)

# 4. 모델 초기화 (멀티 GPU 활용)
base_model = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True  # 필요한 경우 추가
)
ref_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# 5. DPO 설정 (멀티 GPU 최적화)
dpo_config = DPOConfig(
    output_dir="./dpo_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",
    optim="adamw_torch",
    beta=0.1,
    loss_type="sigmoid",
    max_length=512,
    max_prompt_length=256,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    learning_rate=5e-5,
    warmup_steps=100,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

# 6. 트레이너 초기화
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=processed_dataset,
    processing_class=tokenizer,  # 최신 버전에서는 지원됩니다.
)

# 7. 학습 실행 및 평가
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # GPU 정보 출력
    print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    dpo_trainer.train()
    dpo_trainer.save_model("./dpo_final_model")
    
    # 평가: 테스트 프롬프트 예시로 간단한 수학 문제 제공
    test_prompt = "문제: Solve for x: 2x + 5 = 15\nAnswer:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
