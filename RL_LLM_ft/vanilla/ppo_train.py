#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datetime import datetime
import json

# 1. 사용 가능한 GPU 탐지 및 디바이스 설정
def get_available_gpus():
    """사용 가능한 GPU ID 리스트 반환"""
    free_memory = torch.cuda.mem_get_info()
    available_gpus = [i for i, mem in enumerate(free_memory) if mem > 1024 * 1024 * 1024]  # 최소 1GB 여유 메모리
    if not available_gpus:
        raise RuntimeError("사용 가능한 GPU가 없습니다.")
    return available_gpus

available_gpus = get_available_gpus()
device = torch.device(f"cuda:{available_gpus[0]}")  # 첫 번째 사용 가능한 GPU 선택

# 2. 모델명 및 디바이스 세팅
model_name = "Qwen/Qwen2.5-0.5B"  # Qwen 0.5B 모델 사용

# 3. 토크나이저 로드 및 패딩 토큰 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  
tokenizer.padding_side = "left"

# 4. 보상 모델 정의 (다른 GPU에 할당)
class RewardModel(torch.nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.scorer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=torch.float32
        ).to(device)
        self.scorer.config.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask):
        logits = self.scorer(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits.squeeze(-1)

reward_device = torch.device(f"cuda:{available_gpus[1]}") if len(available_gpus) > 1 else device

# 5. 데이터셋 준비 (수정 후: GSM8k 데이터셋 사용)
dataset = load_dataset("gsm8k", "main", split="train[:10%]")

def preprocess(example):
    # GSM8k 데이터셋의 실제 필드명은 'question'이 아닌 'question'과 'answer'임
    # 실제 필드명을 확인하고 수정
    return {"query": f"문제: {example['question']}"}


dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=4)

# 6. PPO 설정 수정
ppo_config = PPOConfig(
    learning_rate=1e-5,      # 학습률 유지
    batch_size=16,           # 배치 크기 증가 (4 -> 16)
    mini_batch_size=4,       # 미니배치 크기 증가 (1 -> 4)
    optimize_device_cache=True,
    max_grad_norm=1.0,
    gradient_accumulation_steps=2,  # 그래디언트 누적 추가
    target_kl=1.0            # 목표 KL 값 유지
)

# 7. 생성 모델 로드 및 패딩 토큰 설정 (첫 번째 GPU에 할당)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# 보상 모델 생성 (다른 GPU에 할당)
reward_model = RewardModel(model_name, reward_device)

# 8. PPO 트레이너 초기화
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset
)

# 9. 생성 파라미터 설정 수정
generation_kwargs = {
    "max_new_tokens": 100,
    "pad_token_id": tokenizer.eos_token_id,
    "temperature": 0.8,      # 온도 증가 (0.7 -> 0.8)
    "do_sample": True,
    "top_p": 0.95,          # top_p 증가 (0.9 -> 0.95)
    "top_k": 100,           # top_k 증가 (50 -> 100)
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 3,
    "num_beams": 1
}

# 입력 데이터 검증 함수 강화
def validate_input(text):
    if not isinstance(text, str):
        return False
    if not text.strip():
        return False
    if len(text) < 5:  # 최소 길이 체크
        return False
    if len(text) > 500:  # 최대 길이 제한
        return False
    return True

# 안전한 생성을 위한 래퍼 함수 추가
def safe_generate(model, inputs, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            with torch.cuda.amp.autocast(enabled=True):  # 자동 혼합 정밀도 사용
                outputs = model.generate(**inputs, **generation_kwargs)
                if torch.isfinite(outputs).all():
                    return outputs
        except RuntimeError as e:
            print(f"Generation attempt {attempt + 1} failed: {e}")
            torch.cuda.empty_cache()
            if attempt == max_attempts - 1:
                return None
    return None

# 10. PPO 학습 루프 수정
num_epochs = 20  # epoch 수 증가 (3 -> 20)
early_stopping_patience = 5  # 조기 종료 patience 설정
best_val_loss = float('inf')
patience_counter = 0
last_stats = None  # 마지막 stats 저장 변수 추가

# 결과 저장을 위한 디렉토리 생성
result_dir = os.path.join(".", "ppo_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(result_dir, exist_ok=True)

for epoch in range(num_epochs):
    epoch_stats = []  # 각 에폭의 stats 저장
    for batch_idx, batch in enumerate(tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch+1}")):
        try:
            # 입력 데이터 검증
            valid_queries = [q for q in batch["query"] if validate_input(q)]
            if not valid_queries:
                print("Warning: Skipping batch due to invalid inputs")
                continue

            inputs = tokenizer(
                valid_queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            query_tensors = [tensor for tensor in inputs["input_ids"]]
            
            # 안전한 생성 시도
            torch.cuda.synchronize()  # GPU 동기화
            response_tensors = safe_generate(ppo_trainer.model, query_tensors)
            
            if response_tensors is None:
                print("Warning: Generation failed after maximum attempts")
                continue

            # 텐서 안정화 처리
            processed_responses = []
            for resp in response_tensors:
                resp = resp.float().nan_to_num(nan=0.0, posinf=1e5, neginf=-1e5)
                resp = torch.clamp(resp, min=-10.0, max=10.0)
                processed_responses.append(resp)

            # 출력 검증
            valid_responses = []
            valid_queries = []
            for q, r in zip(query_tensors, processed_responses):
                if torch.isnan(r).any() or torch.isinf(r).any():
                    print("Warning: Invalid response detected, skipping")
                    continue
                valid_responses.append(r)
                valid_queries.append(q)
            
            if not valid_responses:
                print("Warning: No valid responses in batch")
                continue

            responses_text = [tokenizer.decode(resp, skip_special_tokens=True) for resp in valid_responses]
            
            # 보상 계산 및 안정화
            reward_inputs = tokenizer(
                responses_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(reward_device)
            
            with torch.no_grad():
                rewards = reward_model(reward_inputs["input_ids"], reward_inputs["attention_mask"])
                rewards = torch.clamp(rewards, min=-10, max=10)  # 보상값 제한
            
            # 텐서 차원 처리
            queries_1d = [q.squeeze() if q.dim() > 1 else q for q in valid_queries]
            responses_1d = [r.squeeze() if r.dim() > 1 else r for r in valid_responses]
            rewards_list = list(rewards.unbind())
            
            # PPO 스텝
            with torch.cuda.amp.autocast(enabled=True):  # 혼합 정밀도 학습
                stats = ppo_trainer.step(queries_1d, responses_1d, rewards_list)
                epoch_stats.append(stats)  # stats 저장
                last_stats = stats  # 마지막 stats 업데이트
            
            # 메모리 정리
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Batch processing error: {e}")
            torch.cuda.empty_cache()
            continue

        # 매 100 배치마다 중간 저장
        if (epoch * len(ppo_trainer.dataloader) + batch_idx) % 100 == 0:
            checkpoint_path = os.path.join(result_dir, f"ppo_checkpoint_e{epoch}_b{batch_idx}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    # 각 에폭 종료시 모델 저장
    epoch_path = os.path.join(result_dir, f"ppo_model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), epoch_path)

    # Early stopping 체크 - 에폭의 평균 손실 사용
    if epoch_stats:  # epoch_stats가 비어있지 않은 경우에만 실행
        epoch_loss = sum(s.get("loss", float('inf')) for s in epoch_stats) / len(epoch_stats)
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_model_path = os.path.join(result_dir, "ppo_best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# 11. 최종 모델 저장
final_model_path = os.path.join(result_dir, "ppo_final_model.pt")
torch.save(model.state_dict(), final_model_path)

# 학습 설정 저장
config_info = {
    "model_name": model_name,
    "ppo_config": ppo_config.__dict__,
    "generation_kwargs": generation_kwargs,
    "num_epochs": num_epochs,
    "early_stopping_patience": early_stopping_patience,
    "final_loss": last_stats.get("loss", None) if last_stats else None,
    "best_val_loss": best_val_loss,
    "training_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(result_dir, "training_config.json"), "w") as f:
    json.dump(config_info, f, indent=4)

print(f"\nTraining completed. All results saved in: {result_dir}")

# 12. 간단한 평가 (첫 번째 GPU 사용)
if __name__ == "__main__":
    test_prompt = "Solve for x: 2x + 5 = 15"
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**test_inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Test prompt:", test_prompt)
    print("Generated output:", generated_text)
