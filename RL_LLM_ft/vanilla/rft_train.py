# -*- coding: utf-8 -*-
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
import os
import tempfile

# Function to print available GPUs
def print_available_gpus():
    if torch.cuda.is_available():
        print("Available GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")

# Ask user to select a GPU
def select_gpu():
    print_available_gpus()
    choice = input("Enter the GPU index to use (e.g., 0, 1, 2,...): ")
    try:
        torch.cuda.set_device(int(choice))
        print(f"Using GPU {choice} - {torch.cuda.get_device_name(int(choice))}")
    except Exception as e:
        print(f"Error setting GPU: {e}")
        print("Defaulting to GPU 0")
        torch.cuda.set_device(0)

select_gpu()

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"  # GPU 3,4번만 사용
os.environ["TMPDIR"] = tempfile.mkdtemp(prefix="rft_tmp_")

# 1. 환경 설정
model_name = "Qwen/Qwen2.5-0.5B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 초기화 - 왼쪽 패딩으로 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # 왼쪽 패딩 설정

# 2. 데이터 준비
dataset = load_dataset("squad", split="train[:5%]")

def preprocess_function(examples):
    inputs = [
        context + " Question: " + question + " Answer: "
        for context, question in zip(examples["context"], examples["question"])
    ]
    targets = [
        answer["text"][0] if answer["text"] else "No answer available"
        for answer in examples["answers"]
    ]
    
    # 입력 토큰화 - text_target 사용
    model_inputs = tokenizer(
        inputs, 
        text_target=targets,  # text_target 사용
        max_length=512, 
        truncation=True, 
        padding="max_length",
        return_tensors="pt"
    )
    
    return model_inputs

# 데이터셋 전처리
dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset.column_names,
    num_proc=4
)

# 3. 보상 모델 정의
class AnswerRewardModel:
    def __init__(self):
        self.answer_prefix = "Answer:"
    
    def compute_reward(self, generated, reference):
        gen_answer = self._extract_answer(generated)
        ref_answer = reference
        return 1.0 if gen_answer.lower() == ref_answer.lower() else 0.0
    
    def _extract_answer(self, text):
        if self.answer_prefix in text:
            return text.split(self.answer_prefix)[-1].strip()
        return text.strip()

reward_model = AnswerRewardModel()

# 4. 사용자 정의 손실 함수
class RewardLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []
        self.reward_history = []

    def compute_loss(self, model, inputs, return_outputs=False):
        # 레이블 추출
        labels = inputs.pop("labels")
        
        # 어텐션 마스크 확인
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # 모델 출력 계산
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 생성 수행
        generation_kwargs = {
            "max_new_tokens": 128,  # max_length 대신 max_new_tokens 사용
            "pad_token_id": tokenizer.pad_token_id,
            "attention_mask": inputs["attention_mask"],
            "use_cache": False  # 그래디언트 체크포인팅과 호환되도록
        }
        
        generated_ids = model.generate(
            inputs["input_ids"],
            **generation_kwargs
        )
        
        # 텍스트 디코딩
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        reference_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 보상 계산
        rewards = [
            reward_model.compute_reward(gen, ref)
            for gen, ref in zip(generated_text, reference_text)
        ]
        rewards_tensor = torch.tensor(rewards, device=device).unsqueeze(-1)
        
        # 로그 확률 계산
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 배치 크기 일치 확인
        if log_probs.size(0) != labels.size(0):
            print(f"배치 크기 불일치: logits={log_probs.size()}, labels={labels.size()}")
            min_batch = min(log_probs.size(0), labels.size(0))
            log_probs = log_probs[:min_batch]
            labels = labels[:min_batch]
            rewards_tensor = rewards_tensor[:min_batch]
        
        # 선택된 토큰의 로그 확률 추출
        selected_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 손실 계산
        loss = -torch.mean(selected_log_probs * rewards_tensor)
        
        # 손실 및 보상 기록
        self.loss_history.append(loss.item())
        self.reward_history.append(np.mean(rewards))
        
        print(f"현재 손실: {loss.item()}, 평균 보상: {np.mean(rewards)}")  # 진행 상황 출력
        
        return (loss, outputs) if return_outputs else loss

# 5. 메인 실행
if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 모델 초기화
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # bfloat16 대신 float32 사용
        device_map="auto",
        use_cache=False  # 그래디언트 체크포인팅과 호환되도록
    )
    
    # 학습 인자
    args = TrainingArguments(
        output_dir="./rft_results",
        per_device_train_batch_size=1,  # 배치 크기 감소
        gradient_accumulation_steps=8,  # 그래디언트 누적 단계 증가
        num_train_epochs=3,
        fp16=False,
        bf16=False,  # bfloat16 비활성화
        logging_steps=10,  # 로깅 빈도 증가
        save_strategy="epoch",
        optim="adamw_torch",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        warmup_steps=100,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 트레이너
    trainer = RewardLossTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # 학습 실행
    trainer.train()

    # 학습 기록 저장
    np.save("loss_history.npy", trainer.loss_history)
    np.save("reward_history.npy", trainer.reward_history)

    # 모델 저장
    model.save_pretrained("./rft_final_model")
