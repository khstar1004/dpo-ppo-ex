import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random
from trl import AutoModelForCausalLMWithValueHead  # trl에서 Value Head 모델 임포트 추가
import os
from datetime import datetime

def is_answer_correct(decoded_output: str, correct_answer: str) -> bool:
    """
    생성된 출력에서 숫자를 추출하여, 올바른 정답과 비교합니다.
    만약 숫자 추출이 어려운 경우, 단순 문자열 포함 여부로 판단합니다.
    """
    # 출력에서 모든 숫자(정수 혹은 실수) 추출
    matches = re.findall(r"[-+]?\d*\.?\d+", decoded_output)
    if matches:
        try:
            expected = float(correct_answer)
            for match in matches:
                try:
                    num = float(match)
                    if abs(num - expected) < 1e-5:  # 아주 작은 오차 허용
                        return True
                except:
                    continue
            return False
        except:
            return correct_answer.strip() in decoded_output
    else:
        return correct_answer.strip() in decoded_output

def generate_arithmetic_problems(num_problems: int):
    """
    임의의 산술 문제와 정답을 생성합니다.
    
    Args:
        num_problems (int): 생성할 문제 수.
    
    Returns:
        tuple: 문제 리스트와 정답 리스트.
    """
    problems = []
    answers = []
    operators = ["+", "-", "*", "/"]
    for i in range(num_problems):
        op = random.choice(operators)
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        if op == "/":
            # 나눗셈의 경우 정수 결과를 보장하기 위해 a를 b의 배수로 설정
            a = a * b
        problem_str = f"문제: {a} {op} {b}는 얼마입니까?"
        if op == "+":
            answer = str(a + b)
        elif op == "-":
            answer = str(a - b)
        elif op == "*":
            answer = str(a * b)
        elif op == "/":
            answer = str(a // b)
        problems.append(problem_str)
        answers.append(answer)
    return problems, answers

def evaluate_math_problems_advanced(model, tokenizer, problems, answers, max_new_tokens=128, num_samples=1):
    """
    여러 샘플링과 개선된 정답 검증 방식을 적용하여 산술 문제에 대해 모델을 평가합니다.
    
    Args:
        model: PreTrainedModel 객체.
        tokenizer: 해당 토크나이저.
        problems (list[str]): 산술 문제 리스트.
        answers (list[str]): 각 문제에 대한 정답 리스트.
        max_new_tokens (int): 생성할 최대 토큰 수.
        num_samples (int): 각 문제마다 생성할 샘플 수.
    
    Returns:
        tuple: (정확도, 대표 출력 리스트)
    """
    model.eval()
    correct = 0
    outputs = []

    for problem, correct_answer in zip(problems, answers):
        inputs = tokenizer(problem, return_tensors="pt")
        generated_answers = []
        for i in range(num_samples):
            with torch.no_grad():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(num_samples > 1)  # 다중 샘플링 시 샘플링 모드 활성화
                )
            decoded_output = tokenizer.decode(generation[0], skip_special_tokens=True)
            generated_answers.append(decoded_output)
        
        # 여러 샘플 중 하나라도 정답을 포함하면 정답으로 간주
        if any(is_answer_correct(ans, correct_answer) for ans in generated_answers):
            correct += 1
        outputs.append(generated_answers[0])  # 첫 번째 생성 결과를 대표 출력으로 기록

    accuracy = correct / len(problems) if problems else 0.0
    return accuracy, outputs

def compare_models(
        base_model_name="Qwen/Qwen2.5-0.5B",
        dpo_path="dpo_results/checkpoint-1641",
        ppo_path="/home/intern/ian/LLaMA-Factory/ppo_results/20250305_001353/ppo_final_model.pt",
        num_problems=100,
        num_samples=3,
        save_dir="evaluation_results"
    ):
    """
    세 모델(PLAIN, DPO, PPO)을 로드하여 산술 문제에 대해 평가하고 정확도를 시각화합니다.
    결과를 보고서로 작성하고 그래프를 저장합니다.
    
    Args:
        base_model_name (str): 기본 모델 이름
        dpo_path (str): DPO 모델 경로
        ppo_path (str): PPO 모델의 state_dict 경로
        num_problems (int): 평가할 문제 수
        num_samples (int): 각 문제마다 생성할 샘플 수
        save_dir (str): 결과를 저장할 디렉토리 경로
    """
    # 대량의 산술 문제 생성
    problems, answers = generate_arithmetic_problems(num_problems)
    print(f"총 {num_problems}개의 산술 문제로 평가합니다.")

    # ------------------------
    # 1) PLAIN 모델 로드 및 평가
    # ------------------------
    print("Loading PLAIN model from:", base_model_name)
    tokenizer_plain = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer_plain.pad_token = tokenizer_plain.eos_token
    model_plain = AutoModelForCausalLM.from_pretrained(base_model_name)
    plain_accuracy, plain_outputs = evaluate_math_problems_advanced(
        model_plain, tokenizer_plain, problems, answers, num_samples=num_samples
    )
    print(f"PLAIN model accuracy: {plain_accuracy:.3f}")
    for i in range(min(5, len(plain_outputs))):
        print(f"[PLAIN] Q: {problems[i]} -> A: {plain_outputs[i]} (정답: {answers[i]})")

    # ------------------------
    # 2) DPO 체크포인트 로드 및 평가
    # ------------------------
    print("\nLoading DPO model from:", dpo_path)
    tokenizer_dpo = AutoTokenizer.from_pretrained(dpo_path)
    model_dpo = AutoModelForCausalLM.from_pretrained(dpo_path)
    dpo_accuracy, dpo_outputs = evaluate_math_problems_advanced(
        model_dpo, tokenizer_dpo, problems, answers, num_samples=num_samples
    )
    print(f"DPO model accuracy: {dpo_accuracy:.3f}")
    for i in range(min(5, len(dpo_outputs))):
        print(f"[DPO] Q: {problems[i]} -> A: {dpo_outputs[i]} (정답: {answers[i]})")

    # ------------------------
    # 3) PPO 모델 로드 및 평가 (수정된 부분)
    # ------------------------
    print("\nLoading PPO model from:", ppo_path)
    # 기존 모델 구조로 먼저 로드한 후 state_dict 적용
    model_ppo = AutoModelForCausalLM.from_pretrained(base_model_name)
    state_dict = torch.load(ppo_path, map_location="cpu", weights_only=True)
    
    # 키 이름에서 'pretrained_model.' 프리픽스 제거 (필요한 경우)
    new_state_dict = {k.replace("pretrained_model.", ""): v for k, v in state_dict.items()}
    model_ppo.load_state_dict(new_state_dict, strict=False)
    
    tokenizer_ppo = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer_ppo.pad_token = tokenizer_ppo.eos_token
    
    ppo_accuracy, ppo_outputs = evaluate_math_problems_advanced(
        model_ppo, tokenizer_ppo, problems, answers, num_samples=num_samples
    )
    print(f"PPO model accuracy: {ppo_accuracy:.3f}")
    for i in range(min(5, len(ppo_outputs))):
        print(f"[PPO] Q: {problems[i]} -> A: {ppo_outputs[i]} (정답: {answers[i]})")

    # ------------------------
    # 4) 평가 결과 시각화 및 저장
    # ------------------------
    model_names = ["PLAIN Model", "DPO Model", "PPO Model"]
    accuracies = [plain_accuracy, dpo_accuracy, ppo_accuracy]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, 
                  color=["gray", "blue", "green"], 
                  alpha=0.7)
    plt.ylim([0, 1])
    plt.title("산술 문제 평가 정확도 비교")
    plt.ylabel("정확도")
    
    for bar, acc in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    
    plt.xticks(rotation=15)
    plt.tight_layout()

    # 결과 저장을 위한 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    # 그래프 저장
    plt.savefig(os.path.join(save_path, "accuracy_comparison.png"))
    plt.close()

    # 보고서 작성
    report = f"""# 모델 성능 평가 보고서
날짜: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 평가 설정
- 평가 문제 수: {num_problems}개
- 문제당 샘플 수: {num_samples}개
- 기본 모델: {base_model_name}
- DPO 모델 경로: {dpo_path}
- PPO 모델 경로: {ppo_path}

## 평가 결과
1. PLAIN 모델 정확도: {plain_accuracy:.3f}
2. DPO 모델 정확도: {dpo_accuracy:.3f}
3. PPO 모델 정확도: {ppo_accuracy:.3f}

## 샘플 출력 비교
"""
    
    # 샘플 출력 추가
    for i in range(min(5, len(problems))):
        report += f"\n### 문제 {i+1}\n"
        report += f"Q: {problems[i]}\n"
        report += f"- PLAIN 출력: {plain_outputs[i]}\n"
        report += f"- DPO 출력: {dpo_outputs[i]}\n"
        report += f"- PPO 출력: {ppo_outputs[i]}\n"
        report += f"- 정답: {answers[i]}\n"

    # 성능 분석 추가
    report += "\n## 성능 분석\n"
    best_model = model_names[accuracies.index(max(accuracies))]
    report += f"- 최고 성능 모델: {best_model} (정확도: {max(accuracies):.3f})\n"
    report += f"- 성능 향상도:\n"
    report += f"  * DPO vs PLAIN: {((dpo_accuracy/plain_accuracy) - 1)*100:.1f}%\n"
    report += f"  * PPO vs PLAIN: {((ppo_accuracy/plain_accuracy) - 1)*100:.1f}%\n"

    # 보고서 저장
    with open(os.path.join(save_path, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n결과가 다음 경로에 저장되었습니다: {save_path}")
    
    return {
        "accuracies": {
            "plain": plain_accuracy,
            "dpo": dpo_accuracy,
            "ppo": ppo_accuracy
        },
        "sample_outputs": {
            "plain": list(zip(problems[:5], plain_outputs[:5], answers[:5])),
            "dpo": list(zip(problems[:5], dpo_outputs[:5], answers[:5])),
            "ppo": list(zip(problems[:5], ppo_outputs[:5], answers[:5]))
        },
        "save_path": save_path
    }

if __name__ == "__main__":
    results = compare_models()
