import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path):
    """Load model from either a directory or a .pt file"""
    print(f"Loading model from: {model_path}")
    
    # Check if model_path is a directory or a .pt file
    if os.path.isdir(model_path):
        # DPO style - directory with model files
        model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif model_path.endswith('.pt'):
        # PPO style - .pt state dict file
        # Need to determine the base model name
        base_model = "Qwen/Qwen2.5-0.5B"  # Default base model used in training
        model = AutoModelForCausalLM.from_pretrained(base_model).to("cuda")
        
        # Load state dict and filter out PPO-specific keys (v_head)
        state_dict = torch.load(model_path, map_location="cuda", weights_only=True)
        # Remove v_head keys that aren't part of the base model
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not k.startswith('v_head')}
        
        model.load_state_dict(filtered_state_dict)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        raise ValueError(f"Unsupported model path format: {model_path}")
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# 모델 경로 설정 (DPO 또는 PPO 모델 경로 지정)
# model_path = "./dpo_results/checkpoint-final"  # DPO 모델 경로
model_path = "/home/intern/ian/LLaMA-Factory/ppo_results/20250304_175942/ppo_final_model.pt"  # PPO 모델 경로 예시

# 모델 및 토크나이저 로드
model, tokenizer = load_model(model_path)

# 수학 문제 생성 및 풀이 함수
def solve_math_problem(problem):
    input_text = f"Question: {problem} Answer:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    return answer

# 예시 문제들
math_problems = [
    "What is 12 + 7?",
    "Solve for x: 2x + 3 = 11",
    "Calculate 15 divided by 3",
    "What is 9 multiplied by 8?",
    "What is the square root of 144?",
    "Solve for y: 3y - 5 = 16",
    "What is 25% of 80?",
    "If a triangle has sides of length 3, 4, and 5, what is its perimeter?",
    "Convert 3/4 to a decimal.",
    "What is the sum of the first 10 natural numbers?",
    "If a circle has a radius of 5, what is its circumference? (Use π ≈ 3.14)",
    "What is the value of 5! (5 factorial)?",
    "A train travels 120 km in 2 hours. What is its average speed?",
    "If x^2 = 49, what are the possible values of x?",
    "Simplify: (2x + 3) + (5x - 2)",
    "What is the least common multiple (LCM) of 6 and 8?",
    "Find the area of a rectangle with length 7 and width 4.",
    "Solve for x: 4(x - 2) = 20",
    "Convert 150° to radians.",
    "A store sells 3 apples for $2. How much do 9 apples cost?",
    "If f(x) = 2x + 1, what is f(5)?",
    "What is the remainder when 25 is divided by 4?",
    "Find the median of the set: [3, 7, 2, 9, 5].",
    "Expand: (x + 2)(x - 3).",
    "Solve for x: 3^(x+1) = 27.",
    "Find the derivative of f(x) = 3x^2 + 4x + 5.",
    "If a die is rolled, what is the probability of getting an even number?",
    "Convert 0.75 to a fraction in simplest form.",
    "If log_2(x) = 5, what is the value of x?",
    "What is the sum of the interior angles of a pentagon?"
]

# 문제 풀이 실행
for problem in math_problems:
    answer = solve_math_problem(problem)
    print(f"{problem} -> {answer}")