import json
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
from matplotlib.ticker import MaxNLocator

def visualize_dpo_training():
    # 체크포인트 경로 설정 - 모든 체크포인트 자동 탐색
    base_dir = "../dpo_results"
    checkpoint_dirs = sorted(glob.glob(os.path.join(base_dir, "checkpoint-*")))
    
    plt.figure(figsize=(12, 8))
    
    # 각 체크포인트의 손실(loss) 시각화
    all_steps = []
    all_losses = []
    
    for checkpoint in checkpoint_dirs:
        checkpoint_name = os.path.basename(checkpoint)
        trainer_state_path = os.path.join(checkpoint, "trainer_state.json")
        
        # trainer_state.json 파일 로드
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            
            # 손실(loss) 값 추출
            steps = [log["step"] for log in trainer_state["log_history"] if "loss" in log]
            losses = [log["loss"] for log in trainer_state["log_history"] if "loss" in log]
            
            all_steps.extend(steps)
            all_losses.extend(losses)
            
            # 손실 그래프 그리기
            plt.plot(steps, losses, 'o-', label=f"{checkpoint_name}", alpha=0.7)
    
    # 전체 추세선 추가
    if all_steps and all_losses:
        z = np.polyfit(all_steps, all_losses, 1)
        p = np.poly1d(z)
        plt.plot(sorted(all_steps), p(sorted(all_steps)), "r--", linewidth=2, label="Overall Trend")
    
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("DPO Training Loss Over Time", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("dpo_training_loss.png", dpi=300)
    plt.show()

def visualize_rft_training():
    # RFT 학습 결과 시각화
    loss_history = np.load("loss_history.npy")
    reward_history = np.load("reward_history.npy")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 손실 그래프
    ax1.plot(loss_history, 'b-', linewidth=2)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("RFT Training Loss", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 보상 그래프
    ax2.plot(reward_history, 'g-', linewidth=2)
    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Average Reward", fontsize=12)
    ax2.set_title("RFT Training Reward", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("rft_training_metrics.png", dpi=300)
    plt.show()

def compare_methods():
    # 여러 방법론 비교 (가정: 각 방법의 평가 결과가 있다고 가정)
    methods = ["RFT", "DPO", "PPO"]
    accuracy = [0.75, 0.82, 0.78]  # 예시 데이터
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracy, color=['blue', 'green', 'orange'])
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.0)
    plt.xlabel("Training Method", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Comparison of Different Training Methods", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("method_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # 모든 시각화 실행
    visualize_dpo_training()
    
    # RFT 결과 파일이 있는 경우에만 실행
    if os.path.exists("loss_history.npy") and os.path.exists("reward_history.npy"):
        visualize_rft_training()
    
    # 방법론 비교 (선택적)
    # compare_methods() 