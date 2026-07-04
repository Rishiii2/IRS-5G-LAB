import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from irs_env import IRSEnv
from sac_agent import SACAgent
import os
import torch

def main():
    print("--- IRS-Sim DRL Training (SAC) ---")
    
    # 1. Initialize Environment and Agent
    env = IRSEnv(room_size=100.0, num_elements_x=8, num_elements_y=8)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    
    num_episodes = 1000
    batch_size = 64
    
    rewards_history = []
    
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
    print("Starting Training Loop...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # SAC inherently explores during training via stochastic policy
            action = agent.select_action(state, evaluate=False)
            
            next_state, reward, done, _, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train the network
            if len(agent.memory) > batch_size:
                policy_loss, qf_loss = agent.train(batch_size)
                
            state = next_state
            episode_reward += reward
            
        rewards_history.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes} | Total Reward (Average SNR): {episode_reward:.2f} dB")
        
    print("Training Complete!")
    
    # Save the trained Actor network weights
    save_path = os.path.join(os.path.dirname(__file__), 'sac_actor.pth')
    torch.save(agent.actor.state_dict(), save_path)
    print(f"Saved SAC actor weights to: {save_path}")
    
    # Plot Learning Curve
    plt.figure(figsize=(10,5))
    plt.plot(rewards_history, label='Episode Reward (SNR)', color='cyan')
    plt.title('SAC Agent Learning Curve (IRS Beamforming)')
    plt.xlabel('Episode')
    plt.ylabel('Reward (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(__file__), 'learning_curve.png')
    plt.savefig(plot_path)
    print(f"Saved learning curve plot to: {plot_path}")

if __name__ == "__main__":
    main()
