import os
import sys
import traci
from sumolib import checkBinary
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, speed_levels=5, headway_levels=5, actions=7, alpha=0.1, gamma=0.9, epsilon=0.1):
        # init Q-table
        self.Q = np.zeros((speed_levels, headway_levels, actions))  # Q table

        # set up variables
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # explore rate

        # states
        self.speed_range = np.linspace(0, 100, speed_levels)
        self.headway_range = np.linspace(0, 100, headway_levels)

    def to_index(self, speed, headway):
        speed_index = np.digitize(speed, self.speed_range) - 1
        headway_index = np.digitize(headway, self.headway_range) - 1 if headway is not None else 0  # 为 None 时设置为 0
        return speed_index, headway_index

    def choose_action(self, speed_index, headway_index):
       
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2,3,4,5,6])  # randomly choose actions
        else:
            return np.argmax(self.Q[speed_index, headway_index])  # pick action with max Q value

    def  update_Q(self, speed_index, headway_index, action, reward, new_speed_index, new_headway_index):
        
        td_error =  reward + self.gamma * np.max(self.Q[new_speed_index, new_headway_index,action]) - self.Q[speed_index, headway_index, action]
        td_error = float(td_error)  #
        self.Q[speed_index, headway_index, action] = self.alpha * td_error + self.Q[speed_index, headway_index, action]
        return td_error

class SumoController:
    def __init__(self, config_path, show_gui=True):
    
        self.config_path = config_path
        self.show_gui = show_gui

    def start_SUMO(self):
        
        if 'SUMO_HOME' not in os.environ:
            sys.exit("请设置 'SUMO_HOME' 环境变量")

        sumoBinary = checkBinary('sumo-gui' if self.show_gui else 'sumo')
        sumoCmd = [sumoBinary, "-c", self.config_path, "--quit-on-end"]
        traci.start(sumoCmd)

    def get_observation(self):
        
        vehicle_ids = traci.vehicle.getIDList()
        speeds = {}
        headways = {}

        for vehicle_id in vehicle_ids:
            speeds[vehicle_id] = traci.vehicle.getSpeed(vehicle_id)
            #headways[vehicle_id] = traci.vehicle.getLeader(vehicle_id)[1] if traci.vehicle.getLeader(vehicle_id) else None
        
            leader_info = traci.vehicle.getLeader(vehicle_id)
            headways[vehicle_id] = leader_info[1] if leader_info else 500  # set up as None, when no leader vehicles
        return speeds, headways

    def get_reward(self):
        
        vehicle_ids = traci.vehicle.getIDList()
        movement_rewards = {}
        stability_rewards = {}

        current_speeds = {vehicle_id: traci.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicle_ids}

        for vehicle_id in vehicle_ids:
            current_speed = current_speeds[vehicle_id]
            movement_rewards[vehicle_id] = 5 if current_speed > 0 else -2 #10, -2 previous

            #prev_speed = current_speeds.get(vehicle_id, current_speed)
            #speed_change = abs(current_speed - prev_speed)
            #stability_rewards[vehicle_id] =  if speed_change < 4 else 10

        total_movement_reward = sum(movement_rewards.values())
        #total_stability_reward = sum(stability_rewards.values())
        total_reward = total_movement_reward #+ total_stability_reward

        return total_reward

    def control_speed(self, action_type):
        
        vehicle_ids = traci.vehicle.getIDList()
        time_step = traci.simulation.getDeltaT() / 1.0  # get time step
        speeds, headways = self.get_observation()

        for vehicle_id in vehicle_ids:
            current_speed = speeds[vehicle_id]
            leader_gap = headways[vehicle_id] if headways[vehicle_id] is not None else float('inf')

            # take action to speed according to their states
            if action_type == '-3':
                proposed_speed = current_speed-3
            elif action_type == '-2':
                proposed_speed = current_speed-2
            elif action_type == '-1':
                proposed_speed = current_speed-1
            elif action_type == '0':
                proposed_speed = current_speed
            elif action_type == '1':
                proposed_speed = current_speed+1
            elif action_type == '2':
                proposed_speed = current_speed+2
            else:
                proposed_speed = current_speed+3

            #required_gap = proposed_speed * time_step

            #if leader_gap < required_gap:
                #proposed_speed = leader_gap / time_step if leader_gap != float('inf') else proposed_speed

            traci.vehicle.setSpeed(vehicle_id, proposed_speed)

    def simulate_step(self, action_idx):
        
        action_set = {
            0: '-3',
            1: '-2',
            2: '-1',
            3:'0',
            4:'1',
            5:'2',
            6:'3',

        }

        action_type = action_set.get(action_idx, 'invalid')
        if action_type == 'invalid':
            print("Invalid action index.")
            return None

        self.control_speed(action_type)
        

        return self.get_observation(), self.get_reward()

# train
config_path ="C:/softwares/SUMO/ECO_Driving_2/RL.sumocfg"#Load the path of the sumo
sumo_controller = SumoController(config_path, show_gui=False)
sumo_controller.start_SUMO()

agent = QLearningAgent()

losses = [] 
rewards = [] 

for episode in range(300):  # SUMO, Run 200 eposide
    speeds, headways = sumo_controller.get_observation()
    

    episode_loss = 0
    episode_reward = 0

    for vehicle_id, (speed, headway) in zip(speeds.keys(), zip(speeds.values(), headways.values())):
        
        for step in range(10):
            speed_index, headway_index = agent.to_index(speed, headway)

            action = agent.choose_action(speed_index, headway_index)

            new_speeds, new_headways = sumo_controller.simulate_step(action)[0][0], sumo_controller.simulate_step(action)[0][1]

            new_speed_index, new_headway_index = agent.to_index(new_speeds[vehicle_id], new_headways[vehicle_id])

            reward = sumo_controller.get_reward()

            td_error = agent.update_Q(speed_index, headway_index, action, reward, new_speed_index, new_headway_index)

        episode_loss += abs(td_error)
        episode_reward += reward

    losses.append(episode_loss)
    rewards.append(episode_reward)

    print(f"Episode {episode} - Loss: {episode_loss}, Reward: {episode_reward}")
    traci.simulationStep(episode + 1)

# final Q-table
print("Final Q-table:\n", agent.Q)

# loss curve
def plot_data(episodes, losses, rewards):
    """Plot loss and reward by episode."""
    plt.figure(figsize=(12, 6))

    # Plot loss vs. episode
    plt.subplot(2, 1, 1)
    plt.plot(episodes, losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    # Plot reward vs. episode
    plt.subplot(2, 1, 2)
    plt.plot(episodes, rewards, label="Reward", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.tight_layout()
    plt.show()
# plt.plot(range(len(losses)), losses, label="Loss")
# plt.xlabel("Episode")
# plt.ylabel("Loss (Temporal Difference Error)")
# plt.title("Q-learning Loss Over Episodes")
# plt.legend()
# plt.show()
plot_data(episode,losses,rewards)

# close SUMO
traci.close()

