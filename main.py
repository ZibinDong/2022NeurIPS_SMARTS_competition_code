from src.agent import SACAgent
from src.sac import DiscreteSAC
from src.replay_buffer import ReplayBuffer
import smarts_env_v4 as smt

if __name__ == '__main__':
    env = smt.make(
        "loop", 
        is_endless_goal=True, 
        is_multi_scenario=True,
        visdom = False,
        average_sample=True,
    )
    replay_buffer= ReplayBuffer(env)
    agent = SACAgent(
        env , DiscreteSAC, device="cuda",
        target_entropy_coeff=0.1,
        actor_learning_rate=5E-4,
        critic_learning_rate=5E-4,
        temperature_learning_rate=5E-4
    )


