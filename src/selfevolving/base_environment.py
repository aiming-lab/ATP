from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class HistoryRecord:
    """Record for storing interaction history"""
    
    def __init__(self):
        self.rounds: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, str]] = []
        self.total_reward = 0.0
        self.start_time = datetime.now()
        
    def add_round(self, round_data: Dict[str, Any]):
        """Add a round of interaction to history"""
        round_data["round_id"] = len(self.rounds)
        round_data["timestamp"] = datetime.now().isoformat()
        self.rounds.append(round_data)
        
        question = round_data.get('question', '')
        if question:
            self.messages.append({"role": "user", "content": question})
        
        agent_response = round_data.get('agent_response', '')
        if agent_response:
            self.messages.append({"role": "assistant", "content": agent_response})
        
        reward = round_data.get('reward', 0)
        reward_explanation = round_data.get('reward_explanation', '')
        if reward_explanation:
            reward_feedback = f"Reward: {reward}. {reward_explanation}"
        else:
            reward_feedback = f"Reward: {reward}"
        self.messages.append({"role": "user", "content": reward_feedback})
        
    def get_history_messages(self, recent_rounds: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history as messages list"""
        if recent_rounds is None or recent_rounds * 3 >= len(self.messages):
            return self.messages.copy()
        else:
            start_idx = max(0, len(self.messages) - (recent_rounds * 3))
            return self.messages[start_idx:].copy()
        
    def get_history_summary(self, recent_rounds: Optional[int] = None) -> str:
        """Get formatted history summary (kept for compatibility)"""
        if recent_rounds:
            rounds_to_show = self.rounds[-recent_rounds:]
        else:
            rounds_to_show = self.rounds
            
        history_text = "=== Previous Interaction History ===\n"
        for round_data in rounds_to_show:
            history_text += f"Round {round_data['round_id']}:\n"
            history_text += f"  Question: {round_data.get('question', 'N/A')}\n"
            if 'agent_response' in round_data:
                history_text += f"  Your Response: {round_data['agent_response']}\n"
            if 'reward' in round_data:
                history_text += f"  Reward: {round_data['reward']}\n"
            if 'reward_explanation' in round_data:
                history_text += f"  Explanation: {round_data['reward_explanation']}\n"
            history_text += "\n"
        return history_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Export history to dictionary"""
        return {
            "rounds": self.rounds,
            "messages": self.messages,  # Include message history
            "total_reward": self.total_reward,
            "start_time": self.start_time.isoformat(),
            "total_rounds": len(self.rounds)
        }

class BaseSelfEvolvingEnvironment(ABC):
    """Base class for self-evolving environments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.environment_name = config.get("environment_name", "Unknown")
        self.max_rounds = config.get("max_rounds", 10)
        self.history = HistoryRecord()
        
    @abstractmethod
    def generate_question(self, round_id: int, history: HistoryRecord) -> str:
        """Generate question for the current round"""
        pass
    
    @abstractmethod
    def calculate_reward(self, agent_response: str, supervision_state: int, 
                        round_id: int, history: HistoryRecord) -> Tuple[float, str]:
        """Calculate reward and provide explanation"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for agents"""
        pass
    
    def is_supervised(self, round_id: int) -> int:
        """Determine supervision state for current round
        Returns: 1 (supervised), -1 (unsupervised), 0 (unknown/neutral)
        """
        supervision_pattern = self.config.get("supervision_pattern", "random")
        if supervision_pattern == "alternating":
            return 1 if round_id % 2 == 0 else -1
        elif supervision_pattern == "mostly_supervised":
            return -1 if round_id % 4 == 3 else 1
        elif supervision_pattern == "mostly_unsupervised":
            return 1 if round_id % 4 == 0 else -1
        elif supervision_pattern == "last_unsupervised":
            if round_id == self.max_rounds - 1:
                return -1
            else:
                return 1
        elif supervision_pattern == "last_supervised":
            if round_id == self.max_rounds - 1:
                return 1
            else:
                return -1
        elif supervision_pattern == "all_supervised":
            return 1
        elif supervision_pattern == "all_unsupervised":
            return -1
        else:
            return 1 if (round_id + self.config.get("seed", 0)) % 3 == 0 else -1
    
    def run_single_round(self, agent, round_id: int) -> Dict[str, Any]:
        """Run a single round of interaction"""
        question = self.generate_question(round_id, self.history)
        supervision_state = self.is_supervised(round_id)
        messages = self._prepare_agent_messages(question, supervision_state)
        
        try:
            agent_response, _ = agent.predict_message(messages)
        except Exception as e:
            agent_response = f"Error: {str(e)}"
        
        reward, reward_explanation = self.calculate_reward(
            agent_response, supervision_state, round_id, self.history
        )
        
        round_data = {
            "question": question,
            "supervision_state": supervision_state,
            "agent_response": agent_response,
            "reward": reward,
            "reward_explanation": reward_explanation,
            "round_id": round_id
        }
        
        self.history.add_round(round_data)
        self.history.total_reward += reward
        
        return round_data
    
    def _prepare_agent_messages(self, question: str, supervision_state: int) -> List[Dict[str, str]]:
        """Prepare messages for agent including history"""
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        
        if len(self.history.rounds) > 0:
            history_messages = self.history.get_history_messages(
                recent_rounds=self.config.get("history_window", None)
            )
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": question})
        
        return messages
    
    def run_evolution(self, agent, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Run full evolution process"""
        if num_rounds is None:
            num_rounds = self.max_rounds
            
        for round_id in range(num_rounds):
            round_data = self.run_single_round(agent, round_id)
            print(f"Round {round_id}: Reward = {round_data['reward_explanation']}")
        
        return self.get_evolution_summary()
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process"""
        return {
            "environment_name": self.environment_name,
            "total_rounds": len(self.history.rounds),
            "total_reward": self.history.total_reward,
            "average_reward": self.history.total_reward / len(self.history.rounds) if self.history.rounds else 0,
            "history": self.history.to_dict(),
            "config": self.config
        }