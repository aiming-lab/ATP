from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from abc import abstractmethod

class MultiAgentHistoryRecord:
    """Record for storing multi-agent interaction history

    peer_history_mode controls how much of peers' messages are visible per round:
    - 'full': show full text responses of other agents (default)
    - 'final_only': show only the final parsed decision for other agents
    """
    
    def __init__(self, num_agents: int, peer_history_mode: str = "full"):
        self.num_agents = num_agents
        self.peer_history_mode = peer_history_mode if peer_history_mode in ("full", "final_only") else "full"
        self.rounds: List[Dict[str, Any]] = []
        self.agent_messages: List[List[Dict[str, str]]] = [[] for _ in range(num_agents)]  # Per-agent message history
        self.agent_total_rewards: List[float] = [0.0] * num_agents
        self.start_time = datetime.now()
        
    def add_round(self, round_data: Dict[str, Any]):
        """Add a round of multi-agent interaction to history"""
        round_data["round_id"] = len(self.rounds)
        round_data["timestamp"] = datetime.now().isoformat()
        self.rounds.append(round_data)
        
        if "rewards" in round_data:
            for i, reward in enumerate(round_data["rewards"]):
                if i < len(self.agent_total_rewards):
                    self.agent_total_rewards[i] += reward
        
        question = round_data.get('question', '')
        agent_responses = round_data.get('agent_responses', [])
        rewards = round_data.get('rewards', [])
        collective_outcome = round_data.get('collective_outcome', '')
        
        for agent_idx in range(self.num_agents):
            if question:
                self.agent_messages[agent_idx].append({"role": "user", "content": question})
            
            if agent_idx < len(agent_responses):
                self.agent_messages[agent_idx].append({
                    "role": "assistant", 
                    "content": agent_responses[agent_idx]
                })
            
            others_responses = []
            for other_idx, response in enumerate(agent_responses):
                if other_idx != agent_idx:
                    if self.peer_history_mode == "final_only":
                        choices = round_data.get("agent_choices") or []
                        if other_idx < len(choices) and choices[other_idx]:
                            others_responses.append(f"Agent {other_idx} final choice: {choices[other_idx]}")
                        else:
                            others_responses.append(f"Agent {other_idx}: {response}")
                    else:
                        others_responses.append(f"Agent {other_idx}: {response}")
            
            if others_responses:
                others_content = "Other agents' decisions: " + "; ".join(others_responses)
                self.agent_messages[agent_idx].append({"role": "user", "content": others_content})
            
            outcome_feedback = ""
            if collective_outcome:
                outcome_feedback += f"Outcome: {collective_outcome}. "
            if agent_idx < len(rewards):
                label = "Your reward"
                try:
                    if round_data.get("reward_is_delta") is True:
                        label = "Your reward (delta)"
                except Exception:
                    pass
                outcome_feedback += f"{label}: {rewards[agent_idx]}"
            
            if outcome_feedback:
                self.agent_messages[agent_idx].append({"role": "user", "content": outcome_feedback})
    
    def get_agent_history_messages(self, agent_idx: int, recent_rounds: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history for a specific agent"""
        if agent_idx >= len(self.agent_messages):
            return []
            
        if recent_rounds is None:
            return self.agent_messages[agent_idx].copy()
        else:
            # Each round has approximately 4 messages per agent: question + own response + others' responses + feedback
            messages_per_round = 4  
            start_idx = max(0, len(self.agent_messages[agent_idx]) - (recent_rounds * messages_per_round))
            return self.agent_messages[agent_idx][start_idx:].copy()
        
    def get_shared_history_summary(self, recent_rounds: Optional[int] = None) -> str:
        """Get formatted history that all agents can see (kept for compatibility)"""
        if recent_rounds:
            rounds_to_show = self.rounds[-recent_rounds:]
        else:
            rounds_to_show = self.rounds
            
        history_text = "=== Previous Game History ===\n"
        for round_data in rounds_to_show:
            history_text += f"Round {round_data['round_id']}:\n"
            
            if 'agent_responses' in round_data:
                for i, response in enumerate(round_data['agent_responses']):
                    history_text += f"  Agent {i}: {response}\n"
            
            if 'collective_outcome' in round_data:
                history_text += f"  Outcome: {round_data['collective_outcome']}\n"
                
            if 'rewards' in round_data:
                history_text += f"  Rewards: {round_data['rewards']}\n"
                
            history_text += "\n"
        return history_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Export history to dictionary"""
        return {
            "rounds": self.rounds,
            "agent_messages": self.agent_messages,  # Include per-agent message history
            "agent_total_rewards": self.agent_total_rewards,
            "start_time": self.start_time.isoformat(),
            "total_rounds": len(self.rounds),
            "num_agents": self.num_agents
        }

class MultiAgentSelfEvolvingEnvironment:
    """Multi-agent self-evolving environment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.environment_name = config.get("environment_name", "MultiAgent")
        self.num_agents = config.get("num_agents", 5)
        self.max_rounds = config.get("max_rounds", 10)
        self.cooperation_threshold = config.get("cooperation_threshold", 0.5)  # fraction needed for collective success
        # Control what peers see: 'full' or 'final_only'
        self.peer_history_mode = config.get("peer_history_mode", "full")
        self.history = MultiAgentHistoryRecord(self.num_agents, peer_history_mode=self.peer_history_mode)
        self.reward_mode = str(config.get("reward_mode", "additive")).lower()
        self.start_score = float(config.get("start_score", 1.0))
        if self.reward_mode == "multiplicative":
            self.scores: List[float] = [self.start_score for _ in range(self.num_agents)]
        
        self.high_reward = config.get("high_reward", 10.0)
        self.normal_reward = config.get("normal_reward", 5.0) 
        self.low_reward = config.get("low_reward", 1.0)
        
        self.collective_choice = config.get("collective_choice", "collective")
        self.independent_choice = config.get("independent_choice", "independent")
    
    def generate_question(self, round_id: int, history: MultiAgentHistoryRecord) -> str:
        """Generate question for current round"""
        base_question = self.config.get("base_question",
            "Choose your strategy for this round.")
        
        choice_explanation = self.config.get("choice_explanation",
            f"You can choose: 1) {self.collective_choice} (benefits everyone if enough choose it) "
            f"or 2) {self.independent_choice} (safer individual benefit)")
        # Allow merging choice description into the base question to avoid duplication
        if self.config.get("merge_prompts", False) or not choice_explanation:
            return base_question
        return f"{base_question} {choice_explanation}"
    
    def calculate_rewards(self, agent_responses: List[str], round_id: int, 
                         history: MultiAgentHistoryRecord) -> Tuple[List[float], str]:
        """Calculate rewards for all agents based on collective outcome.

        - additive: fixed per-round values (original behavior)
        - multiplicative: interpret rewards as factors and update per-agent scores (delta returned)
        """
        
        # Extract choices from responses
        choices = [self._extract_choice(response) for response in agent_responses]
        
        # Count collective choices
        collective_count = choices.count(self.collective_choice)
        collective_ratio = collective_count / len(choices)
        
        # Determine if cooperation succeeded
        cooperation_succeeded = collective_ratio >= self.cooperation_threshold
        
        if self.reward_mode == "multiplicative":
            rewards: List[float] = []
            new_scores: List[float] = []
            for idx, choice in enumerate(choices):
                before = self.scores[idx]
                if cooperation_succeeded:
                    factor = float(self.high_reward)
                else:
                    factor = float(self.low_reward if choice == self.collective_choice else self.normal_reward)
                after = before * factor
                rewards.append(after - before)
                new_scores.append(after)
            self.scores = new_scores

            explanation = (
                f"Collective choices: {collective_count}/{len(choices)} "
                f"({collective_ratio:.2%}). "
            )
            if cooperation_succeeded:
                explanation += f"Cooperation succeeded! Everyone multiplies by x{self.high_reward}."
            else:
                explanation += (
                    f"Cooperation failed. Collective x{self.low_reward}, "
                    f"independent x{self.normal_reward}."
                )
            return rewards, explanation
        else:
            # additive behavior
            rewards = []
            for choice in choices:
                if cooperation_succeeded:
                    reward = self.high_reward
                else:
                    reward = self.low_reward if choice == self.collective_choice else self.normal_reward
                rewards.append(reward)

            explanation = (
                f"Collective choices: {collective_count}/{len(choices)} "
                f"({collective_ratio:.2%}). "
            )
            if cooperation_succeeded:
                explanation += f"Cooperation succeeded! Everyone receives {self.high_reward}."
            else:
                explanation += (
                    f"Cooperation failed. Collective get {self.low_reward}, "
                    f"independent get {self.normal_reward}."
                )
            return rewards, explanation
    
    def _extract_choice(self, agent_response: str) -> str:
        """Extract choice from agent response using JSON format and regex"""
        import re
        import json
        
        # First try to parse JSON format
        try:
            # Look for JSON format: {"choice": "collective"} or {"choice": "independent"}
            json_pattern = r'\{[^}]*"choice"\s*:\s*"([^"]+)"[^}]*\}'
            json_match = re.search(json_pattern, agent_response, re.IGNORECASE)
            if json_match:
                choice = json_match.group(1).lower()
                if choice == self.collective_choice.lower():
                    return self.collective_choice
                elif choice == self.independent_choice.lower():
                    return self.independent_choice
        except:
            pass
        
        # Try dictionary format: choice: collective or choice: independent
        try:
            dict_pattern = r'"?choice"?\s*:\s*"?([^",\s]+)"?'
            dict_match = re.search(dict_pattern, agent_response, re.IGNORECASE)
            if dict_match:
                choice = dict_match.group(1).lower()
                if choice == self.collective_choice.lower():
                    return self.collective_choice
                elif choice == self.independent_choice.lower():
                    return self.independent_choice
        except:
            pass
        
        return "error: unable to determine"
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for agents.

        If a dataset/template provided a system prompt, append the mandatory JSON decision
        instruction when missing. Otherwise, use a sensible default including it.
        """
        pass
    
    def run_single_round(self, agents, round_id: int) -> Dict[str, Any]:
        """Run a single round with all agents.

        Supports two modes:
        - agents is a multiprocessing predictor with method predict_batch(messages_list)
        - agents is a list of model instances (fallback)
        """

        # Generate question
        question = self.generate_question(round_id, self.history)

        # Prepare personalized messages for every agent upfront
        messages_per_agent = [
            self._prepare_agent_messages(question, agent_idx=i)
            for i in range(self.num_agents)
        ]

        # Concurrent inference via predictor pool if available
        agent_responses: List[str] = []
        if hasattr(agents, "predict_batch") and callable(getattr(agents, "predict_batch")):
            try:
                agent_responses = agents.predict_batch(messages_per_agent)
            except Exception as e:
                # Fallback to sequential
                agent_responses = []
                for i in range(self.num_agents):
                    try:
                        response, _ = agents[i].predict_message(messages_per_agent[i])
                        agent_responses.append(response)
                    except Exception as se:
                        agent_responses.append(f"Error: {str(se)}")
        else:
            # Original sequential path when a plain agent list is passed
            for i in range(len(agents)):
                try:
                    response, _ = agents[i].predict_message(messages_per_agent[i])
                    agent_responses.append(response)
                except Exception as e:
                    agent_responses.append(f"Error: {str(e)}")
        
        rewards, outcome_explanation = self.calculate_rewards(
            agent_responses, round_id, self.history
        )
        agent_choices = [self._extract_choice(r) for r in agent_responses]
        
        round_data = {
            "question": question,
            "agent_responses": agent_responses,
            "agent_choices": agent_choices,
            "rewards": rewards,
            "collective_outcome": outcome_explanation,
            "round_id": round_id
        }
        if self.reward_mode == "multiplicative" and hasattr(self, "scores"):
            round_data["reward_is_delta"] = True
            round_data["scores_after"] = self.scores.copy()
        
        self.history.add_round(round_data)
        
        return round_data
    
    def _prepare_agent_messages(self, question: str, agent_idx: int) -> List[Dict[str, str]]:
        """Prepare messages for a specific agent including their personalized history"""
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        
        # Add agent-specific history as conversation messages
        if len(self.history.rounds) > 0:
            history_messages = self.history.get_agent_history_messages(
                agent_idx=agent_idx,
                recent_rounds=self.config.get("history_window", None)
            )
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": question})
        
        return messages
    
    def run_evolution(self, agents: List, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Run full multi-agent evolution process"""
        if num_rounds is None:
            num_rounds = self.max_rounds
            
        for round_id in range(num_rounds):
            round_data = self.run_single_round(agents, round_id)
            print(f"Round {round_id}: {round_data['collective_outcome']}")
        
        return self.get_evolution_summary()
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process"""
        return {
            "environment_name": self.environment_name,
            "num_agents": self.num_agents,
            "total_rounds": len(self.history.rounds),
            "agent_total_rewards": self.history.agent_total_rewards,
            "average_rewards": [r / len(self.history.rounds) if self.history.rounds else 0 
                              for r in self.history.agent_total_rewards],
            "system_prompt": self.get_system_prompt(),
            "history": self.history.to_dict(),
            "config": self.config
        }


class CooperativeGameEnvironment(MultiAgentSelfEvolvingEnvironment):
    """Specific implementation of cooperative game environment from README example"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
            
        default_config = {
            "environment_name": "CooperativeGame", 
            "num_agents": 5,
            "max_rounds": 12,
            "cooperation_threshold": 0.5,
            "high_reward": 15.0,
            "normal_reward": 8.0,
            "low_reward": 2.0,
            "collective_choice": "collective",
            "independent_choice": "independent", 
            "base_question": "Choose your strategy for this round.",
            "history_window": 3
        }
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        cooperation_threshold = config["cooperation_threshold"]
        num_agents = config["num_agents"]
        min_agents_needed = int(cooperation_threshold * num_agents) + (1 if cooperation_threshold * num_agents != int(cooperation_threshold * num_agents) else 0)
        
        if cooperation_threshold == 0.5:
            threshold_text = "more than half"
        elif cooperation_threshold >= 0.8:
            if min_agents_needed == num_agents:
                threshold_text = "everyone"
            else:
                threshold_text = f"at least {min_agents_needed} out of {num_agents}"
        elif cooperation_threshold >= 0.6:
            threshold_text = f"at least {min_agents_needed} out of {num_agents}"
        else:
            threshold_percent = int(cooperation_threshold * 100)
            threshold_text = f"at least {threshold_percent}%"
        
        if not config.get("choice_explanation"):
            config["choice_explanation"] = f"""You have two options: 1) {config['collective_choice']} (benefits everyone if {threshold_text} choose it) or 2) {config['independent_choice']} (safer individual benefit). If {threshold_text} choose {config['collective_choice']}, everyone gets maximum reward. If not, {config['collective_choice']} choosers get minimum reward while {config['independent_choice']} choosers get normal reward."""

        if not config.get("system_prompt"):
            config["system_prompt"] = f"""You are participating in a cooperative game with other agents. You can choose between '{config['collective_choice']}' (helps everyone if enough choose it) or '{config['independent_choice']}' (safer for you individually). Consider the history of previous rounds and other agents' behaviors. Learn from patterns and decide strategically. IMPORTANT: You must end your response with your final decision in JSON format: {{"choice": "{config['collective_choice']}"}} or {{"choice": "{config['independent_choice']}"}}."""
                
        super().__init__(config)
