from typing import List, Dict, Any, Optional, Tuple

from .multi_agent_environment import (
    CooperativeGameEnvironment,
    MultiAgentHistoryRecord,
)

class InvestmentCooperativeGameEnvironment(CooperativeGameEnvironment):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.start_capital: float = float(config.get("start_capital", 1.0))
        self.success_applies_to_all: bool = bool(config.get("success_applies_to_all", True))
        self.fixed_cooperators_k: int = int(config.get("fixed_cooperators_k", 0))
        self.fixed_cooperators_strategy: str = str(config.get("fixed_cooperators_strategy", "tail"))
        self.capitals: List[float] = [self.start_capital for _ in range(self.num_agents)]

    def calculate_rewards(
        self,
        agent_responses: List[str],
        round_id: int,
        history: MultiAgentHistoryRecord,
    ) -> Tuple[List[float], str]:
        """Compute per-round rewards and update capitals.

        Supports two reward modes:
        - multiplicative (default): capital_after = capital_before * factor
        - additive: capital_after = capital_before + (factor - 1.0) * additive_base
          where additive_base = config.get('additive_base', start_capital)

        On success: if success_applies_to_all, all use high; else only collective choosers use high.
        On failure: collective choosers use low; independent use normal.
        """

        choices = [self._extract_choice(response) for response in agent_responses]

        collective_count = choices.count(self.collective_choice)
        collective_ratio = collective_count / len(choices)
        cooperation_succeeded = collective_ratio >= self.cooperation_threshold

        rewards: List[float] = []
        new_capitals: List[float] = []
        reward_mode = str(self.config.get("reward_mode", "multiplicative")).lower()
        additive_base = float(self.config.get("additive_base", self.start_capital))

        for idx, choice in enumerate(choices):
            capital_before = self.capitals[idx]

            if cooperation_succeeded:
                if self.success_applies_to_all:
                    factor = float(self.high_reward)
                else:
                    factor = float(self.high_reward if choice == self.collective_choice else self.normal_reward)
            else:
                factor = float(self.low_reward if choice == self.collective_choice else self.normal_reward)

            if reward_mode == "additive":
                delta = (factor - 1.0) * additive_base
                capital_after = capital_before + delta
            else:
                capital_after = capital_before * factor
                delta = capital_after - capital_before

            new_capitals.append(capital_after)
            rewards.append(delta)

        self.capitals = new_capitals
        explanation = (
            f"Collective choices: {collective_count}/{len(choices)} "
            f"({collective_ratio:.2%}). "
        )
        if reward_mode == "additive":
            def fmt_delta(f):
                return f"+{(float(f) - 1.0) * additive_base:.2f}"
            if cooperation_succeeded:
                if self.success_applies_to_all:
                    explanation += f"Cooperation succeeded! Everyone's capital adds {fmt_delta(self.high_reward)}."
                else:
                    explanation += (
                        f"Cooperation succeeded! Collective add {fmt_delta(self.high_reward)}, "
                        f"independent add {fmt_delta(self.normal_reward)}."
                    )
            else:
                explanation += (
                    f"Cooperation failed. Collective add {fmt_delta(self.low_reward)}, "
                    f"independent add {fmt_delta(self.normal_reward)}."
                )
        else:
            if cooperation_succeeded:
                if self.success_applies_to_all:
                    explanation += f"Cooperation succeeded! Everyone's capital multiplies by x{self.high_reward}."
                else:
                    explanation += (
                        f"Cooperation succeeded! Collective x{self.high_reward}, "
                        f"independent x{self.normal_reward}."
                    )
            else:
                explanation += (
                    f"Cooperation failed. Collective x{self.low_reward}, "
                    f"independent x{self.normal_reward}."
                )
        
        return rewards, explanation

    def run_single_round(self, agents, round_id: int) -> Dict[str, Any]:
        question = self.generate_question(round_id, self.history)
        k = max(0, min(self.fixed_cooperators_k, self.num_agents))
        if k > 0:
            if self.fixed_cooperators_strategy == "head":
                fixed_indices = list(range(k))
            else:
                fixed_indices = list(range(self.num_agents - k, self.num_agents))
        else:
            fixed_indices = []
        predict_indices = [i for i in range(self.num_agents) if i not in fixed_indices]

        messages_per_predict_agent = [
            self._prepare_agent_messages(question, agent_idx=i)
            for i in predict_indices
        ]

        agent_responses_full: List[str] = [""] * self.num_agents
        predicted_outputs: List[str] = []
        if len(predict_indices) > 0:
            if hasattr(agents, "predict_batch") and callable(getattr(agents, "predict_batch")):
                try:
                    predicted_outputs = agents.predict_batch(messages_per_predict_agent)
                except Exception:
                    predicted_outputs = []
                    for idx in predict_indices:
                        try:
                            response, _ = agents[idx].predict_message(self._prepare_agent_messages(question, agent_idx=idx))
                            predicted_outputs.append(response)
                        except Exception as se:
                            predicted_outputs.append(f"Error: {str(se)}")
            else:
                for idx in predict_indices:
                    try:
                        response, _ = agents[idx].predict_message(self._prepare_agent_messages(question, agent_idx=idx))
                        predicted_outputs.append(response)
                    except Exception as e:
                        predicted_outputs.append(f"Error: {str(e)}")
        pi = 0
        for i in range(self.num_agents):
            if i in fixed_indices:
                agent_responses_full[i] = f'{{"choice": "{self.collective_choice}"}}'
            else:
                agent_responses_full[i] = predicted_outputs[pi] if pi < len(predicted_outputs) else ""
                pi += 1

        capitals_before = self.capitals.copy()
        rewards, outcome_explanation = self.calculate_rewards(agent_responses_full, round_id, self.history)
        agent_choices = [self._extract_choice(r) for r in agent_responses_full]
        capitals_after = self.capitals.copy()

        round_data = {
            "question": question,
            "agent_responses": agent_responses_full,
            "agent_choices": agent_choices,
            "rewards": rewards,
            "reward_is_delta": True,
            "capitals_before": capitals_before,
            "capitals_after": capitals_after,
            "collective_outcome": outcome_explanation,
            "round_id": round_id,
            "fixed_cooperators_indices": fixed_indices,
            "predicted_indices": predict_indices,
        }

        self.history.add_round(round_data)
        return round_data

    def get_evolution_summary(self) -> Dict[str, Any]:
        base = super().get_evolution_summary()
        base.update({
            "start_capital": self.start_capital,
            "final_capitals": self.capitals,
            "total_wealth": sum(self.capitals),
        })
        return base

    def get_system_prompt(self) -> str:
        base_prompt = self.config.get(
            "system_prompt",
            (
                f"You are participating in a cooperative investment game. You can choose between "
                f"'{self.collective_choice}' (helps everyone if enough choose it) or "
                f"'{self.independent_choice}' (safer for you individually)."
            ),
        )

        mode = str(self.config.get("reward_mode", "multiplicative")).lower()
        if mode == "additive":
            base = float(self.config.get("additive_base", self.start_capital))
            rules = (
                "\n\nInvestment rules: Rewards are additive increments based on a fixed base per round, "
                f"not compounding. Each round adds (factor−1)×{base} to your capital. The 'Your reward' feedback each round "
                "is the change in your capital for that round (delta = new_capital - old_capital). "
                "Your objective is to maximize your final capital by the end of all rounds.\n\n"
                f"IMPORTANT: End with JSON: {{\"choice\": \"{self.collective_choice}\"}} or "
                f"{{\"choice\": \"{self.independent_choice}\"}}."
            )
        else:
            rules = (
                "\n\nInvestment rules: Rewards are multiplicative returns applied to your current capital, "
                "so returns compound across rounds (path-dependent). The 'Your reward' feedback each round "
                "is the change in your capital for that round (delta = new_capital - old_capital), which depends on your "
                "previous capital. Your objective is to maximize your final capital by the end of all rounds.\n\n"
                f"IMPORTANT: End with JSON: {{\"choice\": \"{self.collective_choice}\"}} or "
                f"{{\"choice\": \"{self.independent_choice}\"}}."
            )
        
        return base_prompt + rules

    def generate_question(self, round_id: int, history: MultiAgentHistoryRecord) -> str:
        """Return the base question. No extra note to avoid duplication with templates."""
        return super().generate_question(round_id, history)


def config_from_investment_dataset_item(
    item: Dict[str, Any],
    max_rounds: int = 10,
    history_window: int = 3,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a dataset item into an investment-based cooperative environment config.

    Interprets reward values as multiplicative factors applied to capitals.
    """
    params = item.get("params", {})
    template = item.get("template", {})

    n = int(params.get("n") or item.get("num_agents") or 5)
    coop_min = int(params.get("cooperation_threshold") or item.get("cooperation_threshold") or max(1, int(0.5 * n)))
    high_reward = float(params.get("high_reward", 1.25))
    normal_reward = float(params.get("normal_reward", 0.9))
    low_reward = float(params.get("low_reward", 0.5))
    start_capital = float(params.get("start_capital", 1.0))
    success_applies_to_all = bool(params.get("success_applies_to_all", True))

    if overrides:
        if "num_agents" in overrides and overrides["num_agents"]:
            try:
                n = int(overrides["num_agents"])
            except Exception:
                pass
        if "cooperation_threshold" in overrides and overrides["cooperation_threshold"] is not None:
            try:
                ct_val = float(overrides["cooperation_threshold"])
                if ct_val <= 1.0:
                    import math
                    coop_min = int(math.ceil(ct_val * n))
                else:
                    coop_min = int(ct_val)
            except Exception:
                pass
        try:
            if "high_reward" in overrides and overrides["high_reward"] is not None:
                high_reward = float(overrides["high_reward"])
        except Exception:
            pass
        try:
            if "normal_reward" in overrides and overrides["normal_reward"] is not None:
                normal_reward = float(overrides["normal_reward"])
        except Exception:
            pass
        try:
            if "low_reward" in overrides and overrides["low_reward"] is not None:
                low_reward = float(overrides["low_reward"])
        except Exception:
            pass
        try:
            if "start_capital" in overrides and overrides["start_capital"] is not None:
                start_capital = float(overrides["start_capital"])
        except Exception:
            pass
        try:
            if "success_applies_to_all" in overrides and overrides["success_applies_to_all"] is not None:
                success_applies_to_all = bool(overrides["success_applies_to_all"])
        except Exception:
            pass

    cooperation_threshold = coop_min / float(n)

    fmt = {
        "n": n,
        "cooperation_threshold": coop_min,
        "high_reward": high_reward,
        "normal_reward": normal_reward,
        "low_reward": low_reward,
        "delta_high": round((high_reward - 1.0) * start_capital, 2),
        "delta_normal": round((normal_reward - 1.0) * start_capital, 2),
        "delta_low": round((low_reward - 1.0) * start_capital, 2),
    }

    collective_choice = template.get("collective_choice") or item.get("cooperative_choice", "collective")
    independent_choice = template.get("independent_choice") or item.get("independent_choice", "independent")

    system_prompt = None
    base_question = None
    if template:
        if template.get("system_prompt"):
            try:
                system_prompt = template["system_prompt"].format(**fmt)
            except Exception:
                system_prompt = template["system_prompt"]
        reward_mode = "multiplicative"
        if overrides and "reward_mode" in overrides and overrides["reward_mode"]:
            reward_mode = str(overrides["reward_mode"]).lower()
        elif "reward_mode" in item:
            reward_mode = str(item["reward_mode"]).lower()
        prompt_src = None
        if reward_mode == "multiplicative" and template.get("user_prompt_mult"):
            prompt_src = template.get("user_prompt_mult")
        elif reward_mode == "additive" and template.get("user_prompt_add"):
            prompt_src = template.get("user_prompt_add")
        else:
            prompt_src = template.get("user_prompt") or template.get("question")
        if prompt_src:
            try:
                base_question = prompt_src.format(**fmt)
                base_question = "This is a new round.\n" + base_question
            except Exception:
                base_question = prompt_src

    if system_prompt is None:
        system_prompt = item.get("system_prompt")
    if base_question is None:
        base_question = item.get("user_prompt") or item.get("question") or "Choose your strategy for this round."

    config: Dict[str, Any] = {
        "environment_type": "cooperative_investment",
        "environment_name": item.get("environment") or item.get("environment_name", "DatasetInvestmentCoop"),
        "num_agents": n,
        "max_rounds": max_rounds,
        "cooperation_threshold": cooperation_threshold,
        "high_reward": high_reward,
        "normal_reward": normal_reward,
        "low_reward": low_reward,
        "collective_choice": collective_choice,
        "independent_choice": independent_choice,
        "system_prompt": system_prompt,
        "base_question": base_question,
        "choice_explanation": "",
        "merge_prompts": True,
        "reward_mode": (overrides or {}).get("reward_mode", "multiplicative"),
        "additive_base": (overrides or {}).get("additive_base", start_capital),
        "history_window": history_window,
        "start_capital": start_capital,
        "success_applies_to_all": success_applies_to_all,
    }

    if overrides:
        config.update(overrides)
    return config
