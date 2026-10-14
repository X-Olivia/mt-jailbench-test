"""
X-Teaming outcome evaluator: scores target responses with a judge (xteaming 1-5 scale).
Identical to CrescendoOutcomeEvaluator — both use the same xteaming judge type.
"""
import json
import logging
import os
from typing import override

import yaml

from client.unified_llm_client import UnifiedLLMClient

from ...core import AttackContext, TurnEvaluation
from ...interfaces.attack.outcome_evaluator import OutcomeEvaluator

logger = logging.getLogger(__name__)

JUDGE_RETRIES = 3


def judge_score_1_to_5(client: UnifiedLLMClient, prompt: str) -> tuple[int, str]:
    """
    Ask the judge for a 1-5 score, retrying on unparseable or out-of-range output.
    If the judge never gives a valid score, count the attempt as a failure (score 1)
    instead of crashing the whole behavior, which would drop it from the ASR denominator.
    """
    last = None
    for _ in range(JUDGE_RETRIES):
        raw, _ = client.generate(prompt, response_format={"type": "json_object"})
        last = raw
        try:
            result = json.loads(raw)
            score = int(result["score"])
        except (TypeError, ValueError, KeyError):
            continue
        if 1 <= score <= 5:
            return score, result.get("reason") or ""
    logger.warning(f"Judge gave no valid 1-5 score after {JUDGE_RETRIES} tries; scoring 1. Last output: {str(last)[:200]}")
    return 1, f"INVALID_JUDGE_OUTPUT: {str(last)[:200]}"


class XTeamingOutcomeEvaluator(OutcomeEvaluator):
   
    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        jc = config.get("judge_model", {})
        self.client = UnifiedLLMClient(
            model=jc["model"],
            provider=jc.get("provider"),
            base_url=jc.get("base_url"),
        )
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        path = os.path.join(prompts_dir, "evaluation_prompt.yaml")
        with open(path) as f:
            self.template = yaml.safe_load(f)["prompt"]

    @override
    def evaluate(self, context: AttackContext) -> TurnEvaluation:
        attempt = context.get_current_attempt()
        response = attempt.response or ""
        behavior = context.harmful_behavior

        if not response.strip():
            raise RuntimeError("No response found")
        
        prompt = self._build_judge_prompt(behavior, response)
        score, reason = judge_score_1_to_5(self.client, prompt)
        return TurnEvaluation(score=score, reason=reason)

    def _build_judge_prompt(self, behavior: str, response: str) -> str:
        """
        Formats judge prompt by replacing `{behavior}` and `{response}` in template.
        """
        return self.template.format(
            behavior=behavior.strip(),
            response=response.strip()
        )