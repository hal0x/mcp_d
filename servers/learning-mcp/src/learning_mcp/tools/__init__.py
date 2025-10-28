"""MCP tools registration for learning server."""

from __future__ import annotations

from typing import Optional

from fastmcp import FastMCP

from ..models import DecisionProfile, TrainingRequest
from ..services.pattern_analyzer import PatternAnalyzer
from ..services.policy_client import PolicyClient
from ..services.supervisor_client import SupervisorClient
from ..services.trainer import TrainerService
from ..services.online_learning import OnlineLearningService


def register_tools(
    mcp: FastMCP,
    trainer_service: TrainerService,
    pattern_analyzer: PatternAnalyzer,
    policy_client: PolicyClient,
    supervisor_client: SupervisorClient,
    online_learning_service: OnlineLearningService,
) -> None:
    """Register all MCP tools."""
    
    # Training tools
    @mcp.tool()
    async def train_offline(
        window: str = "7d",
        min_samples: int = 100,
        focus_metric: str = "success_rate",
        constraints: dict = None
    ) -> dict:
        """Trains a decision profile on historical data from supervisor aggregates."""
        
        request = TrainingRequest(
            window=window,
            min_samples=min_samples,
            focus_metric=focus_metric,
            constraints=constraints
        )
        
        try:
            result = await trainer_service.train_offline(request)
            
            return {
                "success": True,
                "profile": result.profile.dict(),
                "training_duration": result.training_duration,
                "samples_used": result.samples_used,
                "validation_score": result.validation_score,
                "insights": result.insights
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    async def propose_profile(profile_data: dict, activate: bool = False) -> dict:
        """Proposes a new decision profile and synchronizes it with Policy MCP."""
        try:
            profile = DecisionProfile(**profile_data)
            policy_response = await policy_client.upsert_profile(profile, activate=activate)
            return {
                "success": True,
                "profile_id": profile.profile_id,
                "status": "activated" if activate else "proposed",
                "policy_response": policy_response,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    async def validate_profile(profile_data: dict, compare_active: bool = False) -> dict:
        """Validates a decision profile against current metrics and optionally compares with active policy."""
        try:
            profile = DecisionProfile(**profile_data)
            # Проверяем базовые требования
            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }
            
            # Проверка весов
            weight_sum = sum(profile.weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                validation_results["issues"].append(
                    f"Weights don't sum to 1.0 (sum={weight_sum:.3f})"
                )
                validation_results["valid"] = False
            
            # Проверка порогов
            if profile.thresholds.get("min_confidence", 0) < 0.5:
                validation_results["warnings"].append(
                    "Low confidence threshold may lead to unreliable decisions"
                )
            
            # Проверка ограничений рисков
            if not profile.risk_limits:
                validation_results["warnings"].append(
                    "No risk limits defined"
                )
            
            comparison: Optional[dict] = None
            if compare_active:
                try:
                    current = await policy_client.get_active_profile()
                    comparison = current.get("profile")
                except Exception:
                    comparison = None

            response = {
                "success": True,
                "validation": validation_results,
                "profile_id": profile.profile_id,
            }

            if comparison:
                response["active_profile"] = comparison

            return response
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # Pattern analysis tools
    @mcp.tool()
    async def analyze_successful_patterns(
        window: str = "7d",
        min_frequency: int = 5
    ) -> dict:
        """Analyzes and returns successful execution patterns from supervisor data."""
        
        try:
            facts = await supervisor_client.fetch_facts(window=window)
            patterns = pattern_analyzer.get_successful_patterns(facts)
            
            return {
                "success": True,
                "patterns": [
                    {
                        "id": p.pattern_id,
                        "type": p.pattern_type,
                        "description": p.description,
                        "frequency": p.frequency,
                        "confidence": p.confidence,
                        "conditions": p.conditions,
                        "outcomes": p.outcomes
                    }
                    for p in patterns
                ],
                "total_patterns": len(patterns)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    async def analyze_failed_patterns(
        window: str = "7d",
        min_frequency: int = 5
    ) -> dict:
        """Analyzes and returns failed execution patterns to avoid."""
        
        try:
            facts = await supervisor_client.fetch_facts(window=window)
            patterns = pattern_analyzer.get_failed_patterns(facts)
            
            return {
                "success": True,
                "patterns": [
                    {
                        "id": p.pattern_id,
                        "type": p.pattern_type,
                        "description": p.description,
                        "frequency": p.frequency,
                        "confidence": p.confidence,
                        "conditions": p.conditions,
                        "outcomes": p.outcomes
                    }
                    for p in patterns
                ],
                "total_patterns": len(patterns)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @mcp.tool()
    async def policy_health() -> dict:
        """Checks Policy MCP health endpoint."""
        try:
            health = await policy_client.health()
            return {"success": True, "health": health}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def list_policy_profiles(active_only: bool = False) -> dict:
        """Lists profiles registered in Policy MCP."""
        try:
            response = await policy_client.list_profiles(active_only=active_only)
            return {"success": True, **response}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def get_active_policy_profile(profile_id: Optional[str] = None) -> dict:
        """Fetches the active profile from Policy MCP."""
        try:
            response = await policy_client.get_active_profile(profile_id=profile_id)
            return {"success": True, **response}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Comparison tools
    @mcp.tool()
    async def compare_profiles(profile_a_id: str, profile_b_id: str) -> dict:
        """Compares two decision profiles and recommends the better one."""
        try:
            resp_a = await policy_client.get_active_profile(profile_id=profile_a_id)
            profile_a = resp_a.get("profile")
            resp_b = await policy_client.get_active_profile(profile_id=profile_b_id)
            profile_b = resp_b.get("profile")

            if not profile_a or not profile_b:
                return {"success": False, "error": "Profiles not found"}

            comparison = online_learning_service.compare_profiles(
                DecisionProfile(**profile_a),
                profile_b,
            )
            comparison.update({
                "profile_a_id": profile_a_id,
                "profile_b_id": profile_b_id,
            })
            return {"success": True, "comparison": comparison}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @mcp.tool()
    async def trigger_online_learning() -> dict:
        """Manually trigger online learning cycle and A/B evaluation."""
        try:
            result = await online_learning_service.run_cycle()
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Meta tools
    @mcp.tool()
    async def health() -> dict:
        """Checks the health status of learning MCP server."""
        import time
        
        return {
            "status": "healthy",
            "service": "learning-mcp",
            "version": "1.0.0",
            "capabilities": [
                "offline_training",
                "profile_generation",
                "pattern_analysis",
                "profile_comparison"
            ]
        }
    
    @mcp.tool()
    async def get_training_status() -> dict:
        """Returns current training status and statistics."""
        
        return {
            "success": True,
            "status": "idle",
            "last_training": None,
            "total_profiles_generated": 0,
            "active_patterns": 0
        }
