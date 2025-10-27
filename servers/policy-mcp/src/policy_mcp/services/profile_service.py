"""Profile service for managing decision profiles."""

from typing import List, Optional
from ..models import DecisionProfile, DecisionRequest, DecisionResponse


class ProfileService:
    """Service for managing decision profiles."""
    
    def __init__(self):
        self._profiles: dict[str, DecisionProfile] = {}
    
    async def create_profile(self, profile: DecisionProfile) -> DecisionProfile:
        """Create a new decision profile."""
        self._profiles[profile.id] = profile
        return profile
    
    async def get_profile(self, profile_id: str) -> Optional[DecisionProfile]:
        """Get a decision profile by ID."""
        return self._profiles.get(profile_id)
    
    async def list_profiles(self, active_only: bool = False) -> List[DecisionProfile]:
        """List all decision profiles."""
        profiles = list(self._profiles.values())
        if active_only:
            profiles = [p for p in profiles if p.active]
        return profiles
    
    async def update_profile(self, profile: DecisionProfile) -> DecisionProfile:
        """Update a decision profile."""
        self._profiles[profile.id] = profile
        return profile
    
    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a decision profile."""
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            return True
        return False
    
    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate a decision request against active profiles."""
        profile = self._profiles.get(request.profile_id)
        
        if not profile or not profile.active:
            return DecisionResponse(
                profile_id=request.profile_id,
                decision="deny",
                reason="Profile not found or not active",
                rules_applied=[]
            )
        
        # Simple rule evaluation
        rules_applied = []
        decision = "allow"
        reason = "All rules passed"
        
        # In a real implementation, this would evaluate the actual rules
        for rule_name, rule_config in profile.rules.items():
            rules_applied.append(rule_name)
            # Evaluation logic here
        
        return DecisionResponse(
            profile_id=request.profile_id,
            decision=decision,
            reason=reason,
            rules_applied=rules_applied
        )

