"""MCP tools for Policy MCP."""

from fastmcp import FastMCP
from ..models import DecisionProfile, DecisionRequest
from ..services.profile_service import ProfileService


def register_tools(mcp: FastMCP, profile_service: ProfileService) -> None:
    """Register all MCP tools."""
    
    @mcp.tool()
    async def create_profile(
        profile_id: str,
        name: str,
        description: str,
        rules: dict = None,
        metadata: dict = None,
        active: bool = True
    ) -> dict:
        """Create a new decision profile."""
        profile = DecisionProfile(
            id=profile_id,
            name=name,
            description=description,
            rules=rules or {},
            metadata=metadata or {},
            active=active
        )
        result = await profile_service.create_profile(profile)
        return result.model_dump()
    
    @mcp.tool()
    async def get_profile(profile_id: str) -> dict:
        """Get a decision profile by ID."""
        profile = await profile_service.get_profile(profile_id)
        if not profile:
            return {"error": "Profile not found"}
        return profile.model_dump()
    
    @mcp.tool()
    async def list_profiles(active_only: bool = False) -> dict:
        """List all decision profiles."""
        profiles = await profile_service.list_profiles(active_only=active_only)
        return {
            "profiles": [p.model_dump() for p in profiles],
            "count": len(profiles)
        }
    
    @mcp.tool()
    async def update_profile(
        profile_id: str,
        name: str = None,
        description: str = None,
        rules: dict = None,
        metadata: dict = None,
        active: bool = None
    ) -> dict:
        """Update a decision profile."""
        profile = await profile_service.get_profile(profile_id)
        if not profile:
            return {"error": "Profile not found"}
        
        if name is not None:
            profile.name = name
        if description is not None:
            profile.description = description
        if rules is not None:
            profile.rules = rules
        if metadata is not None:
            profile.metadata = metadata
        if active is not None:
            profile.active = active
        
        result = await profile_service.update_profile(profile)
        return result.model_dump()
    
    @mcp.tool()
    async def delete_profile(profile_id: str) -> dict:
        """Delete a decision profile."""
        success = await profile_service.delete_profile(profile_id)
        return {"success": success}
    
    @mcp.tool()
    async def evaluate_decision(
        profile_id: str,
        context: dict,
        actor: str = None,
        correlation_id: str = None
    ) -> dict:
        """Evaluate a decision request under a policy."""
        request = DecisionRequest(
            profile_id=profile_id,
            context=context,
            actor=actor,
            correlation_id=correlation_id
        )
        result = await profile_service.evaluate(request)
        return result.model_dump()

