"""Example usage of learning MCP server."""

import asyncio
from learning_mcp.server import create_server
from learning_mcp.models import TrainingRequest


async def main():
    """Example usage of learning MCP server."""
    server = create_server()
    
    print("Learning MCP server created successfully!")
    print("\nAvailable capabilities:")
    print("- Offline training on supervisor aggregates")
    print("- Decision profile generation")
    print("- Pattern analysis (successful/failed)")
    print("- Profile comparison and validation")
    
    # Example training request
    training_request = TrainingRequest(
        window="7d",
        min_samples=100,
        focus_metric="success_rate",
        constraints={"max_complexity": 10}
    )
    
    print(f"\nExample training request:")
    print(f"  Window: {training_request.window}")
    print(f"  Min samples: {training_request.min_samples}")
    print(f"  Focus metric: {training_request.focus_metric}")
    
    print("\nServer ready to process training requests!")


if __name__ == "__main__":
    asyncio.run(main())
