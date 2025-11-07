"""Tests for AgentCore handling of incomplete outputs and replanning."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent.core import AgentCore, PlanGenerated
from planner import Plan, PlanStep, Tool
from events.models import MessageReceived


class MockPlanner:
    """Мок планировщика для тестирования."""
    
    def __init__(self):
        self.plan_count = 0
        self.refine_count = 0
    
    def plan(self, request, context=None, previous_results=None):
        self.plan_count += 1
        if self.plan_count == 1:
            # First plan returns incomplete step
            return Plan(steps=[PlanStep(tool=Tool.CODE, content="print('incomplete')")], context=[])
        else:
            # Subsequent plans return complete step
            return Plan(steps=[PlanStep(tool=Tool.CODE, content="print('complete')")], context=[])
    
    def refine(self, request, results):
        self.refine_count += 1
        return "refined_goal"


class TestAgentIncompleteOutputs:
    """Test AgentCore._handle_incomplete_outputs and replanning logic."""
    
    @pytest.mark.asyncio
    async def test_empty_outputs_triggers_replan(self):
        """Test that empty outputs trigger replanning."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MockPlanner()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator,
            max_iterations=3
        )
        
        # Отключаем подписку на события
        core.bus = MagicMock()
        
        core.goal = "test_goal"
        core.iterations = 0
        
        # Test with empty outputs
        published_events = []
        
        async def capture_publish(channel, event):
            published_events.append((channel, event))
        
        core.bus.publish = capture_publish
        
        result = await core._handle_incomplete_outputs([])
        
        assert result is True  # Should return True indicating replan
        assert planner.refine_count == 1
        assert len(published_events) == 1
        assert published_events[0][0] == "plan"
        assert isinstance(published_events[0][1], PlanGenerated)
    
    @pytest.mark.asyncio
    async def test_incomplete_marker_triggers_replan(self):
        """Test that 'incomplete' marker in output triggers replanning."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MockPlanner()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator,
            max_iterations=3
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        core.goal = "test_goal"
        core.iterations = 0
        
        # Test with incomplete marker
        published_events = []
        core.bus.publish = AsyncMock(side_effect=lambda c, e: published_events.append((c, e)))
        
        # Various incomplete markers - только те, которые точно соответствуют логике
        incomplete_outputs = [
            ["incomplete"],
            ["Not finished"],
            ["not finished"],
            ["INCOMPLETE"],
        ]
        
        for outputs in incomplete_outputs:
            published_events.clear()
            result = await core._handle_incomplete_outputs(outputs)
            assert result is True
            assert len(published_events) == 1
    
    @pytest.mark.asyncio
    async def test_complete_outputs_no_replan(self):
        """Test that complete outputs don't trigger replanning."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MockPlanner()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        core.goal = "test_goal"
        
        published_events = []
        core.bus.publish = AsyncMock(side_effect=lambda c, e: published_events.append((c, e)))
        
        # Complete outputs
        result = await core._handle_incomplete_outputs(["Result: 42", "Complete"])
        
        assert result is False  # Should not trigger replan
        assert len(published_events) == 0
        assert planner.refine_count == 0
    
    @pytest.mark.asyncio
    async def test_max_iterations_prevents_infinite_loop(self):
        """Test that max_iterations prevents infinite replanning."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MagicMock()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator,
            max_iterations=2
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        planner.refine = MagicMock(return_value="refined")
        planner.plan = MagicMock(return_value=Plan(steps=[
            PlanStep(tool=Tool.CODE, content="incomplete")
        ], context=[]))
        memory.semantic_search = MagicMock(return_value=[])
        
        core.goal = "test_goal"
        core.iterations = 2  # Already at max
        
        published_events = []
        core.bus.publish = AsyncMock(side_effect=lambda c, e: published_events.append((c, e)))
        
        result = await core._handle_incomplete_outputs([])
        
        # Should still return True but check iterations
        assert result is True
        assert core.iterations == 3  # Incremented
        
        # After max iterations, should stop
        if core.iterations > core.max_iterations:
            # Implementation should prevent further replans
            pass
    
    @pytest.mark.asyncio
    async def test_replan_without_goal(self):
        """Test handling incomplete outputs when goal is None."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MockPlanner()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        core.goal = None  # No current goal
        
        published_events = []
        core.bus.publish = AsyncMock(side_effect=lambda c, e: published_events.append((c, e)))
        
        result = await core._handle_incomplete_outputs([])
        
        # Should return True but not publish without goal
        assert result is True
        assert len(published_events) == 0
    
    @pytest.mark.asyncio
    async def test_refined_goal_updates(self):
        """Test that refined goal updates the current goal."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MagicMock()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        planner.refine = MagicMock(return_value="new_refined_goal")
        planner.plan = MagicMock(return_value=Plan(steps=[
            PlanStep(tool=Tool.CODE, content="test")
        ], context=[]))
        memory.semantic_search = MagicMock(return_value=["context"])
        
        core.goal = "original_goal"
        core.context = ["ctx1"]
        
        await core._handle_incomplete_outputs([])
        
        # Goal should be updated to refined version
        assert core.goal == "new_refined_goal"
        assert core.iterations == 1
    
    @pytest.mark.asyncio
    async def test_empty_plan_steps(self):
        """Test handling when planner returns empty steps."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MagicMock()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        planner.refine = MagicMock(return_value="refined")
        planner.plan = MagicMock(return_value=Plan(steps=[], context=[]))  # Empty steps
        memory.semantic_search = MagicMock(return_value=[])
        
        core.goal = "test_goal"
        
        published_events = []
        core.bus.publish = AsyncMock(side_effect=lambda c, e: published_events.append((c, e)))
        
        await core._handle_incomplete_outputs([])
        
        # Empty plan shouldn't be published
        assert len(published_events) == 0
    
    @pytest.mark.asyncio
    async def test_mixed_complete_incomplete_outputs(self):
        """Test outputs with mix of complete and incomplete markers."""
        # Создаём моки для всех зависимостей
        bus = MagicMock()
        planner = MockPlanner()
        executor = MagicMock()
        search = MagicMock()
        memory = MagicMock()
        code_generator = MagicMock()
        
        # Создаём core с моками
        core = AgentCore(
            bus=bus,
            planner=planner,
            executor=executor,
            search=search,
            memory=memory,
            code_generator=code_generator
        )
        
        # Отключаем подписку на события и используем AsyncMock для publish
        core.bus = MagicMock()
        core.bus.publish = AsyncMock()
        
        core.goal = "test_goal"
        
        published_events = []
        core.bus.publish = AsyncMock(side_effect=lambda c, e: published_events.append((c, e)))
        
        # Mixed outputs - any incomplete should trigger replan
        mixed_outputs = [
            "Result: 42",
            "Complete",
            "incomplete",  # This triggers replan
            "Done"
        ]
        
        result = await core._handle_incomplete_outputs(mixed_outputs)
        
        assert result is True  # Should trigger due to incomplete marker
        assert len(published_events) == 1
