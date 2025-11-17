"""Profile service for managing decision profiles."""

from __future__ import annotations

from typing import AsyncContextManager, Callable, List, Optional, Dict, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..pydantic_models import DecisionProfile, DecisionRequest, DecisionResponse, ProfileVersion
from ..models.orm import DecisionProfileORM, ProfileVersionORM

SessionProvider = Callable[[], AsyncContextManager[AsyncSession]]


class ProfileService:
    """Service for managing decision profiles backed by PostgreSQL."""

    def __init__(self, session_provider: SessionProvider):
        self._session_provider = session_provider

    async def create_profile(self, profile: DecisionProfile) -> DecisionProfile:
        """Create a new decision profile."""
        async with self._session_provider() as session:
            record = DecisionProfileORM(
                id=profile.id,
                name=profile.name,
                description=profile.description,
                active=profile.active,
                version=profile.version,
                rules=profile.rules,
                metadata_json=profile.metadata,
            )
            session.add(record)
            session.add(self._make_version_record(record))
            await session.commit()
            await session.refresh(record)
            return self._to_domain(record)

    async def get_profile(self, profile_id: str) -> Optional[DecisionProfile]:
        """Get a decision profile by ID."""
        async with self._session_provider() as session:
            record = await session.get(DecisionProfileORM, profile_id)
            if record is None:
                return None
            return self._to_domain(record)

    async def list_profile_versions(self, profile_id: str) -> List[ProfileVersion]:
        """Return historical versions for a profile."""
        async with self._session_provider() as session:
            stmt = (
                select(ProfileVersionORM)
                .where(ProfileVersionORM.profile_id == profile_id)
                .order_by(ProfileVersionORM.created_at.desc())
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
        return [self._version_to_domain(record) for record in records]

    async def rollback_profile(
        self,
        profile_id: str,
        *,
        version: Optional[str] = None,
        version_id: Optional[int] = None,
    ) -> Optional[DecisionProfile]:
        """Rollback profile to a previous version."""
        if version is None and version_id is None:
            raise ValueError("Either version or version_id must be provided for rollback")

        async with self._session_provider() as session:
            stmt = select(ProfileVersionORM).where(ProfileVersionORM.profile_id == profile_id)
            if version_id is not None:
                stmt = stmt.where(ProfileVersionORM.id == version_id)
            if version is not None:
                stmt = stmt.where(ProfileVersionORM.version == version)

            version_record = await session.scalar(stmt)
            if version_record is None:
                return None

            payload: Dict[str, Any] = dict(version_record.payload or {})
            record = await session.get(DecisionProfileORM, profile_id)
            if record is None:
                return None

            record.name = payload.get("name", record.name)
            record.description = payload.get("description", record.description)
            record.active = payload.get("active", record.active)
            record.version = payload.get("version", record.version)
            record.rules = payload.get("rules", record.rules)
            record.metadata_json = payload.get("metadata", record.metadata_json)

            session.add(self._make_version_record(record))
            await session.commit()
            await session.refresh(record)
            return self._to_domain(record)

    async def get_active_profile(self, profile_id: Optional[str] = None) -> Optional[DecisionProfile]:
        """Get an active profile (optionally by ID)."""
        async with self._session_provider() as session:
            stmt = select(DecisionProfileORM).where(DecisionProfileORM.active.is_(True))
            if profile_id:
                stmt = stmt.where(DecisionProfileORM.id == profile_id)
            stmt = stmt.order_by(DecisionProfileORM.updated_at.desc())
            result = await session.execute(stmt)
            record = result.scalars().first()
            if record is None:
                return None
            return self._to_domain(record)

    async def list_profiles(self, active_only: bool = False) -> List[DecisionProfile]:
        """List all decision profiles."""
        async with self._session_provider() as session:
            stmt = select(DecisionProfileORM).order_by(DecisionProfileORM.created_at.asc())
            if active_only:
                stmt = stmt.where(DecisionProfileORM.active.is_(True))
            result = await session.execute(stmt)
            records = result.scalars().all()
            return [self._to_domain(record) for record in records]

    async def update_profile(self, profile: DecisionProfile) -> Optional[DecisionProfile]:
        """Update a decision profile (returns None if it does not exist)."""
        async with self._session_provider() as session:
            record = await session.get(DecisionProfileORM, profile.id)
            if record is None:
                return None

            record.name = profile.name
            record.description = profile.description
            record.active = profile.active
            record.version = profile.version
            record.rules = profile.rules
            record.metadata_json = profile.metadata

            session.add(self._make_version_record(record))
            await session.commit()
            await session.refresh(record)
            return self._to_domain(record)

    async def activate_profile(self, profile_id: str) -> Optional[DecisionProfile]:
        """Mark profile as active."""
        async with self._session_provider() as session:
            record = await session.get(DecisionProfileORM, profile_id)
            if record is None:
                return None

            # Deactivate other profiles to ensure single active profile.
            await session.execute(
                update(DecisionProfileORM)
                .where(DecisionProfileORM.id != profile_id)
                .values(active=False)
            )
            record.active = True
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return self._to_domain(record)

    async def delete_profile(self, profile_id: str) -> bool:
        """Delete a decision profile."""
        async with self._session_provider() as session:
            record = await session.get(DecisionProfileORM, profile_id)
            if record is None:
                return False
            await session.delete(record)
            await session.commit()
            return True

    async def configure_experiment(
        self, profile_id: str, experiment: str, weight: float, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DecisionProfile]:
        """Attach experiment metadata to profile."""
        async with self._session_provider() as session:
            record = await session.get(DecisionProfileORM, profile_id)
            if record is None:
                return None

            metadata_json = dict(record.metadata_json or {})
            experiments = dict(metadata_json.get("experiments", {}))
            experiments[experiment] = {
                "weight": weight,
                "metadata": metadata or {},
            }
            metadata_json["experiments"] = experiments
            record.metadata_json = metadata_json

            session.add(self._make_version_record(record))
            await session.commit()
            await session.refresh(record)
            return self._to_domain(record)

    async def list_experiments(self) -> Dict[str, Dict[str, Any]]:
        """List experiments per profile."""
        async with self._session_provider() as session:
            result = await session.execute(select(DecisionProfileORM))
            records = result.scalars().all()
        experiments: Dict[str, Dict[str, Any]] = {}
        for record in records:
            metadata_json = dict(record.metadata_json or {})
            if "experiments" in metadata_json:
                experiments[record.id] = metadata_json["experiments"]
        return experiments

    async def evaluate(self, request: DecisionRequest) -> DecisionResponse:
        """Evaluate a decision request against active profiles."""
        async with self._session_provider() as session:
            stmt = select(DecisionProfileORM).where(DecisionProfileORM.id == request.profile_id)
            result = await session.execute(stmt)
            record = result.scalars().first()

            if record is None or not record.active:
                return DecisionResponse(
                    profile_id=request.profile_id,
                    decision="deny",
                    reason="Profile not found or not active",
                    rules_applied=[],
                )

            rules_applied = list(record.rules.keys())

        return DecisionResponse(
            profile_id=request.profile_id,
            decision="allow",
            reason="All rules passed",
            rules_applied=rules_applied,
        )

    def _to_domain(self, record: DecisionProfileORM) -> DecisionProfile:
        """Convert ORM record to domain model."""
        return DecisionProfile(
            id=record.id,
            name=record.name,
            description=record.description,
            active=record.active,
            version=record.version,
            rules=dict(record.rules or {}),
            metadata=dict(record.metadata_json or {}),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _make_version_record(self, record: DecisionProfileORM) -> ProfileVersionORM:
        """Create snapshot record for version history."""
        snapshot = ProfileVersion(
            profile_id=record.id,
            version=record.version,
            payload={
                "id": record.id,
                "name": record.name,
                "description": record.description,
                "active": record.active,
                "version": record.version,
                "rules": record.rules,
                "metadata": record.metadata_json,
            },
        )
        return ProfileVersionORM(
            profile_id=snapshot.profile_id,
            version=snapshot.version,
            payload=snapshot.payload,
        )

    def _version_to_domain(self, record: ProfileVersionORM) -> ProfileVersion:
        return ProfileVersion(
            profile_id=record.profile_id,
            version=record.version,
            payload=dict(record.payload or {}),
            created_at=record.created_at,
        )
