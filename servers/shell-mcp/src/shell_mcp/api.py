"""FastAPI integration for shell-mcp HTTP server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from .config import get_settings

logger = logging.getLogger(__name__)

# Pydantic models for requests
class RunCodeSimpleRequest(BaseModel):
    """Модель запроса для запуска кода."""
    code: Optional[str] = Field(None, description="Исходный код для выполнения")
    language: str = Field("python", description="Язык выполнения (python/bash/node)")
    image: Optional[str] = Field(None, description="Образ Docker")
    command: Optional[str] = Field(None, description="Кастомная команда")
    timeout_seconds: int = Field(120, description="Таймаут выполнения (сек)")
    network_enabled: Optional[bool] = Field(None, description="Разрешить сетевой доступ")
    env: Optional[List[str]] = Field(None, description="Переменные окружения KEY=VALUE")
    memory: Optional[str] = Field(None, description="Лимит памяти (256m, 1g)")
    cpus: Optional[str] = Field(None, description="Лимит CPU (0.5, 1.0)")
    readonly_fs: Optional[bool] = Field(None, description="Только чтение файловой системы")
    dependencies: Optional[List[str]] = Field(None, description="Зависимости для pip")
    out_artifacts_path: Optional[str] = Field(None, description="Путь для артефактов")
    script_path: Optional[str] = Field(None, description="Путь к существующему скрипту")
    save_name: Optional[str] = Field(None, description="Сохранить как")


class RunSavedScriptRequest(BaseModel):
    """Модель запроса для запуска сохранённого скрипта."""
    name: str = Field(..., description="Имя сохранённого скрипта")
    timeout_seconds: int = Field(120, description="Таймаут выполнения (сек)")
    network_enabled: Optional[bool] = Field(None, description="Разрешить сетевой доступ")
    env: Optional[List[str]] = Field(None, description="Переменные окружения")
    memory: Optional[str] = Field(None, description="Лимит памяти")
    cpus: Optional[str] = Field(None, description="Лимит CPU")
    readonly_fs: Optional[bool] = Field(None, description="Только чтение ФС")
    dependencies: Optional[List[str]] = Field(None, description="Зависимости")
    out_artifacts_path: Optional[str] = Field(None, description="Путь для артефактов")
    save_name: Optional[str] = Field(None, description="Сохранить как")


class RunCodeMultiStepRequest(BaseModel):
    """Модель запроса для многозадачного выполнения."""
    steps: List[str] = Field(..., description="Список Python кода для последовательного выполнения")
    language: str = Field("python", description="Язык (только python)")
    image: Optional[str] = Field(None, description="Образ Docker")
    command: Optional[str] = Field(None, description="Кастомная команда")
    timeout_seconds: int = Field(120, description="Таймаут (секунды)", ge=1)
    network_enabled: Optional[bool] = Field(None, description="Разрешить сетевой доступ")
    env: Optional[List[str]] = Field(None, description="Переменные окружения KEY=VALUE")
    memory: Optional[str] = Field(None, description="Лимит памяти (256m, 1g)")
    cpus: Optional[str] = Field(None, description="Лимит CPU (0.5, 1.0)")
    readonly_fs: Optional[bool] = Field(None, description="Только чтение ФС")
    dependencies: Optional[List[str]] = Field(None, description="Зависимости для pip")
    out_artifacts_path: Optional[str] = Field(None, description="Путь для артефактов")
    save_name: Optional[str] = Field(None, description="Сохранить под именем")


class RunSavedArgsModel(BaseModel):
    """Модель аргументов для запуска сохранённого скрипта."""
    name: str = Field(..., description="Имя сохранённого скрипта")
    timeout_seconds: int = Field(120, description="Таймаут (сек)", ge=1)
    network_enabled: Optional[bool] = Field(None, description="Разрешить сетевой доступ")
    env: Optional[List[str]] = Field(None, description="Переменные окружения KEY=VALUE")
    memory: Optional[str] = Field(None, description="Лимит памяти")
    cpus: Optional[str] = Field(None, description="Лимит CPU")
    readonly_fs: Optional[bool] = Field(None, description="Только чтение ФС")
    dependencies: Optional[List[str]] = Field(None, description="Зависимости")
    out_artifacts_path: Optional[str] = Field(None, description="Путь для артефактов")
    save_name: Optional[str] = Field(None, description="Сохранить как")


class RunScriptsBatchRequest(BaseModel):
    """Модель запроса для батчевого запуска скриптов."""
    scripts: List[RunSavedArgsModel] = Field(..., description="Список скриптов для последовательного запуска")


class SaveTempScriptRequest(BaseModel):
    """Модель запроса для сохранения временного скрипта."""
    name: str = Field(..., description="Имя скрипта")
    language: str = Field(..., description="Язык (python/bash/node)")
    code: str = Field(..., description="Исходный код")


class PromoteTempScriptRequest(BaseModel):
    """Модель запроса для промоции временного скрипта."""
    slug: str = Field(..., description="Слаг временного скрипта")


class DeleteScriptRequest(BaseModel):
    """Модель запроса для удаления скрипта."""
    name: str = Field(..., description="Имя скрипта")


class DeleteScriptsRequest(BaseModel):
    """Модель запроса для батчевого удаления скриптов."""
    names: List[str] = Field(..., description="Список имён скриптов для удаления", min_length=1)


class PromoteTempScriptsRequest(BaseModel):
    """Модель запроса для батчевой промоции временных скриптов."""
    slugs: List[str] = Field(..., description="Список слогов временных скриптов", min_length=1)


def create_app() -> FastAPI:
    """Create and configure FastAPI application for HTTP transport."""
    settings = get_settings()
    
    app = FastAPI(
        title="Shell MCP",
        version="0.1.0",
        description="MCP сервер для выполнения кода в изолированных Docker контейнерах"
    )
    
    # Import tools
    from .tools import run as run_tools, meta as meta_tools
    
    # Initialize executor and store
    executor = run_tools.DockerExecutor()
    store = run_tools.ScriptStore()
    s = get_settings()
    
    # Register endpoints
    @app.get("/health", operation_id="check_health")
    async def health() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "name": "shell-mcp",
            "version": "0.1.0"
        }
    
    @app.get("/version", operation_id="get_version")
    async def version() -> dict:
        """Return server version and configuration."""
        import importlib.metadata as _meta
        try:
            ver = _meta.version("shell-mcp")
        except Exception:
            ver = "0.0.0"
        return {
            "name": "shell-mcp",
            "version": ver,
            "defaults": {
                "image": s.DEFAULT_IMAGE,
                "workdir": s.CONTAINER_WORKDIR,
                "network": s.DEFAULT_NETWORK,
            },
        }
    
    @app.get("/check_docker", operation_id="check_docker")
    async def check_docker() -> dict:
        """Check Docker availability."""
        try:
            from .services.docker_executor import ensure_docker_available
            ensure_docker_available()
            return {"ok": True, "message": "ok"}
        except Exception as e:
            return {"ok": False, "message": str(e)}
    
    # Run code endpoints
    @app.post("/run_code_simple", operation_id="run_code")
    async def run_code_simple(request: RunCodeSimpleRequest) -> dict:
        """Execute code inside a Docker container."""
        import threading
        from .services.docker_executor import DockerExecutor
        
        # Create executor
        exec_instance = DockerExecutor()
        sem = threading.BoundedSemaphore(s.MAX_CONCURRENCY) if (s.MAX_CONCURRENCY and s.MAX_CONCURRENCY > 0) else None
        
        # Normalize env
        env_map = {}
        if request.env:
            for item in request.env:
                key, value = item.split("=", 1)
                env_map[key.strip()] = value
        
        # Get source code
        source_code = request.code
        if request.script_path:
            from pathlib import Path
            source_path = Path(request.script_path)
            source_code = source_path.read_text(encoding="utf-8")
        
        if not source_code:
            raise ValueError("No code provided to execute")
        
        # Determine effective limits
        from .tools.run import _choose_stricter_memory, _choose_stricter_cpus
        network_eff = (s.DEFAULT_NETWORK and request.network_enabled) if request.network_enabled is not None else s.DEFAULT_NETWORK
        mem_eff = _choose_stricter_memory(request.memory, s.MEMORY)
        cpus_eff = _choose_stricter_cpus(request.cpus, s.CPUS)
        readonly_eff = s.READONLY_FS or bool(request.readonly_fs)
        
        # Execute
        if sem:
            sem.acquire()
        try:
            result = exec_instance.run(
                code=source_code,
                language=request.language,
                image=request.image,
                command=request.command,
                network=network_eff,
                timeout=request.timeout_seconds,
                env=env_map,
                memory=mem_eff,
                cpus=cpus_eff,
                readonly_fs=readonly_eff,
                dependencies=request.dependencies,
                out_artifacts_path=request.out_artifacts_path,
            )
        finally:
            if sem:
                sem.release()
        
        result_dict = result.to_dict()
        
        # Save if requested
        if request.save_name and result.exit_code == 0:
            try:
                saved_meta = store.save(request.save_name, request.language, source_code)
                result_dict["saved_script"] = saved_meta
            except Exception as exc:
                result_dict["save_error"] = str(exc)
        
        return result_dict
    
    @app.get("/list_saved_scripts", operation_id="list_saved")
    async def list_saved_scripts() -> list:
        """Returns metadata for saved scripts."""
        return store.list()
    
    @app.post("/run_saved_script", operation_id="run_saved")
    async def run_saved_script(request: RunSavedScriptRequest) -> dict:
        """Run a previously saved script."""
        from .tools.run import RunCodeArgs
        import threading
        
        try:
            meta = store.get(request.name)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        
        exec_instance = DockerExecutor()
        sem = threading.BoundedSemaphore(s.MAX_CONCURRENCY) if (s.MAX_CONCURRENCY and s.MAX_CONCURRENCY > 0) else None
        
        run_args = RunCodeArgs(
            code=None,
            language=meta.get("language", "python"),
            image=None,
            command=None,
            timeout_seconds=request.timeout_seconds,
            network_enabled=request.network_enabled,
            env=request.env,
            memory=request.memory,
            cpus=request.cpus,
            readonly_fs=request.readonly_fs,
            dependencies=request.dependencies,
            out_artifacts_path=request.out_artifacts_path,
            script_path=meta.get("path"),
            save_name=request.save_name,
        )
        
        # Execute (simplified version of _execute)
        from .tools.run import _normalize_env, _choose_stricter_memory, _choose_stricter_cpus
        env_map = _normalize_env(run_args.env)
        
        network_eff = (s.DEFAULT_NETWORK and run_args.network_enabled) if run_args.network_enabled is not None else s.DEFAULT_NETWORK
        mem_eff = _choose_stricter_memory(run_args.memory, s.MEMORY)
        cpus_eff = _choose_stricter_cpus(run_args.cpus, s.CPUS)
        readonly_eff = s.READONLY_FS or bool(run_args.readonly_fs)
        
        if sem:
            sem.acquire()
        try:
            result = exec_instance.run(
                code=run_args.code or "",
                language=run_args.language,
                image=run_args.image,
                command=run_args.command,
                network=network_eff,
                timeout=run_args.timeout_seconds,
                env=env_map,
                memory=mem_eff,
                cpus=cpus_eff,
                readonly_fs=readonly_eff,
                dependencies=run_args.dependencies,
                out_artifacts_path=run_args.out_artifacts_path,
            )
        finally:
            if sem:
                sem.release()
        
        result_dict = result.to_dict()
        if run_args.save_name and result.exit_code == 0:
            try:
                saved_meta = store.save(run_args.save_name, run_args.language, "")
                result_dict["saved_script"] = saved_meta
            except Exception as exc:
                result_dict["save_error"] = str(exc)
        
        return result_dict
    
    @app.post("/run_multi_step", operation_id="run_multi_step")
    async def run_multi_step(request: RunCodeMultiStepRequest) -> dict:
        """Execute multiple Python snippets sequentially inside one container."""
        from .tools.run import RunCodeArgs, _execute
        lang = (request.language or "").strip().lower()
        if lang and lang != "python":
            raise ValueError("run_multi_step currently supports only Python")
        
        # Build wrapper code
        from .tools.run import _build_multi_step_wrapper
        wrapper_code = _build_multi_step_wrapper(request.steps)
        
        args = RunCodeArgs(
            code=wrapper_code,
            language="python",
            image=request.image,
            command=request.command,
            timeout_seconds=request.timeout_seconds,
            network_enabled=request.network_enabled,
            env=request.env,
            memory=request.memory,
            cpus=request.cpus,
            readonly_fs=request.readonly_fs,
            dependencies=request.dependencies,
            out_artifacts_path=request.out_artifacts_path,
            script_path=None,
            save_name=request.save_name,
        )
        return _execute(args)
    
    @app.post("/run_scripts_batch", operation_id="run_batch")
    async def run_scripts_batch(request: RunScriptsBatchRequest) -> list:
        """Запустить несколько сохранённых скриптов последовательно."""
        from .tools.run import RunSavedArgs, RunCodeArgs, _execute
        
        results = []
        for script_args_data in request.scripts:
            try:
                script_args = RunSavedArgs(**script_args_data.model_dump())
                meta = store.get(script_args.name)
                run_args = RunCodeArgs(
                    code=None,
                    language=meta.get("language", "python"),
                    image=None,
                    command=None,
                    timeout_seconds=script_args.timeout_seconds,
                    network_enabled=script_args.network_enabled,
                    env=script_args.env,
                    memory=script_args.memory,
                    cpus=script_args.cpus,
                    readonly_fs=script_args.readonly_fs,
                    dependencies=script_args.dependencies,
                    out_artifacts_path=script_args.out_artifacts_path,
                    script_path=meta.get("path"),
                    save_name=script_args.save_name,
                )
                payload = _execute(run_args)
                results.append({"name": script_args.name, "success": True, "result": payload})
            except Exception as exc:
                logger.exception(f"run_scripts_batch failure name={script_args_data.name}")
                results.append({"name": script_args_data.name, "success": False, "error": str(exc)})
        return results
    
    @app.post("/delete_saved_script", operation_id="delete_saved")
    async def delete_saved_script(request: DeleteScriptRequest) -> dict:
        """Удалить сохранённый скрипт по имени."""
        try:
            meta = store.delete(request.name)
            return meta
        except Exception as exc:
            raise ValueError(f"Failed to delete script '{request.name}': {exc}") from exc
    
    @app.post("/delete_saved_scripts", operation_id="delete_batch")
    async def delete_saved_scripts(request: DeleteScriptsRequest) -> list:
        """Удалить несколько сохранённых скриптов."""
        results = []
        for name in request.names:
            try:
                meta = store.delete(name)
                results.append({"name": name, "success": True, "metadata": meta})
            except KeyError:
                results.append({"name": name, "success": False, "error": "Script not found"})
            except Exception as exc:
                results.append({"name": name, "success": False, "error": str(exc)})
        return results
    
    @app.get("/index_existing_scripts", operation_id="index_scripts")
    async def index_existing_scripts() -> dict:
        """Проиндексировать существующие скрипты на диске."""
        try:
            indexed_files = store.index_existing_files()
            return {
                "success": True,
                "indexed_count": len(indexed_files),
                "indexed_files": indexed_files,
                "message": f"Successfully indexed {len(indexed_files)} files"
            }
        except Exception as exc:
            return {
                "success": False,
                "indexed_count": 0,
                "indexed_files": [],
                "message": f"Failed to index files: {exc}"
            }
    
    @app.post("/save_temp_script", operation_id="save_temp")
    async def save_temp_script(request: SaveTempScriptRequest) -> dict:
        """Сохранить временный скрипт (удерживается 3 дня)."""
        try:
            meta = store.save_temp(request.name, request.language, request.code)
            return {"success": True, "script": meta, "message": f"Temporary script '{request.name}' saved successfully"}
        except Exception as exc:
            return {"success": False, "script": None, "message": f"Failed to save temporary script '{request.name}': {exc}"}
    
    @app.get("/list_temp_scripts", operation_id="list_temp")
    async def list_temp_scripts() -> list:
        """Список текущих временных скриптов."""
        try:
            return store.list_temp()
        except Exception as exc:
            raise ValueError(f"Failed to list temporary scripts: {exc}") from exc
    
    @app.post("/promote_temp_script", operation_id="promote_temp")
    async def promote_temp_script(request: PromoteTempScriptRequest) -> dict:
        """Повысить временный скрипт до постоянного."""
        try:
            meta = store.promote_temp_to_permanent(request.slug)
            return {"success": True, "script": meta, "message": f"Temporary script '{request.slug}' promoted to permanent"}
        except KeyError:
            return {"success": False, "script": None, "message": f"Temporary script '{request.slug}' not found"}
        except Exception as exc:
            return {"success": False, "script": None, "message": f"Failed to promote temporary script '{request.slug}': {exc}"}
    
    @app.post("/promote_temp_scripts", operation_id="promote_batch")
    async def promote_temp_scripts(request: PromoteTempScriptsRequest) -> list:
        """Повысить несколько временных скриптов до постоянных."""
        results = []
        for slug in request.slugs:
            try:
                meta = store.promote_temp_to_permanent(slug)
                results.append({"slug": slug, "success": True, "script": meta})
            except KeyError:
                results.append({"slug": slug, "success": False, "message": "Temporary script not found"})
            except Exception as exc:
                results.append({"slug": slug, "success": False, "message": str(exc)})
        return results
    
    @app.post("/cleanup_old_temp_scripts", operation_id="cleanup_temp")
    async def cleanup_old_temp_scripts() -> dict:
        """Удалить временные скрипты старше 3 дней."""
        try:
            cleaned_scripts = store.cleanup_old_temp_scripts()
            return {
                "success": True,
                "cleaned_count": len(cleaned_scripts),
                "cleaned_scripts": cleaned_scripts,
                "message": f"Cleaned {len(cleaned_scripts)} old temporary scripts"
            }
        except Exception as exc:
            return {
                "success": False,
                "cleaned_count": 0,
                "cleaned_scripts": [],
                "message": f"Failed to cleanup old temporary scripts: {exc}"
            }
    
    @app.on_event("startup")
    async def startup() -> None:
        logger.info("Shell MCP FastAPI startup")
    
    @app.on_event("shutdown")
    async def shutdown() -> None:
        logger.info("Shell MCP FastAPI shutdown")
    
    logger.info("FastAPI app created successfully")
    return app

