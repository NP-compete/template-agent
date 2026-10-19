"""Bound the number of runs a pod will start at once.

Without a bound the pod accepts every request it is handed and finds
out it cannot serve them only when the connection pools are gone. The
failure then lands on requests that were already half-served —
someone three minutes into a long turn — rather than on the request
that should never have been admitted. Shedding at the door is worse
for the marginal user and much better for everyone already in flight.

This is not rate limiting, which lives in the gateway and is about
per-user fairness. The gateway cannot do this job: it knows how many
requests it has sent, not how many connections this pod has left.

Only run-starting requests are gated. Reads, health and the rest of
the API are cheap and must keep answering — a pod at capacity still
has to tell the probes it is alive and let the UI poll a run that is
already going.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# POST /runs, POST /threads/{id}/runs, and their streaming variants.
_RUN_PATH = re.compile(r"^/(?:threads/[^/]+/)?runs(?:/(?:stream|wait))?/?$")


class AdmissionControlMiddleware(BaseHTTPMiddleware):
    """Reject new runs past a concurrency limit instead of queueing them."""

    def __init__(self, app: Any, max_concurrent: int) -> None:
        """Gate run starts at ``max_concurrent``; zero disables the gate."""
        super().__init__(app)
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        if self._sem is not None:
            logger.info("Admission control: max %d concurrent runs", max_concurrent)

    def _gated(self, request: Request) -> bool:
        return request.method == "POST" and bool(_RUN_PATH.match(request.url.path))

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Admit the request, or shed it with 503 when at capacity."""
        if self._sem is None or not self._gated(request):
            return await call_next(request)

        if self._sem.locked():
            # Fail now rather than wait. A caller queued behind a long
            # turn learns nothing by waiting and holds a connection
            # while it does; a 503 with Retry-After lets the gateway
            # try another pod immediately.
            logger.warning(
                "Shedding run: %d concurrent runs already in flight",
                self.max_concurrent,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
                content={
                    "detail": (
                        "The agent is at capacity. Please retry in a few seconds."
                    ),
                    "error_type": "at_capacity",
                },
            )

        async with self._sem:
            return await call_next(request)
