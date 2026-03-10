"""
Token-bucket rate limiter using Redis.
Limits each IP to N requests per minute.
"""

import time
import os
import redis as redis_lib
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        try:
            self.redis = redis_lib.from_url(
                f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}",
                decode_responses=True,
            )
        except Exception:
            self.redis = None

    async def dispatch(self, request: Request, call_next):
        if self.redis is None:
            return await call_next(request)

        # Skip rate limiting for health/metrics
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        window = 60  # seconds

        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            count, _ = pipe.execute()

            if count > self.rpm:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Max 60 requests/minute."},
                    headers={"Retry-After": str(window)},
                )
        except Exception:
            pass  # Fail open if Redis is unavailable

        return await call_next(request)
