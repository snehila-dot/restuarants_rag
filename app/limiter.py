"""Rate limiter configuration for the application."""

from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory rate limiter, keyed by client IP address.
# For production with multiple workers, swap to a Redis backend:
#   limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
limiter = Limiter(key_func=get_remote_address)
