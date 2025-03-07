# app/routes/__init__.py

from .weekly import weekly_bp
from .monthly import monthly_bp
from .daily import daily_bp

__all__ = ["weekly_bp", "monthly_bp", "daily_bp"]
