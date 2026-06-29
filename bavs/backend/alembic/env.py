import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Ensure the backend/ directory (containing the `app` package) is
# importable regardless of the working directory `alembic` is invoked
# from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402

# Import every model module so its table is registered on Base.metadata
# before autogenerate runs. This list must be kept in sync with
# app/db/models/ -- a model not imported here will be invisible to
# `alembic revision --autogenerate` and silently excluded from migrations.
from app.db.models import (  # noqa: E402, F401
    user,
    kyc_case,
    document,
    ocr_extraction,
    tampering_result,
    face_verification_result,
    cross_document_result,
    liveness_result,
    fraud_risk_score,
    audit_log,
    chat,
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the static sqlalchemy.url from alembic.ini with our app's
# actual DATABASE_URL (sourced from environment variables via
# app.config.Settings), so migrations always target the same database
# the application itself connects to -- never a stale hardcoded URL.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
