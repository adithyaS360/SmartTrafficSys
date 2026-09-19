"""
Alembic environment.

WHY MIGRATIONS INSTEAD OF create_all():

create_all() only ever CREATES tables. It cannot alter one that already exists.
The moment you have data and want to add a column - and you will, as soon as the
LSTM needs a feature the schema does not have - create_all() silently does
nothing and the application fails against a table that is missing the column.
Migrations record each schema change as a reviewable, ordered, reversible step.

This file resolves the database URL from config.yaml at runtime rather than from
alembic.ini, so migrations always target the same database the app uses, and no
password is ever written into a committed file. It also means `alembic upgrade
head` works unchanged when you move from SQLite to Postgres.

render_as_batch=True matters for SQLite: SQLite has no real ALTER TABLE, so
Alembic emulates it by creating a new table, copying the rows, and swapping it
in. Without this flag, any migration that alters a column fails on SQLite.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

import sqlalchemy as sa
from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project root importable regardless of where alembic is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.models import Base, UTCDateTime  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def render_item(type_, obj, autogen_context) -> object:
    """
    Teach autogenerate how to write our custom column types into a migration.

    WHY THIS IS NEEDED: by default Alembic renders a custom TypeDecorator using
    its fully-qualified name - 'src.database.models.UTCDateTime()' - but does NOT
    add the matching import to the generated file. The migration is then written
    successfully and fails at 'alembic upgrade' with a bare
    'NameError: name src is not defined'.

    Returning a rendered string here AND registering the import in
    autogen_context.imports produces a migration that actually runs. Any future
    custom type needs a branch added here too.
    """
    if type_ == "type" and isinstance(obj, UTCDateTime):
        autogen_context.imports.add("from src.database.models import UTCDateTime")
        return "UTCDateTime()"

    if type_ == "type" and isinstance(obj, sa.JSON):
        # Default rendering produces JSONB(astext_type=Text()) without importing
        # Text - the same missing-import failure as above, one type further down.
        # Rendering the variant explicitly keeps the migration portable: JSONB on
        # Postgres, plain JSON everywhere else.
        autogen_context.imports.add("import sqlalchemy as sa")
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "sa.JSON().with_variant(postgresql.JSONB(), 'postgresql')"

    return False  # fall back to Alembic's default rendering


def _database_url() -> str:
    """Resolve the URL from config.yaml, falling back to a local SQLite file."""
    try:
        return load_config().database_url()
    except Exception as exc:  # noqa: BLE001
        print(f"[alembic] Could not read config.yaml ({exc}); using data/traffic.db")
        return "sqlite:///data/traffic.db"


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting - useful for reviewing a change."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and apply migrations."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,          # detect column type changes
            compare_server_default=True,
            render_as_batch=True,       # required for ALTER on SQLite
            render_item=render_item,    # emit imports for our custom types
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
