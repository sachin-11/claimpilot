.PHONY: up down logs test lint fmt migrate seed revision shell

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f backend

# Database tests need postgres (they use a separate "<db>_test" database).
test:
	docker compose run --rm backend pytest

lint:
	docker compose run --rm --no-deps backend ruff check .

fmt:
	docker compose run --rm --no-deps backend sh -c "ruff format . && ruff check --fix ."

migrate:
	docker compose exec backend alembic upgrade head

seed:
	docker compose exec backend python -m app.scripts.seed

# usage: make revision m="add claims table"
revision:
	docker compose exec backend alembic revision --autogenerate -m "$(m)"

shell:
	docker compose exec backend bash
