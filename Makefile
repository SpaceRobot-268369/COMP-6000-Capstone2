.PHONY: help branch push pull repro diff status ai

help:
	@echo "git+dvc commands:"
	@echo "  make branch b=<name>   checkout branch and sync DVC data"
	@echo "  make push              git push + dvc push"
	@echo "  make pull              git pull + dvc pull"
	@echo "  make repro             re-run changed DVC pipeline stages"
	@echo "  make diff              git diff + dvc params diff"
	@echo "  make status            git status + dvc status"
	@echo ""
	@echo "AI server:"
	@echo "  make ai                start AI server locally (port 8000)"

branch:
	git checkout $(b) && python3 -m dvc checkout

push:
	git push && python3 -m dvc push

pull:
	git pull && python3 -m dvc pull

repro:
	python3 -m dvc repro

diff:
	git diff && python3 -m dvc params diff

status:
	git status && python3 -m dvc status

ai:
	cd acoustic_ai && uvicorn server.server:app --reload --port 8000
