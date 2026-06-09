uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${AIRCRAFT_DATA_ROOT}" --data.dataset_name "FGVCAircraft"

uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${CARS_DATA_ROOT}" --data.dataset_name "StanfordCars"

uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${CUB_DATA_ROOT}" --data.dataset_name "CUB-200-2011"
