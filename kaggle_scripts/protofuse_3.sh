
uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${CUB_DATA_ROOT}" --data.dataset_name "CUB-200-2011"

uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${DTD_DATA_ROOT}" --data.dataset_name "DTD"

uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${EUROSAT_DATA_ROOT}" --data.dataset_name "EuroSAT"