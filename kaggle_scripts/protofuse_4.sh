
uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${UCF_DATA_ROOT}" --data.dataset_name "UCF101"

uv run protofuse_5shot_5seed.py --config configs/protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101"
