uv run apt.py --config configs/apt_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 8 --data.seed 10000

uv run apt.py --config configs/apt_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 16 --data.seed 1

uv run apt.py --config configs/apt_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 16 --data.seed 10

uv run apt.py --config configs/apt_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 16 --data.seed 100

uv run apt.py --config configs/apt_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 16 --data.seed 1000

uv run apt.py --config configs/apt_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 16 --data.seed 10000

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 1 --data.seed 1

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 1 --data.seed 10

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 1 --data.seed 100

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 1 --data.seed 1000

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 1 --data.seed 10000

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 2 --data.seed 1

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 2 --data.seed 10

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 2 --data.seed 100

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 2 --data.seed 1000

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 2 --data.seed 10000

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 4 --data.seed 1

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 4 --data.seed 10

uv run coop.py --config configs/coop_protofuse.yaml \
--data.root "${FOOD_DATA_ROOT}" --data.dataset_name "Food-101" \
--data.kshot 4 --data.seed 100
