from ultralytics.models.fastsam import FastSAMPredictor

overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-x.pt", save=False, imgsz=1024)
predictor = FastSAMPredictor(overrides=overrides)

imgs = [
    'datasets/cub-200-2011-renamed/Acadian_Flycatcher/Acadian_Flycatcher_0006_795595.jpg',
    'datasets/cub-200-2011-renamed/Baltimore_Oriole/Baltimore_Oriole_0014_87690.jpg',
    # 'datasets/cub-200-2011-renamed/American_Crow/American_Crow_0074_25350.jpg',
]
for img in imgs:
    everything_results = predictor(img)
    text_results = predictor.prompt(everything_results, texts="a photo of a bird")
    text_results[0].save()