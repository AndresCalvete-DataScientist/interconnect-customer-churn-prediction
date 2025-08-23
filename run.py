import argparse
from pipeline import p03_analysis_pipeline

parser = argparse.ArgumentParser(description="Ejecutor de pipelines del proyecto")
parser.add_argument(
    "pipelines",
    nargs="+",   
    choices=["p01", "p02", "p03"],
    help="Pipeline(s) a ejecutar en orden: p01 (train), p02 (predict), p03 (analysis)"
)
args = parser.parse_args()

for pl in args.pipelines:
    if pl == "p01":
        #p01_train_pipeline.main()
        pass
    elif pl == "p02":
        #p02_predict_pipeline.main()
        pass
    elif pl == "p03":
        p03_analysis_pipeline.main()