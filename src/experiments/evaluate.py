import argparse, os, json, matplotlib.pyplot as plt

def main(artifacts_dir):
    metrics_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.json')]
    print('Artifacts at', artifacts_dir)
    for mf in metrics_files:
        d = json.load(open(os.path.join(artifacts_dir, mf)))
        print(mf, d)
    # simple bar plot if tabular metrics present
    m = json.load(open(os.path.join(artifacts_dir, 'tabular_ensemble_metrics.json')))
    names = ['auc','f1','precision','recall']
    vals = [m[k] for k in names]
    plt.bar(names, vals); plt.title('Tabular Ensemble Metrics'); plt.ylim(0,1.0)
    plt.savefig(os.path.join(artifacts_dir, 'tabular_metrics_bar.png'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', required=True)
    args = ap.parse_args()
    main(args.artifacts)
