import argparse, json, os, numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ..utils import set_seed, ensure_dir, save_json
from ..metrics import compute_binary_metrics, optimal_threshold
from ..data.build_graph import build_transaction_graph
from ..data.tabular_features import load_tabular
from ..models.gnn_model import SimpleGCN
from ..models.autoencoder import TabularAE
from ..models.tabular_baseline import make_tabular_classifier

def run(cfg):
    set_seed(cfg['seed'])
    results_dir = os.path.join(cfg['paths']['results_dir'], 'run_000')
    ensure_dir(results_dir)

    # ---------- TABULAR BASELINE ----------
    X, y, enc = load_tabular(cfg['paths']['data_csv'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=cfg['seed'], stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    clf = make_tabular_classifier()
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]

    # ---------- AUTOENCODER ----------
    ae = TabularAE(in_dim=X_train.shape[1], hidden=cfg['ae_hidden'])
    opt = torch.optim.Adam(ae.parameters(), lr=cfg['lr'])
    Xt = torch.tensor(X_train, dtype=torch.float32)
    for _ in range(30):
        opt.zero_grad(); recon,_ = ae(Xt); loss = ((Xt-recon)**2).mean(); loss.backward(); opt.step()
    ae_scores = ae.anomaly_score(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    ae_scores = (ae_scores - ae_scores.min())/(ae_scores.ptp()+1e-8)

    # ---------- ENSEMBLE ----------
    # combine calibrated prob with ae anomaly score
    y_prob_ens = 0.7*y_prob + 0.3*ae_scores
    thr, _ = optimal_threshold(y_test, y_prob_ens, strategy=cfg['threshold_strategy'])
    metrics = compute_binary_metrics(y_test, y_prob_ens, threshold=thr)
    save_json(metrics, os.path.join(results_dir, 'tabular_ensemble_metrics.json'))

    # ---------- GNN (optional) ----------
    if cfg['use_gnn']:
        data = build_transaction_graph(cfg['paths']['data_csv'])
        model = SimpleGCN(in_features=data.x.size(1), hidden=cfg['hidden_size'], out_features=16)
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        loss_fn = torch.nn.NLLLoss()
        model.train()
        for _ in range(cfg['train_epochs']):
            optim.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward(); optim.step()

        with torch.no_grad():
            proba = torch.softmax(model(data), dim=1)[:,1].cpu().numpy()
        # crude edge-level eval
        y_true_edges = data.y.cpu().numpy()
        thr_gnn, _ = optimal_threshold(y_true_edges, proba, strategy='f1_max')
        gnn_metrics = compute_binary_metrics(y_true_edges, proba, threshold=thr_gnn)
        save_json(gnn_metrics, os.path.join(results_dir, 'gnn_edge_metrics.json'))

    # save config
    save_json(cfg, os.path.join(results_dir, 'config_used.json'))
    print('Done. Artifacts saved to', results_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.json')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    run(cfg)
