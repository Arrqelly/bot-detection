"""Reproducible temporal comparison; never reads test.csv or writes submission.csv."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from metric import precision_at_recall, recall_at_fpr
from solution import (
    DATE_COLUMNS, ROOT, behavior_features, context_features, create_model,
    events_in_window, extended_pointer_features, pointer_features,
)

# End dates are exclusive. Training windows must be finished by validation start.
FOLDS = (
    ('apr12_13', '2026-04-12', '2026-04-14'),
    ('apr14_16', '2026-04-14', '2026-04-17'),
    ('apr17_19', '2026-04-17', '2026-04-20'),
)
METRICS = ('precision_at_recall70', 'average_precision', 'roc_auc',
           'recall_at_fpr01', 'log_loss', 'fit_seconds', 'predict_ms_per_1000')


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    features: str
    c: float = 1.0
    seed: int = 0


def experiments():
    configs = [Experiment(f'{model}_{group}', model, group)
               for group in ('base66', 'pointer78', 'all110')
               for model in ('lr', 'catboost')]
    configs += [Experiment(f'lr_all110_c{c:g}', 'lr', 'all110', c=c) for c in (0.1, 10.0)]
    configs += [Experiment(f'catboost_all110_seed{seed}', 'catboost', 'all110', seed=seed)
                for seed in (1, 2)]
    return configs


def temporal_split(meta, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train = meta.index[(meta.window_start_ts < start) & (meta.window_end_ts <= start)]
    valid = meta.index[(meta.window_start_ts >= start) & (meta.window_start_ts < end)
                       & (meta.window_end_ts <= end)]
    if not len(train) or not len(valid):
        raise ValueError('Empty temporal split')
    if not meta.cookie_id.is_unique:
        raise ValueError('Expected one row per cookie; repeated cookies require grouped splitting')
    for ids in (train, valid):
        if set(meta.loc[ids, 'target']) != {0, 1}:
            raise ValueError('Each split must contain both target classes')
    return train, valid


def feature_tables(events, meta):
    # No fitted statistics across cookies: each feature is an independent cookie aggregate.
    ev = events_in_window(events, meta.drop(columns='target'))
    base = behavior_features(ev, meta).join(context_features(ev, meta))
    pointer = base.join(pointer_features(ev, meta))
    full = pointer.join(extended_pointer_features(ev, meta))
    tables = {'base66': base, 'pointer78': pointer, 'all110': full}
    for name, expected in [('base66', 66), ('pointer78', 78), ('all110', 110)]:
        frame = tables[name]
        if frame.shape[1] != expected or not frame.index.is_unique or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f'Invalid feature matrix: {name}')
        if {'cookie_id', 'target'} & set(frame.columns):
            raise ValueError('Identifier or label entered feature matrix')
    return tables


def make_estimator(config, threads):
    if config.model == 'lr':
        # fit() sees only the training fold, including StandardScaler statistics.
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=config.c, solver='lbfgs', max_iter=10000, tol=1e-5, random_state=config.seed,
        ))
    return create_model().set_params(random_seed=config.seed, thread_count=threads)


def submission_precision(scores):
    """Evaluate the same five-significant-digit values that the submission stores."""
    return np.asarray([float(f'{value:.5g}') for value in scores])


def score_predictions(y, scores):
    scores = submission_precision(scores)
    return {
        'precision_at_recall70': precision_at_recall(y, scores),
        'average_precision': average_precision_score(y, scores),
        'roc_auc': roc_auc_score(y, scores),
        'recall_at_fpr01': recall_at_fpr(y, scores),
        'log_loss': log_loss(y, scores, labels=[0, 1]),
    }


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plot_results(summary, fold_results, output):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    colors = {'lr': '#3268b0', 'catboost': '#d66b27'}
    for model, label in [('lr', 'Logistic regression (C=1)'), ('catboost', 'CatBoost (seed=0)')]:
        rows = fold_results[fold_results.experiment == f'{model}_all110']
        axes[0].plot(range(3), rows.precision_at_recall70 * 100, 'o-', label=label, color=colors[model])
        means = [summary.loc[f'{model}_{group}', 'precision_at_recall70_mean'] * 100
                 for group in ('base66', 'pointer78', 'all110')]
        axes[1].plot([66, 78, 110], means, 'o-', label=label, color=colors[model])
    axes[0].set_xticks(range(3), ['Apr 12–13', 'Apr 14–16', 'Apr 17–19'])
    axes[0].set_title('Temporal validation · 110 features')
    axes[1].set_xticks([66, 78, 110])
    axes[1].set_xlabel('Number of features')
    axes[1].set_title('Feature ablation · mean of 3 periods')
    for ax in axes:
        ax.set_ylabel('Precision at Recall ≥ 70%, %')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='lower right', fontsize=9)
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)


def run(data_dir, output, threads):
    output.mkdir(parents=True, exist_ok=True)
    configs = experiments()
    train_path, events_path = data_dir / 'train.csv', data_dir / 'events.csv/events.csv'
    metadata = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'logical_cpu_count': os.cpu_count(),
        'threads': threads,
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'catboost', 'scikit-learn', 'matplotlib', 'scipy', 'threadpoolctl']},
        'data_sha256': {'train.csv': digest(train_path), 'events.csv': digest(events_path)},
        'source_sha256': {name: digest(ROOT / name) for name in ['solution.py', 'metric.py', 'benchmark.py']},
        'folds': FOLDS, 'experiments': [asdict(c) for c in configs],
        'catboost_parameters': create_model().set_params(thread_count=threads).get_params(),
        'logistic_regression_parameters': make_estimator(configs[0], threads).named_steps['logisticregression'].get_params(),
        'score_rounding': '5 significant digits for every quality metric',
        'selection': 'Prespecified diagnostic comparisons; no automatic selection or independent holdout.',
        'status': 'running',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    meta = pd.read_csv(train_path, parse_dates=DATE_COLUMNS)
    events = pd.read_csv(events_path, parse_dates=['event_ts'])
    started = perf_counter()
    tables = feature_tables(events, meta)
    metadata['feature_seconds'] = perf_counter() - started
    metadata['feature_columns'] = {name: list(frame.columns) for name, frame in tables.items()}
    del events
    print(f'Features ready in {metadata["feature_seconds"]:.2f}s', flush=True)
    rows, predictions, split_rows = [], [], []
    with threadpool_limits(limits=threads):
        for fold, start, end in FOLDS:
            train_ids, valid_ids = temporal_split(meta, start, end)
            train_meta, valid_meta = meta.loc[train_ids], meta.loc[valid_ids]
            y_train, y_valid = train_meta.target, valid_meta.target
            split_rows.append({
                'fold': fold, 'train_start': str(train_meta.window_start_ts.min().date()),
                'train_end_exclusive': str(train_meta.window_end_ts.max().date()),
                'valid_start': start, 'valid_end_exclusive': end,
                'n_train': len(train_ids), 'n_valid': len(valid_ids),
                'train_positive_rate': float(y_train.mean()), 'valid_positive_rate': float(y_valid.mean()),
            })
            for config in configs:
                frame = tables[config.features]
                x_train, x_valid = frame.loc[train_meta.cookie_id], frame.loc[valid_meta.cookie_id]
                estimator = make_estimator(config, threads)
                started = perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    estimator.fit(x_train, y_train)
                fit_seconds = perf_counter() - started
                predict_kwargs = {'thread_count': threads} if config.model == 'catboost' else {}
                scores = estimator.predict_proba(x_valid, **predict_kwargs)[:, 1]
                timings = []
                for _ in range(5):
                    started = perf_counter()
                    estimator.predict_proba(x_valid, **predict_kwargs)
                    timings.append(perf_counter() - started)
                result = {
                    'experiment': config.name, 'model': config.model, 'features': config.features,
                    'c': config.c if config.model == 'lr' else None, 'seed': config.seed,
                    'fold': fold, 'n_features': frame.shape[1], 'n_train': len(train_ids),
                    'n_valid': len(valid_ids), **score_predictions(y_valid, scores),
                    'fit_seconds': fit_seconds,
                    'predict_ms_per_1000': float(np.median(timings) * 1e6 / len(valid_ids)),
                }
                rows.append(result)
                predictions.append(pd.DataFrame({
                    'experiment': config.name, 'fold': fold, 'cookie_id': valid_meta.cookie_id.to_numpy(),
                    'target': y_valid.to_numpy(), 'score': submission_precision(scores),
                }))
                # Persist completed fits so an interrupted run is inspectable.
                pd.DataFrame(rows).to_csv(output / 'fold_metrics.csv', index=False)
                print(f'{fold} | {config.name:26s} | P@R70={result["precision_at_recall70"]:.4f}'
                      f' | AP={result["average_precision"]:.4f} | fit={fit_seconds:.2f}s', flush=True)
    results = pd.DataFrame(rows)
    summary = results.groupby('experiment', sort=False)[list(METRICS)].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(output / 'summary.csv')
    pd.DataFrame(split_rows).to_csv(output / 'splits.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / 'predictions.csv.gz', index=False, compression='gzip')
    plot_results(summary, results, output)
    metadata['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['completed_fits'] = len(rows)
    metadata['status'] = 'complete'
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Completed {len(rows)} fits. Results: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args.data_dir.resolve(), args.output_dir.resolve(), args.threads)


if __name__ == '__main__':
    main()
