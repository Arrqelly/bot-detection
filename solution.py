"""Подготовка признаков, обучение CatBoost и формирование submission."""

import numpy as np
from pathlib import Path
import argparse
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
DATE_COLUMNS = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
RANDOM_SEED = 0

def create_model():
    return CatBoostClassifier(
        iterations=500,             
        depth=7,
        learning_rate=0.05,          
        l2_leaf_reg=10,               
        random_seed=RANDOM_SEED,     
        verbose=False,              
        allow_writing_files=False,  
    )




import numpy as np
import pandas as pd

# Фиксированный список сохраняет столбцы даже при отсутствии типа события.
EVENT_TYPES = [
    'item_view', 'search_results_view', 'photo_swipe', 'favorite_add',
    'seller_page_view', 'contact_phone_show', 'captcha_shown', 'login',
    'contact_chat_open', 'contact_message_sent',
]

# Оставляем события кук из meta в интервале [начало, конец).
def events_in_window(events, meta):
    ev = events.merge(meta[['cookie_id', 'window_start_ts', 'window_end_ts']],
                      on='cookie_id', how='inner', validate='many_to_one')
    return ev.loc[ev.event_ts.ge(ev.window_start_ts) & ev.event_ts.lt(ev.window_end_ts)].copy()

# 58 признаков: интенсивность, ритм, разнообразие и состав событий.
def behavior_features(ev, meta):
    # Удаляем только полные дубликаты; сортировка нужна для интервалов.
    ev = ev.drop_duplicates().copy()
    ev = ev.sort_values(['cookie_id', 'event_ts'], kind='mergesort')
    g = ev.groupby('cookie_id', sort=False)

    # Объём и разнообразие: агрегаты каждого поля считаем один раз.
    diversity_columns = ['item_id', 'item_category', 'item_location',
                         'search_query', 'search_page', 'event_name']
    unique = g[diversity_columns].nunique()
    filled = g[diversity_columns].count().replace(0, np.nan)
    f = pd.DataFrame({'n_events': g.size(), 'item_nunique': unique.item_id})

    # Для двух базовых счётчиков отсутствие событий означает 0.
    f = f.reindex(meta.cookie_id).fillna(0)
    n = f.n_events.replace(0, np.nan)

    # Знаменатель — заполненные значения поля.
    for col in diversity_columns[1:]:
        f[col + '_nunique'] = unique[col]
        f[col + '_unique_ratio'] = unique[col] / filled[col]
    f['item_unique_ratio'] = unique.item_id / filled.item_id

    # Состав действий: 10 количеств и 10 долей от общего числа событий.
    counts = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_TYPES, fill_value=0)
    for name in EVENT_TYPES:
        f['count_' + name] = counts[name]
        f['share_' + name] = counts[name] / n

    # Активность по календарным минутам и часам; окна данных — сутки от полуночи.
    ev['minute'] = ev.event_ts.dt.floor('min')
    ev['hour'] = ev.event_ts.dt.hour

    # У первого события нет известного интервала: NaN.
    ev['gap'] = g.event_ts.diff().dt.total_seconds()
    g = ev.groupby('cookie_id', sort=False)
    f['active_minutes'] = g.minute.nunique()
    f['active_hours'] = g.hour.nunique()

    # Размах включает паузы, а плотность считается только по активным минутам.
    f['span_seconds'] = (g.event_ts.max() - g.event_ts.min()).dt.total_seconds()
    f['events_per_active_minute'] = n / f.active_minutes.replace(0, np.nan)
    f['peak_events_per_minute'] = ev.groupby(['cookie_id', 'minute']).size().groupby(level=0).max()

    # Ритм: статистики известных интервалов.
    gaps = g.gap.agg(['mean', 'std', 'min', 'max', 'median']).add_prefix('gap_')
    
    f = f.join(gaps)
    f['gap_q25'] = g.gap.quantile(.25)
    f['gap_q75'] = g.gap.quantile(.75)
    f['gap_cv'] = f.gap_std / f.gap_mean.replace(0, np.nan)

    valid_gap = ev.loc[ev.gap.notna()].copy()
    for name, mask in {
        'gap_zero_share': valid_gap.gap.eq(0),
        'gap_le_2s_share': valid_gap.gap.le(2),
        'gap_ge_30m_share': valid_gap.gap.ge(1800),
    }.items():
        f[name] = mask.groupby(valid_gap.cookie_id).mean()

    # Новая сессия после паузы строго >30 минут; доля выше учитывает и ровно 30.
    f['sessions_30m'] = ev.gap.gt(1800).groupby(ev.cookie_id).sum() + 1

    # Глубина выдачи по заполненным номерам страниц.
    f['search_page_max'] = g.search_page.max()
    f['search_page_mean'] = g.search_page.mean()

    # Концентрация и энтропия по заполненным значениям каждого поля.
    for col in ['item_id', 'item_category', 'search_query']:
        freq = ev.groupby(['cookie_id', col]).size()
        proportions = freq / freq.groupby(level=0).transform('sum')
        f[col + '_top_share'] = proportions.groupby(level=0).max()
        
        # H = -sum(p * ln(p)), без нормировки на число объектов.
        f[col + '_entropy'] = (-(proportions * np.log(proportions))).groupby(level=0).sum()

    # -1 отличает неопределённый агрегат от настоящего нуля.
    return f.reindex(meta.cookie_id).replace([np.inf, -np.inf], np.nan).fillna(-1)

# 8 признаков: доли платформ и маркеров User-Agent, возраст куки.
def context_features(ev, meta):
    
    # Та же обработка дубликатов, что и у поведенческих признаков.
    ev = ev.drop_duplicates().copy()
    platform = ev.platform.str.lower().replace({'desktop': 'web', 'iphone': 'ios'})
    f = pd.DataFrame(index=pd.Index(meta.cookie_id, name='cookie_id'))
    for name in ['web', 'android', 'ios']:
        f['platform_share_' + name] = platform.eq(name).groupby(ev.cookie_id).mean()
    ua = ev.user_agent.str.lower()

    # Маркеры клиента — признаки.
    families = {
        'ua_http_library': ua.str.contains('python|curl|scrapy|go-http-client|node-fetch', regex=True),
        'ua_headless': ua.str.contains('headless', regex=False),
        'ua_native_app': ua.str.startswith('avito/'),
        'ua_firefox': ua.str.contains('firefox/', regex=False),
    }
    for name, mask in families.items():
        f[name] = mask.groupby(ev.cookie_id).mean()
    m = meta.set_index('cookie_id')

    # Возраст к концу окна, в днях; кука может существовать дольше самого окна.
    f['cookie_age_days'] = (m.window_end_ts - m.cookie_created_at).dt.total_seconds() / 86400
    return f.fillna(-1)


""" 12 агрегатов координат указателя внутри уже отфильтрованного окна.
    Отсутствие координат не означает нулевые координаты. Между двумя
    наблюдаемыми позициями считаем смещение, а не длину траектории мыши.
    Неоднозначные переходы у одинаковых временных меток исключаем.
"""
def pointer_features(ev, meta):
    
    ev = ev.drop_duplicates().sort_values(['cookie_id', 'event_ts'], kind='mergesort').copy()
    index = pd.Index(meta.cookie_id, name='cookie_id')
    f = pd.DataFrame(index=index)

    has_pointer = ev.pointer_x.notna() & ev.pointer_y.notna()
    f['pointer_share'] = has_pointer.groupby(ev.cookie_id).mean()
    pe = ev.loc[has_pointer].copy()
    pg = pe.groupby('cookie_id', sort=False)

    for col in ['pointer_x', 'pointer_y']:
        f[col + '_std'] = pg[col].std()
        f[col + '_range'] = pg[col].max() - pg[col].min()
        f[col + '_unique_ratio'] = pg[col].nunique() / pg.size()

    positions = pe.groupby(['cookie_id', 'pointer_x', 'pointer_y']).size()

    f['pointer_position_top_share'] = positions.groupby(level=0).max() / pg.size()
    f['pointer_position_unique_ratio'] = positions.groupby(level=0).size() / pg.size()

    dx, dy = pg.pointer_x.diff(), pg.pointer_y.diff()
    dt = pg.event_ts.diff().dt.total_seconds()
    ambiguous = pe.duplicated(['cookie_id', 'event_ts'], keep=False)
    
    previous_ambiguous = ambiguous.groupby(pe.cookie_id).shift(fill_value=False)
    valid = dt.gt(0) & dt.le(1800) & ~ambiguous & ~previous_ambiguous
    movement = np.sqrt(dx * dx + dy * dy).where(valid)
    
    f['pointer_step_median'] = movement.groupby(pe.cookie_id).median()
    f['pointer_step_std'] = movement.groupby(pe.cookie_id).std()
    f['pointer_stationary_share'] = movement.dropna().eq(0).groupby(pe.loc[movement.notna(), 'cookie_id']).mean()

    return f.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(-1).astype(float)

# 32 признака распределения координат и однозначных смещений указателя.
def extended_pointer_features(ev, meta):
    
    ev = ev.drop_duplicates().sort_values(['cookie_id', 'event_ts'], kind='mergesort').copy()
    index = pd.Index(meta.cookie_id, name='cookie_id')
    has = ev.pointer_x.notna() & ev.pointer_y.notna()
    pe = ev.loc[has].copy()
    g = pe.groupby('cookie_id', sort=False)
    geometry = pd.DataFrame(index=index)

    for col in ['pointer_x', 'pointer_y']:
        v = g[col]
        for stat in ['mean', 'median', 'min', 'max', 'skew']:
            geometry[col + '_' + stat] = v.agg(stat)
            
        q25, q75 = v.quantile(.25), v.quantile(.75)
        geometry[col + '_iqr'] = q75 - q25
        geometry[col + '_iqr_range'] = (q75 - q25) / (v.max() - v.min()).replace(0, np.nan)
        geometry[col + '_std_range'] = v.std() / (v.max() - v.min()).replace(0, np.nan)

    # Корреляция, вычисленная на основе центрированных моментов, позволяет избежать использования `groupby.apply`.
    cx = pe.pointer_x - g.pointer_x.transform('mean')
    cy = pe.pointer_y - g.pointer_y.transform('mean')

    cov = (cx * cy).groupby(pe.cookie_id).sum()
    denominator = np.sqrt((cx * cx).groupby(pe.cookie_id).sum() * (cy * cy).groupby(pe.cookie_id).sum())

    geometry['pointer_xy_correlation'] = cov / denominator.replace(0, np.nan)
    geometry['pointer_aspect_std'] = g.pointer_x.std() / g.pointer_y.std().replace(0, np.nan)

    motion = pd.DataFrame(index=index)
    dx, dy = g.pointer_x.diff(), g.pointer_y.diff()
    dt = g.event_ts.diff().dt.total_seconds()

    ambiguous = pe.duplicated(['cookie_id', 'event_ts'], keep=False)
    previous_ambiguous = ambiguous.groupby(pe.cookie_id).shift(fill_value=False)
    valid = dt.gt(0) & dt.le(1800) & ~ambiguous & ~previous_ambiguous

    dx, dy = dx.where(valid), dy.where(valid)
    distance = np.sqrt(dx * dx + dy * dy)
    speed = distance / dt.where(valid)

    for name, values in [('pointer_distance', distance), ('pointer_observed_speed', speed)]:
        vg = values.groupby(pe.cookie_id)
        for stat in ['mean', 'std', 'max']:
            motion[name + '_' + stat] = vg.agg(stat)

        motion[name + '_q25'] = vg.quantile(.25)
        motion[name + '_q75'] = vg.quantile(.75)

    for name, mask in [('same_x', dx.eq(0)), ('same_y', dy.eq(0)), ('small_step', distance.le(5))]:
        motion['pointer_' + name + '_share'] = mask[valid].groupby(pe.loc[valid, 'cookie_id']).mean()

    # Сравнивайте направления только тогда, когда оба последовательных перемещения однозначны и отличны от нуля.
    previous_dx = dx.groupby(pe.cookie_id).shift()
    previous_dy = dy.groupby(pe.cookie_id).shift()

    norm = distance * np.sqrt(previous_dx ** 2 + previous_dy ** 2)
    cosine = (dx * previous_dx + dy * previous_dy) / norm.replace(0, np.nan)
    motion['pointer_turn_cosine_mean'] = cosine.groupby(pe.cookie_id).mean()

    return geometry.join(motion).reindex(index).replace([np.inf, -np.inf], np.nan).fillna(-1).astype(float)


def build_features(events, meta):
    # Сначала ограничиваем события окном, затем считаем агрегаты каждой куки.
    ev = events_in_window(events, meta)
    
    # 58 поведенческих, 8 контекстных, 12 базовых и 32 новых признака указателя.
    # Идентификатор не становится признаком модели.
    return behavior_features(ev, meta).join([
        context_features(ev, meta), pointer_features(ev, meta), extended_pointer_features(ev, meta)
    ])


def run(data_dir, output):
    # 1 Загружаем исходные данные 
    # Параметры модели и функции расчёта признаков определены выше.
    # train: одна строка на куку, границы окна и известный target.
    train = pd.read_csv(data_dir / 'train.csv', parse_dates=DATE_COLUMNS)
    test = pd.read_csv(data_dir / 'test.csv', parse_dates=DATE_COLUMNS)
    events = pd.read_csv(data_dir / 'events.csv/events.csv', parse_dates=['event_ts'])

    # Общая таблица метаданных позволяет применить одинаковую обработку ко всем
    # кукам. Target намеренно удаляем. 
    # ignore_index создаёт обычный последовательный индекс объединённой таблицы.
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)

    # 2 Превращаем поток событий в таблицу «кука × признаки»
    x = build_features(events, meta)


    assert x.shape[1] == 110, "Expected 110 features; restart the kernel and run all cells."


    # Обучаем модель на train.
    print(f'CatBoost: {len(train)} training cookies, {x.shape[1]} features, one model, seed=0', flush=True)
    model = create_model()
    model.fit(x.loc[train.cookie_id], train.target)


    # 4 Получаем непрерывные оценки для test
    # predict_proba возвращает два столбца: для класса 0 и класса 1.
    # [:, 1] выбирает вероятность положительного класса — автоматизированного
    # трафика.
    # Идентификаторы и прогнозы идут в одном порядке — как строки test.
    sub = pd.DataFrame({
        'cookie_id': test.cookie_id,
        'score': model.predict_proba(x.loc[test.cookie_id])[:, 1],
    })

    # 5 Сохраняем CSV 
    # index=False: не добавлять третий столбец с индексом pandas.
    # Явный '\n' задаёт одинаковый перевод строки для воспроизводимого файла.
    output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output, index=False, float_format='%.5g', lineterminator='\n')
    print(f'Saved {output}: {len(sub)} rows')


def main():
    parser = argparse.ArgumentParser(description='Train CatBoost and generate submission.csv.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'submission.csv')
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
