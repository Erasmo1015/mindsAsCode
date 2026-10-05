"""Historical comparator and preferred-style contract checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from analysis.mem.pics_v4_comparator_selectors import historical, REPO, BASE
from utils.mem.pics_v4_source_profile_extensions_frozen import check_extensions


def test_historical_combined_block_equations():
    h=historical();ds=['alpha','beta','gamma']
    occ=pd.DataFrame([dict(dataset=d,dataset_label=d,**{c:(i+1)/(k+4) for k,c in enumerate(h.MOT5)}) for i,d in enumerate(ds)])
    fit=pd.DataFrame({'dataset':ds,'dataset_label':ds,'history_modified':[-2.,0.,4.],'feedback_added':[1.,1.,1.]})
    combined,meta=h.rss.build_combined_signatures(occ,fit)
    def standard(x):
        mean=x.mean(axis=0);sd=x.std(axis=0,ddof=0)
        return np.divide(x-mean,sd,out=np.zeros_like(x),where=sd>=1e-12)
    expected=np.concatenate([standard(occ[list(h.MOT5)].to_numpy())/np.sqrt(2),standard(fit[['history_modified','feedback_added']].to_numpy())/np.sqrt(2)],axis=1)
    np.testing.assert_allclose(combined[meta['feature_names']],expected,rtol=1e-14,atol=1e-14)
    assert meta['n_occurrence_dims']==5
    assert meta['n_fitness_dims']==2


def test_historical_source_eligibility_and_ties():
    h=historical();vecs={d:np.ones(5) for d in [*h.rss.SIX,'14kool2016when']}
    ranks=h.select_margin(vecs,sorted(vecs))
    for r in ranks.itertuples():
        assert r.selected_source_id==min(d for d in h.rss.SIX if d!=r.target_id)


def test_historical_settings_unchanged():
    h=historical()
    assert h.B_FIT==200 and h.B_OCC==1000 and h.SEED==20260920
    assert h.rss.MIN_POSITIVE_ROWS==20
    assert h.rss.MIN_DATASETS_WITH_POSITIVE==3
    assert h.rss.MIN_DATASETS_WITH_BOTH==2


def test_preferred_style_and_fixed_primary_input():
    config=check_extensions(REPO)
    style=json.loads((REPO/BASE/'extensions/scripts/preferred_heatmap_style.json').read_text())
    assert style['column_labels']==['History','Value','Probability','Feedback','Learning']
    assert style['figsize']==[8.2,7.2]
    assert style['cmap_colors']==['#f7f4ef','#c8d5b9','#5b8a72','#1f3d34']
    assert style['vmin']==0 and style['vmax']==1
    assert len(style['target_ids'])==15
    assert config['plot']['input_profile'].endswith('run_v1/primary_runtime_valid/profiles.csv')


def test_changed_extension_refused(tmp_path):
    p=tmp_path/'analysis/config/mem';p.mkdir(parents=True)
    # Test extension guard without touching any real input.
    from utils.mem import pics_v4_source_profile_extensions_frozen as lock
    original=lock.check_frozen
    lock.check_frozen=lambda repo:None
    try:
        (p/'pics_v4_source_profile_extensions_frozen.json').write_text('{}')
        with pytest.raises(ValueError,match='extension configuration changed'):
            lock.check_extensions(tmp_path)
    finally:
        lock.check_frozen=original
