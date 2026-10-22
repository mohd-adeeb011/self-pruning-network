import numpy as np
from train.data import make_data
from train.run import Config, load_model, run
from prune.inference import DenseExport, SparseExport


def test_data_reproducible_and_normalized_on_train_only():
    a,b=make_data(),make_data()
    for name in ('train','val','test'):
        np.testing.assert_array_equal(a[name][0],b[name][0])
    np.testing.assert_allclose(a['train'][0].mean(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(a['train'][0].std(axis=0),1,atol=1e-14)


def test_training_roundtrip_and_reproducibility(tmp_path):
    config=Config(method='magnitude',sparsity=.5,hidden=8,examples=200,
                  epochs=16,prune_start=3,prune_end=12,prune_every=3)
    first=run(config,tmp_path/'a',benchmark=False)
    second=run(config,tmp_path/'b',benchmark=False)
    assert first['history']==second['history']
    assert first['history'][-1]['train_loss']<first['history'][0]['train_loss']
    assert first['achieved_sparsity']>=.5
    model,normalization=load_model(tmp_path/'a'/'model.npz')
    x,y=make_data(examples=200)['test']
    prediction=DenseExport(model)(x)
    assert float((prediction.argmax(axis=1)==y).mean())==first['test']['accuracy']
    np.testing.assert_allclose(prediction,SparseExport(model)(x),atol=1e-12)
    assert all(np.isfinite(value).all() for value in normalization)


def test_hessian_pruning_training_smoke(tmp_path):
    config=Config(method='hessian',sparsity=.5,hidden=2,examples=100,
                  epochs=6,prune_start=1,prune_end=4,prune_every=1,
                  calibration_size=16)
    result=run(config,tmp_path/'hessian',benchmark=False)
    assert result['achieved_sparsity']>=.5
    assert result['pruning_events']
    assert all(np.isfinite(row['train_loss']) for row in result['history'])
    actual_events=[event for event in result['pruning_events'] if event['removed_now']]
    assert all(event['condition_after_shift']>0 for event in actual_events)
    assert all(np.isfinite(event['actual_calibration_loss_change']) for event in actual_events)
