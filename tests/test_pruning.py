import numpy as np
from engine import Tensor, cross_entropy
from nn import MLP, Adam
from prune.hessian import central_hessian, model_scores, removal_scores, stabilize_hessian
from prune.inference import DenseExport, SparseExport
from prune.schedule import cubic_sparsity, prune_to


def test_numerical_hessian_and_taylor_on_known_quadratic():
    matrix=np.array([[4.,1.],[1.,2.]])
    linear=np.array([.3,-.5])
    point=np.array([.2,-.8])
    def gradient(value):
        x=Tensor(value[None,:])
        loss=((x@Tensor(matrix,False))*x).sum()/2+(x*linear).sum()
        loss.backward()
        return x.grad.ravel().copy()
    hessian,asymmetry=central_hessian(gradient,point)
    np.testing.assert_allclose(hessian,matrix,atol=1e-10)
    assert asymmetry<1e-10
    scores=removal_scores(point,gradient(point),hessian)
    def objective(w):
        return .5*w@matrix@w+linear@w
    for i in range(len(point)):
        removed=point.copy()
        removed[i]=0
        np.testing.assert_allclose(scores[i],objective(removed)-objective(point),atol=1e-10)


def test_small_weight_is_not_always_least_important():
    scores=removal_scores(np.array([.1,1.]),np.zeros(2),np.diag([1000.,.01]))
    assert scores[0]>scores[1]


def test_indefinite_and_singular_hessian_stability():
    for h in (np.array([[-2.,.5],[.5,0.]]),np.zeros((3,3))):
        stable,diagnostics=stabilize_hessian(h)
        assert np.linalg.eigvalsh(stable).min()>0
        assert np.isfinite(diagnostics['condition_after_shift'])
        assert diagnostics['condition_after_shift']<=2001


def test_schedule_budget_and_sparse_equivalence():
    model=MLP(hidden=7,seed=3)
    opt=Adam(model.parameters())
    previous=0
    for epoch in range(21):
        target=cubic_sparsity(epoch,3,15,.9)
        assert previous<=target<=.9
        previous=target
    prune_to(model,opt,.9,'magnitude')
    total=sum(p.data.size for p in model.weights())
    active=sum(p.mask.sum() for p in model.weights())
    assert active<=np.floor(.1*total+1e-10)
    x=np.random.default_rng(5).normal(size=(8,2))
    np.testing.assert_allclose(DenseExport(model)(x),SparseExport(model)(x),atol=1e-12)
    old=[p.mask for p in model.weights()]
    prune_to(model,opt,.75,'magnitude')
    for mask,p in zip(old,model.weights()):
        np.testing.assert_array_equal(mask,p.mask)


def test_model_hessian_matches_loss_second_difference_and_restores_state():
    model=MLP(hidden=2,seed=4)
    x=np.array([[.2,.4],[-.5,.8],[.9,-.1]])
    y=np.array([0,1,0])
    snapshots=[(p.data.copy(),p.grad.copy()) for p in model.parameters()]
    scores,diagnostics=model_scores(model,x,y)
    assert all(np.isfinite(s).all() for s in scores)
    assert diagnostics['active_dimension']==8
    for p,(data,grad) in zip(model.parameters(),snapshots):
        np.testing.assert_array_equal(p.data,data)
        np.testing.assert_array_equal(p.grad,grad)
    p=model.weights()[0]
    original=p.data.flat[0]
    def gradient(v):
        p.data.flat[0]=v[0]
        cross_entropy(model(x),y).backward()
        return np.array([p.grad.flat[0]])
    try:
        hessian,_=central_hessian(gradient,np.array([original]))
        h=1e-3
        losses=[]
        for delta in (h,0,-h):
            p.data.flat[0]=original+delta
            losses.append(float(cross_entropy(model(x),y).data))
        numerical=(losses[0]-2*losses[1]+losses[2])/h**2
        np.testing.assert_allclose(hessian[0,0],numerical,rtol=1e-3,atol=1e-6)
    finally:
        p.data.flat[0]=original


def test_hessian_step_size_sensitivity_on_smooth_network():
    model=MLP(hidden=2,seed=6)
    x=np.array([[.2,.4],[-.5,.8],[.9,-.1]])
    y=np.array([0,1,0])
    weights=model.weights()
    original=np.concatenate([p.data.ravel() for p in weights])
    def gradient(vector):
        offset=0
        for p in weights:
            p.data[:]=vector[offset:offset+p.data.size].reshape(p.data.shape)
            offset+=p.data.size
        cross_entropy(model(x),y).backward()
        return np.concatenate([p.grad.ravel() for p in weights])
    try:
        reference,_=central_hessian(gradient,original,step=1e-4)
        for step in (1e-3,1e-5):
            candidate,_=central_hessian(gradient,original,step=step)
            np.testing.assert_allclose(candidate,reference,rtol=2e-3,atol=2e-6)
    finally:
        gradient(original)
