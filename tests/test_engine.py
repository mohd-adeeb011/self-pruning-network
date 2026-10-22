import numpy as np
import pytest
from engine import Tensor, cross_entropy


def check_gradient(function, arrays, atol=2e-6):
    values = [np.array(a, dtype=float, copy=True) for a in arrays]
    tensors = [Tensor(a) for a in values]
    result = function(*tensors)
    result.backward()
    for parameter, expected in zip(values, tensors):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            h = 1e-5 * max(1, abs(original))
            parameter[index] = original+h
            plus = function(*[Tensor(a, False) for a in values]).data.copy()
            parameter[index] = original-h
            minus = function(*[Tensor(a, False) for a in values]).data.copy()
            parameter[index] = original
            numerical[index] = (plus-minus)/(2*h)
        np.testing.assert_allclose(expected.grad, numerical, rtol=2e-5, atol=atol)


@pytest.mark.parametrize('function', [
    lambda a,b: (a+b).sum(), lambda a,b: (a-b).sum(),
    lambda a,b: (a*b).sum(), lambda a,b: (a/b).sum(),
    lambda a,b: (2+a+b).mean(), lambda a,b: (2-a+b).mean(),
    lambda a,b: (3*a+b).sum(), lambda a,b: (2/b+a).sum(),
])
@pytest.mark.parametrize('shape', [(), (3,), (1,3), (2,1,3)])
def test_arithmetic_broadcast(function, shape):
    a = np.arange(6, dtype=float).reshape(2,3)/7 + 0.3
    b = np.full(shape, 1.7)
    check_gradient(function, [a,b])


@pytest.mark.parametrize('axis,keepdims', [(None,False),(0,False),(-1,False),((0,2),False),((0,2),True),((),False)])
@pytest.mark.parametrize('operation', ['sum','mean'])
def test_reductions(axis, keepdims, operation):
    rng = np.random.default_rng(0)
    check_gradient(lambda a: (getattr(a,operation)(axis=axis,keepdims=keepdims)*1.7).sum(),
                   [rng.normal(size=(2,3,2))])


def test_matmul():
    rng = np.random.default_rng(2)
    check_gradient(lambda a,b: ((a@b).tanh()).sum(),
                   [rng.normal(size=(2,3)),rng.normal(size=(3,4))])


@pytest.mark.parametrize('operation', ['tanh','relu'])
def test_activations(operation):
    check_gradient(lambda a: getattr(a,operation)().sum(), [np.array([-3,-0.4,0.3,2.0])])


def test_relu_zero_convention():
    a=Tensor([0.0])
    a.relu().sum().backward()
    np.testing.assert_array_equal(a.grad, [0])


def test_cross_entropy():
    logits=np.array([[1.1,-0.3,0.7],[-1,0.2,2.3]])
    check_gradient(lambda a: cross_entropy(a,np.array([2,0])), [logits])


def test_cross_entropy_extreme_and_shift():
    logits=Tensor([[1000,-1000],[10000,10001]])
    objective=cross_entropy(logits,np.array([1,0]))
    objective.backward()
    assert np.isfinite(objective.data)
    assert np.isfinite(logits.grad).all()
    shifted=cross_entropy(logits+1e4,np.array([1,0]))
    np.testing.assert_allclose(objective.data,shifted.data)
    np.testing.assert_allclose(logits.grad.sum(axis=1),0,atol=1e-14)


def test_shared_paths_reset_and_deep_graph():
    x=Tensor(2)
    a=x*x
    out=a*x+a
    out.backward()
    assert x.grad == 16
    out.backward()
    assert x.grad == 16
    chain=x
    for _ in range(1500):
        chain=chain+1
    chain.backward()
    assert x.grad == 1


def test_vjp_and_saved_forward_values():
    x=Tensor([2,3])
    y=x*x
    x.data[:]=10
    y.backward(np.array([2,4]))
    np.testing.assert_array_equal(x.grad,[8,24])
    with pytest.raises(ValueError):
        y.backward()


def test_constant_and_errors():
    x=Tensor([2.0],False)
    (x*x).sum().backward()
    assert x.grad[0]==0
    with pytest.raises(ZeroDivisionError):
        x/0
    with pytest.raises(ValueError):
        cross_entropy(Tensor([[1,2]]),np.array([2]))
