import numpy as np
from engine import Tensor
from nn import Adam, Parameter


def test_masked_raw_gradient_differs_from_effective_gradient():
    w=Parameter([1.0,2.0],True)
    opt=Adam([w])
    opt.set_mask(w,[1,0])
    effective=w.effective()
    ((effective-3)*(effective-3)).sum().backward()
    np.testing.assert_array_equal(effective.data,[1,0])
    np.testing.assert_array_equal(effective.grad,[-4,-6])
    np.testing.assert_array_equal(w.grad,[-4,0])
    # Perturbing an inactive raw weight cannot change the masked objective.
    w.data[1]=0.001
    plus=((w.effective()-3)*(w.effective()-3)).sum().data
    w.data[1]=-0.001
    minus=((w.effective()-3)*(w.effective()-3)).sum().data
    assert plus == minus
    opt.step()
    assert w.data[1]==0


def test_prune_and_revive_matches_fresh_adam_including_age():
    w=Parameter([1.,1.],True)
    opt=Adam([w],lr=0.02)
    for _ in range(8):
        w.grad[:]=[2,3]
        opt.step()
    opt.set_mask(w,[1,0])
    for _ in range(5):
        w.grad[:]=[1,0]
        opt.step()
    assert w.data[1]==0
    assert opt.state[w]['age'][0]==13
    for state in opt.state[w].values():
        assert state[1]==0
    opt.set_mask(w,[1,1])
    fresh=Parameter([0.0],True)
    fresh_opt=Adam([fresh],lr=0.02)
    for gradient in [2.,-1.,3.]:
        w.grad[:]=[0,gradient]
        fresh.grad[:]=gradient
        opt.step()
        fresh_opt.step()
        np.testing.assert_allclose(w.data[1],fresh.data[0],rtol=1e-13,atol=1e-13)
        for name in ('m','v','age'):
            np.testing.assert_allclose(opt.state[w][name][1],fresh_opt.state[fresh][name][0])


def test_mask_is_snapshot_and_state_is_not_reset_for_unchanged_active_entries():
    w=Parameter([2.,3.],True)
    opt=Adam([w])
    w.grad[:]=1
    opt.step()
    saved=opt.state[w]['m'].copy()
    opt.set_mask(w,[1,1])
    np.testing.assert_array_equal(opt.state[w]['m'],saved)
    external=w.mask
    external[:]=False
    assert w.mask.all()


def test_adam_against_explicit_scalar_reference():
    p=Parameter([1.0])
    opt=Adam([p],lr=0.03)
    value,m,v=1.,0.,0.
    for t,g in enumerate([2.,-3.,1.],1):
        m=.9*m+.1*g
        v=.999*v+.001*g*g
        value-=.03*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        p.grad[:]=g
        opt.step()
        np.testing.assert_allclose(p.data,[value],rtol=1e-13)
