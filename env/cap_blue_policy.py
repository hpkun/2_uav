"""CAP-Blue: exact balanced distance assignment, legacy flight control unchanged."""
from collections import Counter
from itertools import product
from typing import Mapping
import numpy as np
from .blue_policy import BluePolicy, GuidanceState, BLUE_IDS, RED_IDS
from .models import Aircraft


def balanced_assignment(blue: Mapping[str, Aircraft], red: Mapping[str, Aircraft]) -> dict[str, str]:
    """Enumerate <=256 mappings; minimize total Euclidean distance then ID tuple.

    Ties use an absolute 1e-9 metre tolerance relative to the exact minimum,
    not a running pairwise tolerance (which could drift across candidates).
    """
    bs=sorted(b for b in BLUE_IDS if b in blue and blue[b].state.alive)
    rs=sorted(r for r in RED_IDS if r in red and red[r].state.alive)
    if not bs or not rs: return {}
    def position(a): return np.array((a.state.x,a.state.y,a.state.h),dtype=float)
    distances={(b,r):float(np.linalg.norm(position(blue[b])-position(red[r]))) for b in bs for r in rs}
    candidates=[]
    for targets in product(rs,repeat=len(bs)):
        loads=Counter(targets)
        if max(loads.get(r,0) for r in rs)-min(loads.get(r,0) for r in rs)>1: continue
        candidates.append((sum(distances[b,r] for b,r in zip(bs,targets)),targets))
    minimum=min(cost for cost,_ in candidates)
    targets=min(t for cost,t in candidates if abs(cost-minimum)<=1e-9)
    return dict(zip(bs,targets))


class CAPBluePolicy(BluePolicy):
    TARGET_STRATEGY='coordinated_assignment'

    def reset(self,rng):
        super().reset(rng)
        self._prepared_step=-1
        self._last_assignment_step=-1
        self._alive_blue_ids=()
        self._assignment_distances={}
        return self.TARGET_STRATEGY

    def prepare_step(self,all_blue,all_red,decision_step):
        if isinstance(decision_step,bool) or not isinstance(decision_step,(int,np.integer)) or decision_step<0:
            raise ValueError('decision_step must be a nonnegative integer')
        decision_step=int(decision_step)
        if self._prepared_step==decision_step: return
        alive=tuple(sorted(b for b in BLUE_IDS if all_blue[b].state.alive))
        due=(decision_step % self.target_refresh_steps==0 or alive!=self._alive_blue_ids or
             any(super(CAPBluePolicy,self)._refresh_due(self._guidance_state[b],all_red,decision_step) for b in alive))
        if due:
            assignment=balanced_assignment(all_blue,all_red)
            self._assignment_distances={}
            for bid in BLUE_IDS:
                if bid not in assignment:
                    self._guidance_state[bid]=GuidanceState(); continue
                aid=assignment[bid]; blue,target=all_blue[bid],all_red[aid]
                heading,pitch=self._guidance_angles(blue,target)
                self._guidance_state[bid]=GuidanceState(aid,heading,pitch,decision_step,False)
                d=np.array((blue.state.x-target.state.x,blue.state.y-target.state.y,blue.state.h-target.state.h))
                self._assignment_distances[bid]=float(np.linalg.norm(d))
            self._last_assignment_step=decision_step
        self._alive_blue_ids=alive
        self._prepared_step=decision_step

    def _refresh_due(self,state,red,decision_step):
        # The cohort has already refreshed. Force flags set by boundary recovery
        # are consumed by next step's prepare, never a later aircraft's action.
        if decision_step==self._prepared_step: return False
        return super()._refresh_due(state,red,decision_step)

    def action(self,blue,red,decision_step):
        if decision_step!=self._prepared_step:
            raise RuntimeError('CAP-Blue requires prepare_step before all Blue actions')
        return super().action(blue,red,decision_step)

    def diagnostics(self,blue,red,decision_step):
        result=super().diagnostics(blue,red,decision_step)
        state=self._guidance_state[blue.aircraft_id]
        team_due=(decision_step!=self._prepared_step and
                  (decision_step%self.target_refresh_steps==0 or
                   any(super(CAPBluePolicy,self)._refresh_due(self._guidance_state[b],red,decision_step)
                       for b in self._alive_blue_ids)))
        result['blue_guidance_refresh_due']=bool(blue.state.alive and team_due)
        result.update(assigned_target_id=state.target_id,
                      assignment_distance=self._assignment_distances.get(blue.aircraft_id),
                      guidance_age=result['blue_guidance_age'],
                      guidance_refresh_due=result['blue_guidance_refresh_due'])
        return result

    def team_diagnostics(self,all_blue,all_red,decision_step):
        bs=[b for b in BLUE_IDS if all_blue[b].state.alive]
        rs=[r for r in RED_IDS if all_red[r].state.alive]
        loads=Counter(self._guidance_state[b].target_id for b in bs)
        due=(decision_step!=self._prepared_step and
             (tuple(sorted(bs))!=self._alive_blue_ids or decision_step%self.target_refresh_steps==0 or
              any(super(CAPBluePolicy,self)._refresh_due(self._guidance_state[b],all_red,decision_step) for b in bs)))
        return dict(alive_blue_count=len(bs),alive_red_count=len(rs),max_target_load=max(loads.values(),default=0),
                    guidance_refresh_due=bool(due),**{f'target_load_{r}':loads[r] for r in RED_IDS})

    def state_dict(self):
        state=super().state_dict()
        state.update(target_strategy=self.TARGET_STRATEGY,prepared_step=self._prepared_step,
                     last_assignment_step=self._last_assignment_step,alive_blue_ids=list(self._alive_blue_ids),
                     assignment_distances=dict(self._assignment_distances))
        return state

    def load_state_dict(self,payload):
        expected={'guidance_mode','target_refresh_steps','guidance_state','target_strategy','prepared_step',
                  'last_assignment_step','alive_blue_ids','assignment_distances'}
        if not isinstance(payload,Mapping) or set(payload)!=expected or payload['target_strategy']!=self.TARGET_STRATEGY:
            raise ValueError('CAP-Blue state contract mismatch')
        for f in ('prepared_step','last_assignment_step'):
            if isinstance(payload[f],bool) or not isinstance(payload[f],(int,np.integer)) or payload[f]<-1:
                raise ValueError(f'invalid CAP {f}')
        alive=payload['alive_blue_ids']; dd=payload['assignment_distances']
        if alive!=sorted(set(alive)) or any(b not in BLUE_IDS for b in alive): raise ValueError('invalid CAP alive Blue IDs')
        if any(b not in alive or not np.isfinite(d) or d<0 for b,d in dd.items()): raise ValueError('invalid CAP assignment distances')
        super().load_state_dict({k:payload[k] for k in ('guidance_mode','target_refresh_steps','guidance_state')})
        self._prepared_step=int(payload['prepared_step']); self._last_assignment_step=int(payload['last_assignment_step'])
        self._alive_blue_ids=tuple(alive); self._assignment_distances=dict(dd)
