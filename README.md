# model-forensics-after-dark

Longitudinal control forensics — research code for the "what mediates alignment context rot?" study.

The `context_rot` package holds the pipeline: rollout generation, annotation, grading, branching/resampling, activation capture and replay, direction finding, and steering. The authoritative protocol is `execution_plan.md` (v2.2); every module cites the sections it implements.

```python
import context_rot
context_rot.PLAN_VERSION  # "2.2"
```
