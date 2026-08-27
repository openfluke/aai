Legend: `[T]=Tween` · `[S]=Split` · `[FP]=FastProxy` · `[L]=Linear` · `[HP]=HeadProxy`

| # | Full name (Welvet) | Short | Ckpt / CLI token |
|---|--------------------|-------|------------------|
| 1 | NormalBP | NormalBP | `sgd` |
| 2 | StepBP | StepBP | `step_sgd` |
| 3 | Tween | [T] | `tween` |
| 4 | TweenChain | [T]Chain | `tween_chain` |
| 5 | StepTween | Step[T] | `step_tween` |
| 6 | StepTweenChain | Step[T]Chain | `step_tween_chain` |
| 7 | MeshBP | MeshBP | MeshBP |
| 8 | MeshTween | Mesh[T] | MeshTween |
| 9 | MeshTweenChain | Mesh[T]Chain | MeshTweenChain |
| 10 | TweenSplit | [T][S] | TweenSplit |
| 11 | StepTweenSplit | Step[T][S] | StepTweenSplit |
| 12 | TweenAlt | [T]Alt | TweenAlt |
| 13 | StepTweenAlt | Step[T]Alt | StepTweenAlt |
| 14 | TweenSplitHeadProxy | [T][S][HP] | TweenSplitHeadProxy |
| 15 | TweenSplitLinear | [T][S][L] | TweenSplitLinear |
| 16 | TweenSplitFastProxy | [T][S][FP] | TweenSplitFastProxy |
| 17 | TweenSplitLinearCache | [T][S][L]Cache | TweenSplitLinearCache |
| 18 | TweenSplitHeadProxyAsync | [T][S][HP]Async | TweenSplitHeadProxyAsync |
| 19 | TweenSplitSparse | [T][S]Sparse | TweenSplitSparse |
| 20 | MeshTweenSplit | Mesh[T][S] | MeshTweenSplit |
| 21 | MeshTweenAlt | Mesh[T]Alt | MeshTweenAlt |
| 22 | MeshTweenSplitFastProxy | Mesh[T][S][FP] | MeshTweenSplitFastProxy |
| 23 | MeshTweenSplitSparse | Mesh[T][S]Sparse | MeshTweenSplitSparse |
| 24 | StepTweenSplitHeadProxy | Step[T][S][HP] | StepTweenSplitHeadProxy |
| 25 | StepTweenSplitLinear | Step[T][S][L] | StepTweenSplitLinear |
| 26 | StepTweenSplitFastProxy | Step[T][S][FP] | StepTweenSplitFastProxy |
| 27 | StepTweenSplitLinearCache | Step[T][S][L]Cache | StepTweenSplitLinearCache |
| 28 | StepTweenSplitHeadProxyAsync | Step[T][S][HP]Async | StepTweenSplitHeadProxyAsync |
| 29 | StepTweenSplitSparse | Step[T][S]Sparse | StepTweenSplitSparse |

That’s the full **29** test53 / tide `AllModes` set. First six keep Lucy ckpt aliases (`sgd`, `step_sgd`, …); the rest use the Welvet `String()` name as the token.