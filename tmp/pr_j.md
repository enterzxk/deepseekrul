
## J. Observation span as a first-order design variable

The protocol fixes the observation window at 50 cycles, a convention inherited from the
C-MAPSS literature. We tested whether that convention is leaving accuracy on the table.

**Effect of the span.** On the windows where a longer history is available, replacing the
50-cycle window by a 100-cycle window reduces the pooled error from 11.03 to 7.50 cycles
(FD002, windows with at least 100 observed cycles); a 150-cycle window reaches 4.97 on the
windows where it applies. The comparison is made on identical windows, so it is not an
artefact of a changing evaluation set.

**Schedule.** Because a longer window is only available once the engine has been observed
long enough, we use a schedule: windows ending before 100 cycles use the 50-cycle model,
windows ending at or after 100 cycles use the 100-cycle model. Both models are trained
with the frozen budget (80 epochs, all available windows, fixed final epoch, no selection).

**Controls.** Three controls separate the span from competing explanations. (i) A late-window
specialist trained at the same 50-cycle window on the same late windows reaches 10.65
against 11.03 for the generalist and 7.50 for the 100-cycle model, so specialisation alone
is worth 3.5% and the span is worth the rest. (ii) A span control with identical parameter
count (50 positions sampled every second cycle over 100 cycles) reaches 9.62, so both the
span and the temporal resolution contribute. (iii) A three-seed ensemble of the 50-cycle
reference, which has the same model count as the schedule, reaches 12.53 on the fold where
the schedule reaches 11.80, so the gain is not a model-count effect.

**Boundaries.** The gain is concentrated in long-lived engines: -10% to -11% on engines
above the first lifetime quartile against -2% below it, because short-lived engines reach
100 observed cycles only near the end of their life. The pooled error is dominated by early
windows, where no longer history exists, so the whole-window gain is necessarily smaller
than the late-window gain.

**A leakage trap we avoided.** Enforcing a monotone decreasing RUL along the trajectory by
isotonic regression lowers the pooled error by 12.5%, but the fit uses windows that occur
after the one being predicted. The causal version, which fits only on the prefix available
at each prediction time, is worth 2.5% on FD002 and is slightly negative on FD004. We report
the causal result only.
