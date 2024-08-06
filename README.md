### Adapted from https://ykunisato.github.io/lcmr/ ###

Latent Cause Model of Associative Learning

Bayesian Inference with an infinite capacity prior (Infinite Mixture Model)
Chinese Restaurant Process Prior

Prior belief:
- If a latent cause occurs more frequently, it is more likely to cause an observation
- Number of possible latent causes is unbounded


### Static partitioning all the data into latent causes ###
P: Probability of latent cause
C_t: Latent cause of trial t
k: Number of latent causes
N_k: Number of events already caused by latent cause k
alpha: probability of a completely new latent cause

P(C_t = k) = if k is old cause: N_k / (t + alpha)
            = if k is new cause: alpha / (t + alpha)

Likelihood:
- Each latent cause creates similar trials
- How likely is it that the observation was caused by each of the latent causes you previously inferred?

In acquisition the animals infer latent cause X and in extinction they infer different latent cause Y (less likely that the trials in extinction are caused by X, i.e. box is different, no shocks etc.)

Not a model of change over time.
Sampling from latent causes, order is not important.
The more I sample from a latent cause, the more likely it is that I've seen all the latent causes there are.
t grows with time, the prob. of a new latent cause gets smaller.
