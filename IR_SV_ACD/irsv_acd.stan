data {
  int T;            // Number of time points
  int<lower=1> mxpq;         // max Value between p and q
  int<lower=0> p;            // Lag order for g
  int<lower=0> q;            // Lag order for psi
  int<lower=2> k;            // Number of elements in gamdirch and durpar
  vector[T] g;          // Durations data
  vector[T] y;         // Observed log-returns
  real<lower=0> alpha_dir;
}

parameters {
  real<lower=0> omega;
  real<lower=0> delta;
  simplex[k] durpar;
  real mu;
  real<lower=0, upper=1> sigmasq;
  vector[T] x_std;
  real<lower=0, upper=1> phistar;
}

transformed parameters {
  real<lower=-1, upper=1> phi;
  phi=2*phistar-1;
  vector[T] x = x_std*sqrt(sigmasq); //reparametrization
  x[1] /= sqrt(1 - phi*phi);
  x[1] += mu;
  for (t in 2:T) {
    x[t] *= sqrt((1 - phi^(2*ceil(g[t]))) / (1 - phi*phi));
    x[t] += mu+phi^ceil(g[t]) * (x[t - 1] - mu);
  }
}

model {
  vector[T] psi;
  // Priors
  phistar ~ beta(1, 1);
  sigmasq ~ inv_gamma(1, 1);
  mu ~ normal(0, 10);
  delta ~ gamma(0.001, 0.001);
  omega ~ gamma(0.001, 0.001);
  durpar ~ dirichlet(rep_vector(alpha_dir, k));
  
  // Likelihood for g
  for (t in 1:mxpq) {
    psi[t] = omega;
    g[t] ~ gamma(delta, delta/psi[t]);
  }
  
  for (t in (mxpq+1):T) {
    psi[t] = omega;
    for (i in 1:p) {
      psi[t] += durpar[i] * g[t-i];
    }
    for (i in 1:q) {
      psi[t] += durpar[p+i] * psi[t-i];
    }
    g[t] ~ gamma(delta, delta/psi[t]);
  }
  
  // Likelihood for x and y
  x_std ~ std_normal();
  y ~ normal(0, exp(x / 2));
}
