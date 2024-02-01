set.seed(as.integer(1234))

library(cmdstanr)
library(loo)
library(Metrics)

file <- "irsv_acd.stan"

# Compile the stan model
model <- cmdstan_model(stan_file = file,cpp_options = list(stan_threads = TRUE))

##############################################
## Simulation Setup for IR-SV-ACD(1,1)
##############################################

T <- 5020  # Number of time points
p <- 1  # Lag order for g
q <- 1  # Lag order for psi
k <- p+q+1
mxpq <- max(p,q)

# Parameter values for simulations
omega <- 0.1
delta <- 2.2
phiStar <- 0.9
sigma <- 0.8
mu <- -9
durpar <- c(0.2, 0.7)

# Simulate g values using a gamma distribution
g <- numeric(T)
psi_t <- numeric(T)
for (t in 1:T) {
  if (t <= mxpq) {
    psi_t[t] <- omega
    g[t] <- rgamma(1, shape = delta, scale = psi_t[t] / delta)
  } else {
    psi_t[t] <- omega + sum(durpar[1:p] * g[(t-1):(t-p)]) + sum(durpar[(p+1):(p+q)] * psi_t[(t-1):(t-q)])
    g[t] <- rgamma(1, shape = delta, scale = psi_t[t] / delta)
  }
}

# Simulate y values using a normal distribution
y <- numeric(T)
x <- numeric(T)
for (t in 1:T) {
  if (t == 1) {
    x[t] <- rnorm(1, mean = mu, sd = sqrt(sigma / (1 - phiStar^2)))
  } else {
    x[t] <- rnorm(1, mean = mu + phiStar^ceiling(g[t]) * (x[t-1] - mu), 
                  sd = sqrt(sigma * (1 - phiStar^(2*ceiling(g[t])))/ (1 - phiStar^2)))
  }
  y[t] <- rnorm(1, mean = 0, sd = exp(0.5 * x[t]))
}

# create a data list
data_list <- list(
  T = T-20,
  mxpq = mxpq,
  p = p,
  q = q,
  k = k,
  g = g[1:5000],
  y = y[1:5000],
  alpha_dir=0.5
)

options(mc.cores=4) # choose the number of cores you want to run the model on 
fit=model$sample(data=data_list,seed=1,chains = 4,iter_sampling = 1000,parallel_chains = 4,threads_per_chain = 12,iter_warmup=1000,thin=4)
par_summary=fit$summary(variables = c("mu", "phi","sigmasq","omega","durpar","delta"))
summary=cbind(par_summary,fit$summary(c("mu", "phi","sigmasq","omega","durpar","delta"), quantile, .args = list(probs = c(0.025, .975)))[,c(2,3)])
summary
fit$diagnostic_summary()
samples=fit$draws(variables = c("mu", "phi","sigmasq","omega","durpar","delta"),format = "df")
nSamp <- nrow(samples)
lpd=as.matrix(fit$lp(),nrow=nSamp,ncol=1)
waic_est=waic(lpd)

# Save the output you want as an RDS
filename1 <-sprintf("/Results/ppsampfull_g11.rds")
saveRDS(object = samples, file = filename1)

##############################################
## Fitting
##############################################

# Using the posterior samples from the model we can obtain the fitted values
delta_fit <- fit$draws("delta")
omega_fit <- fit$draws("omega")
alpha_fit <- fit$draws(c("durpar[1]"))
betastar_fit <- fit$draws(c("durpar[2]"))
phi_fit <- fit$draws("phi")
sigma_fit <- fit$draws("sigmasq")
mu_fit <- fit$draws("mu")

nSamp <- nrow(samples)
n <- 5000
gppSamples <- matrix(0, nSamp, n)
ppSamples <- matrix(0, nSamp, n)
psifits <- matrix(0, nSamp, n)
lt_st <- matrix(0, nSamp, n)

psi_new <- rep(NA,n)
gppSamples_new <- rep(NA,n)
gj_fit <- rep(NA,n)
ltst_fit <- rep(NA,n)
rt_fit <- rep(NA,n)

for(id in 1:nSamp){
  delta <- delta_fit[id]
  omega <- omega_fit[id]
  alpha <- alpha_fit[id]
  beta <- betastar_fit[id]
  mu <- mu_fit[id]
  sigma <- sigma_fit[id]
  phi <- phi_fit[id]
  T=5000
  
  for (t in 1:mxpq){
    psi_new[t] <- omega
    gppSamples_new[t] <- rgamma(n=1, shape=delta, scale=psi_new[t]/delta)
  }
  for (t in (mxpq+1):(T)){
    psi_new[t] <- omega+alpha[1:p]%*%g[(t-1):(t-p)]+beta[1:q]%*%psi_new[(t-1):(t-q)]
    gppSamples_new[t] <- rgamma(n=1, shape=delta, scale=psi_new[t]/delta)
  }
ltst_fit[1] <- rnorm(n=1, mean=mu, sd = sqrt(sigma / (1-phi*phi)))
rt_fit[1] <-  rnorm(n=1, mean= 0, sd = exp(0.5 * ltst_fit[1]))
for (t in 2:T){
  ltst_fit[t] <- rnorm(n=1, mean=mu+phi^ceiling(gppSamples_new[t]) * (x[t-1]-mu)
                       , sd = sqrt(sigma*(1-phi^(2*ceiling(gppSamples_new[t]))) / (1-phi*phi)))
  rt_fit[t] <-  rnorm(n=1,mean = 0,sd=exp(0.5 * ltst_fit[t]))
}
  gppSamples[id,] <- gppSamples_new
  psifits[id,] <- psi_new
  lt_st[id,] <- ltst_fit
  ppSamples[id,] <- rt_fit
}

##############################################
## psi fits
post_fit <- apply(psifits, 2,function(x) mean(x,na.rm=T))
psi_mae_train=mae(psi_t[1:5000], post_fit)
psi_mdae_train=mdae(psi_t[1:5000], post_fit)

##############################################
## gap fits
post_fitg <- apply(gppSamples, 2,function(x) mean(x,na.rm=T))
gaps_mae_train=mae(g[1:5000], post_fitg)
gaps_mdae_train=mdae(g[1:5000], post_fitg)
gaps_mse_train=mse(g[1:5000], post_fitg)

##############################################
## latent state fits
post_fitx <- apply(lt_st, 2,function(x) mean(x,na.rm=T))
lt_mae_train=mae(x[1:5000], post_fitx)
lt_mdae_train=mdae(x[1:5000], post_fitx)

##############################################
## Export the evaluation metrics
foreval_train=cbind(psi_mae_train, gaps_mae_train, lt_mae_train, psi_mdae_train, gaps_mdae_train, lt_mdae_train)

filename3 <- sprintf("/Results/maetrain_g11acd.rds")
saveRDS(object = foreval_train, file = filename3)

## Export the fitted values
fits_train=cbind(post_fit, post_fitg, post_fitx)

filename4 <- sprintf("/Results/fitstrain_g11acd.rds")
saveRDS(object = fits_train, file = filename4)


##############################################
## Forecast
##############################################

# Using the posterior samples from the model we can obtain the forecast values
nSamp <- nrow(samples)
n <- 20
gaps_fore <- matrix(0, nSamp, n)
ppSamples_fore <- matrix(0, nSamp, n)
psij_fore <- matrix(0, nSamp, n)
lt_st_fore <- matrix(0, nSamp, n)

psi_fore <- rep(NA,n)
gppSamples_fore <- rep(NA,n)
gj_fore <- rep(NA,n)
ltst_fore <- rep(NA,n)
rt_fore <- rep(NA,n)

for(id in 1:nSamp){
  delta <- delta_fit[id]
  omega <- omega_fit[id]
  alpha <- alpha_fit[id]
  beta <- betastar_fit[id]
  mu <- mu_fit[id]
  sigma <- sigma_fit[id]
  phi <- phi_fit[id]
  N=5000
  T=20
  psi_pp <- psifits[id,c((N):(N+1-q))]
  
  for (t in 1:mxpq){
    psi_fore[t] <- omega+alpha[1:p]%*%g[(5000+t-1):(5000+t-p)]+beta[1:q]%*%psi_pp
    gppSamples_fore[t] <- rgamma(n=1, shape=delta, scale=psi_fore[t]/delta)
  }
  for (t in (mxpq+1):(T)){
    psi_fore[t] <- omega+alpha[1:p]%*%gppSamples_fore[(t-1):(t-p)]+beta[1:q]%*%psi_fore[(t-1):(t-q)]
    gppSamples_fore[t] <- rgamma(n=1, shape=delta, scale=psi_fore[t]/delta)
  }
ltst_fore[1] <-  rnorm(n=1, mean=mu+phi^ceiling(gppSamples_fore[1]) * (x[5000]-mu), 
                       sd = sqrt(sigma*(1-phi^(2*ceiling(gppSamples_fore[1]))) / (1-phi*phi)))
rt_fore[1] <-  rnorm(n=1, mean= 0, sd = exp(0.5 * ltst_fore[1]))
for (t in 2:T){
  ltst_fore[t] <- rnorm(n=1, mean=mu+phi^ceiling(gppSamples_fore[t]) * (ltst_fore[t-1]-mu), 
                        sd = sqrt(sigma*(1-phi^(2*ceiling(gppSamples_fore[t]))) / (1-phi*phi)))
  rt_fore[t] <-  rnorm(n=1,mean = 0,sd=exp(0.5 * ltst_fore[t]))
}
  gaps_fore[id,] <- gppSamples_fore
  psij_fore[id,] <- psi_fore
  lt_st_fore[id,] <- ltst_fore
  ppSamples_fore[id,] <- rt_fore
}

##############################################
## psi forecasts
post_fore_psitest <- apply(psij_fore,2,function(x) mean(x)) # we can also obtain the 95% quantile similarly
psi_mae_test=mae(psi_t[5001:5020], post_fore_psitest)
psi_mdae_test=mdae(psi_t[5001:5020], post_fore_psitest)

##############################################
## gaps forecasts
post_fore_g <- apply(gaps_fore,2,function(x)  mean(x)) # we can also obtain the 95% quantile similarly
gaps_mae_test=mae(g[5001:5020], post_fore_g)
gaps_mdae_test=mdae(g[5001:5020], post_fore_g)

##############################################
## latent state forecasts
post_fore_ht <- apply(lt_st_fore,2,function(x)  mean(x)) # we can also obtain the 95% quantile similarly
lt_mae_test=mae(x[5001:5020], post_fore_ht)
lt_mdae_test=mdae(x[5001:5020], post_fore_ht)

##############################################
## Export the forecast evaluation metrics
foreval_test=cbind(psi_mae_test, gaps_mae_test, lt_mae_test, psi_mdae_test, gaps_mdae_test, lt_mdae_test)

filename5 <- sprintf("/Results/maetest_g11acd.rds")
saveRDS(object = foreval_test, file = filename5)

## Export the forecasts
fits_test=cbind(post_fore_psitest, post_fore_g, post_fore_ht)

filename6 <- sprintf("/Results/foretest_g11acd.rds")
saveRDS(object = fits_test, file = filename6)

