import stan

user_model="""
data{
  int <lower=0> J;  // number of components in the pool


  real <lower=0,upper=1>prior_activity[J]; // prior activity scores
  real <lower=0,upper=1>prior_qed[J]; // prior qed scores
  real <lower=0,upper=1>prior_sa[J]; // prior sa scores


  real <lower=0,upper=1>observed_activity[J]; // observed activity scores
  real <lower=0,upper=1>observed_qed[J]; // observed qed scores
  real <lower=0,upper=1>observed_sa[J]; // observed sa scores

}

parameters{

  simplex[3] w;  // weights for activity, qed, sa
  real<lower=1,upper=4> beta1; // bias parameter in prior knowledge
  real<lower=10,upper=40> beta2; // bias parameter in prior knowledge
}


model {  
  w ~ normal(.5, .15);
  beta1~uniform(1,4);
  beta2~uniform(10,40);
  
}





"""