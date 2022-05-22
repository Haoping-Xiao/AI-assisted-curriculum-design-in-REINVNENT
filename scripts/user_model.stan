//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data{
  int <lower=0> J;  // number of components in the pool
  int <lower=0> K; // K interaction
  matrix[K, J] prior_activity;
  matrix[K, J] prior_qed;
  matrix[K, J] prior_sa;
  int prior_choice[K];
}



parameters{
  simplex[3] w;  // weights for activity, qed, sa
  real<lower=1,upper=4> beta1; // bias parameter in prior knowledge
}






model{
  matrix[K, J] prior_prob;
  row_vector[J] prior_scores;
  w ~ normal(.5, .2);
  beta1~uniform(1,4);
  

  
  for (i in 1:K){
    prior_scores=w[1]* prior_activity[i]+ w[2]*prior_qed[i] + w[3]* prior_sa[i];
    prior_prob[i]=to_row_vector(softmax(to_vector(prior_scores*beta1)));
    for (j in 1:J){
      if(prior_prob[i][j]==0){
        prior_prob[i][j]=0.0000001;
      }
    }
    prior_choice[i] ~ categorical(to_vector(prior_prob[i]));
  }

}




