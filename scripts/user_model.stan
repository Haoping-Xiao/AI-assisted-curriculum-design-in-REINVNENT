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

// functions{
//   int choice_rng(row_vector prob){
//     return categorical_rng(to_vector(prob));
//   }
// }

data{
  int <lower=0> J;  // number of components in the pool
  int <lower=0> K; // K interaction
  matrix[K, J] prior_activity;
  matrix[K, J] prior_qed;
  matrix[K, J] prior_sa;
  
  matrix<lower=0,upper=1>[K, J] observed_activity; // observed activity scores
  matrix<lower=0,upper=1>[K, J] observed_qed; // observed qed scores
  matrix<lower=0,upper=1>[K, J] observed_sa; // observed sa scores
  
  int prior_choice[K];
  int <lower=0> advice [K];
  int <lower=0, upper=1> decision[K]; 
  
}

// transformed data{
//   int prior_choice[K];
//   for (i in 1:K){
//     prior_choice[i] =categorical_rng(to_vector(prior_prob[i]));
//   }
// }


parameters{
  simplex[3] w;  // weights for activity, qed, sa
  real<lower=1,upper=4> beta1; // bias parameter in prior knowledge
  real<lower=10,upper=40> beta2; // bias parameter in prior knowledge
}






model{
  
  real advantage[K];
  row_vector[J] observed_scores;
  matrix[K, J] prior_prob;
  row_vector[J] prior_scores;
  w ~ normal(.5, .15);
  beta1~uniform(1,4);
  beta2~uniform(10,40);
  

  
  for (i in 1:K){
    prior_scores=w[1]* prior_activity[i]+ w[2]*prior_qed[i] + w[3]* prior_sa[i];
    prior_prob[i]=to_row_vector(softmax(to_vector(prior_scores*beta1)));
    prior_choice[i] ~ categorical(to_vector(prior_prob[i]));
    // prior_choice[i] =choice_rng(prior_prob[i]);
    // prior_choice[i] =categorical_rng(to_vector(prior_prob[i]));
    observed_scores= w[1]*observed_activity[i]+w[2]*observed_qed[i]+ w[3]*observed_sa[i];
    advantage[i]=observed_scores[advice[i]]- observed_scores[prior_choice[i]];
    decision~bernoulli_logit(beta2*advantage[i]);
  }

}




