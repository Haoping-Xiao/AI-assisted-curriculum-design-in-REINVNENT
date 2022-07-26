# AI-assisted-curriculum-design-in-REINVNENT

This is the implementation of my master thesis "AI-assisted curriculum learning".

human.py is the entry to create a curriculum for sparse reward problems in de novo molecular design. It involves interactions between human and an AI assistant.
The AI assistant doesn't know the private information of the human, such as reward parameters and biases. The AI assistant helps the human by recommendation. The human designs a curriculum with the help of the AI assistant.

mdp.py defines states, actions and a tree for MCTS.

run_curriculum.py is an automation that executes curriculum learning.

scorer.py sets up three scorers, including QED(quantitative estimate of drug-likeness), DRD2 acitivy, synthetic accessibility.

stan_model.py collects AI observation to learn a user model defined in user_model.stan.

statistic.py helps create components (i.e.: curriculum objectives).

utils.py defines class and functions that are frequently used.

enums.py enumerates the curriculum objectives, curriculum design objective function and project configuration.


