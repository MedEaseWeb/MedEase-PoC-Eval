n_shot_classification is used as guardian to prevent unintended use of our models. Before actual API calls, n_shot_clf is used to identify if the user input is medical report or not.

task: given a ~300 tokens input text, classify it as (medical report / other)
