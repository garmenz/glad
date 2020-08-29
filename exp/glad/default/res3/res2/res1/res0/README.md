This training's config is in config.json (batch size=4)
Each turn prediction is given that the highest prob of the value for the slot (contain all the slots in prediction). Then for each turn, it's labeled to be correct if all the predict values for the existing slots are correct. 
