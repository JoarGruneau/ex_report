for epoch in epochs:
	update_generative_network
	if epoch % check_discriminative_network == 0:
		while discriminative_network_loss > cut_off:
			update_discriminative_network
