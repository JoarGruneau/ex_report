for epoch in epochs:
  update_generative_network
  if epoch % check_discriminative_network == 0:
    while discriminative_loss > cutoff:
      update_discriminative_network
