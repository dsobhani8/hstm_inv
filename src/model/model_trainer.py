from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge
import sys
from scipy.special import expit
from mmd_loss import mmd_loss_weighted


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer():
	def __init__(self, model, model_name, use_pretrained, do_pretraining_stage, do_finetuning, save=False, load=False, model_file=None, is_MMD=False, MMD_pen_coeff=2, mmd=False, **kwargs):
		self.model = model
		self.model_name = model_name
		self.use_pretrained = use_pretrained
		self.do_pretraining_stage = do_pretraining_stage
		self.do_finetuning= do_finetuning
		self.save=save
		self.load=load
		self.model_file = model_file
		self.is_MMD = is_MMD
		self.MMD_pen_coeff= MMD_pen_coeff
		self.mmd = mmd
		print('here', self.mmd)

		self.beta_penalty = 1.0
		if use_pretrained:
			self.beta_penalty = 2.0

		self.set_l1_penalty_flags()

	def set_l1_penalty_flags(self):
		if self.model_name == 'hstm':
			self.penalty_bow = False
			self.penalty_gamma = True
		elif self.model_name == 'stm':
			self.penalty_bow=False
			self.penalty_gamma = False
		elif self.model_name == 'stm+bow':
			self.penalty_bow = True
			self.penalty_gamma = False
		elif self.model_name == 'hstm+bow':
			self.penalty_gamma=True
			self.penalty_bow = True
		elif self.model_name == 'hstm-all' or self.model_name == 'hstm-all-2stage':
			self.penalty_gamma = True
			self.penalty_bow = True
		elif self.model_name == 'hstm+topics':
			self.penalty_gamma = True
			self.penalty_bow = False
		elif self.model_name == 'hstm-nobeta':
			self.penalty_bow = True
			self.penalty_gamma = True

	def train(self, training_loader, epochs=10, extra_epochs=10, lr=0.01, weight_decay=1.2e-6):
		self.model.to(device)

		if self.load:
			self.model.load_state_dict(torch.load(self.model_file, map_location=device))
		elif self.model_name == 'prodlda' or self.model_name == 'slda':
			self.train_topic_model(training_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)
		elif self.model_name == 'hstm-all-2stage':
			self.train_regression_model(training_loader, epochs=epochs+extra_epochs, lr=lr, weight_decay=weight_decay)
		else:
			self.train_supervised_model(training_loader, epochs=epochs, extra_epochs=extra_epochs, lr=lr, weight_decay=weight_decay)

		if self.save:
			torch.save(self.model.state_dict(), self.model_file)

	def train_regression_model(self, training_loader, epochs=10, lr=0.01, weight_decay=1.2e-6):
		supervised_params = [self.model.gammas, 
			self.model.bow_weights.weight,
			self.model.bow_weights.bias,
			self.model.topic_weights.weight,
			self.model.topic_weights.bias]

		supervised_optim = optim.Adam(supervised_params, lr=lr ,weight_decay=weight_decay)

		for epoch in range(epochs):
			for _,data in enumerate(training_loader, 0):
				normalized_bow = data['normalized_bow'].to(device, dtype = torch.float)
				bow = data['bow'].to(device, dtype = torch.long)
				labels = data['label'].to(device, dtype = torch.float)
				pretrained_theta = data['pretrained_theta'].to(device, dtype = torch.float)
				recon_loss, supervised_loss, kld_theta = self.model(bow, normalized_bow, labels, 
					theta=pretrained_theta, penalty_bow=self.penalty_bow, penalty_gamma=self.penalty_gamma)

				total_loss = supervised_loss
				supervised_optim.zero_grad()
				total_loss.backward()
				supervised_optim.step()
				
				if _%5000==0:
					acc_loss = torch.sum(recon_loss).item()
					acc_kl_theta_loss = torch.sum(kld_theta).item()
					acc_sup_loss = torch.sum(supervised_loss).item()
					print("Epoch:", epoch, "Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss, "Supervised loss:", acc_sup_loss)

	def train_topic_model(self, training_loader, epochs=10, lr=0.01, weight_decay=1.2e-6):
		optimizer = optim.Adam(self.model.parameters(), lr=lr ,weight_decay=weight_decay)
		for epoch in range(epochs):
			self.model.train()
			for _,data in enumerate(training_loader, 0):
				normalized_bow = data['normalized_bow'].to(device, dtype = torch.float)
				bow = data['bow'].to(device, dtype = torch.long)
				
				if self.model_name == 'slda':
					labels = data['label'].to(device, dtype = torch.float)
					recon_loss, penalty, kld_theta = self.model(bow, normalized_bow, labels)
				else:
					recon_loss, penalty, kld_theta = self.model(bow, normalized_bow)
				optimizer.zero_grad()
				total_loss = recon_loss + penalty + self.beta_penalty*kld_theta

				total_loss.backward()
				optimizer.step()
				
				if _%5000==0:
					acc_loss = torch.sum(recon_loss).item()
					acc_kl_theta_loss = torch.sum(kld_theta).item()
					acc_sup_loss = torch.sum(penalty).item()
					print("Epoch:", epoch, "Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss, "Supervised loss:", acc_sup_loss)

	def compute_kernel(self, x, y, sigma_sqr=None):
		"""
		taken from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
		"""
		x_size = x.size(0)
		y_size = y.size(0)
		dim = x.size(1)
		# sigma_sqr = sigma_sqr if sigma_sqr is not None else dim / 2
		x = x.unsqueeze(1)  # (x_size, 1, dim)
		y = y.unsqueeze(0)  # (1, y_size, dim)
		tiled_x = x.expand(x_size, y_size, dim)
		tiled_y = y.expand(x_size, y_size, dim)
		kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(2 * sigma_sqr)
		return torch.exp(-kernel_input)  # (x_size, y_size)


	def compute_mmd(self, x, y):
		"""
		taken from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
		"""
		sigma_sqr = 10
		x_kernel = self.compute_kernel(x, x, sigma_sqr)
		y_kernel = self.compute_kernel(y, y, sigma_sqr)
		xy_kernel = self.compute_kernel(x, y, sigma_sqr)
		mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
		mmd = torch.maximum(torch.zeros_like(mmd), mmd)
		return mmd

	def train_supervised_model(self, training_loader, epochs=10, extra_epochs=10, lr=0.01, weight_decay=1.2e-6):
		if self.do_pretraining_stage:
			pretraining_optim = optim.Adam(self.model.parameters(), lr=lr ,weight_decay=weight_decay)
			for epoch in range(extra_epochs):
				self.model.train()
				for _,data in enumerate(training_loader, 0):
					normalized_bow = data['normalized_bow'].to(device, dtype = torch.float)
					bow = data['bow'].to(device, dtype = torch.long)
					labels = data['label'].to(device, dtype = torch.float)

					recon_loss, l1_penalty, kld_theta = self.model(bow, normalized_bow, labels, do_prediction=False)

					total_loss = recon_loss + l1_penalty + kld_theta

					pretraining_optim.zero_grad()
					total_loss.backward()
					pretraining_optim.step()

					if _%5000==0:
						acc_loss = torch.sum(recon_loss).item()
						acc_kl_theta_loss = torch.sum(kld_theta).item()
						print("Epoch:", epoch, "Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss)
						sys.stdout.flush()

		print('hola', self.mmd)
		for epoch in range(epochs):
			full_optimizer = optim.Adam(self.model.parameters(), lr=lr ,weight_decay=weight_decay)
			for _,data in enumerate(training_loader, 0):
				normalized_bow = data['normalized_bow'].to(device, dtype = torch.float)
				bow = data['bow'].to(device, dtype = torch.long)
				labels = data['label'].to(device, dtype = torch.float)
				confounder = data['topic'].to(device, dtype=torch.float)  # Add this line
				if self.mmd:
					balanced_weights_pos = data['balanced_weights_pos'].to(device, dtype=torch.float)
					balanced_weights_neg = data['balanced_weights_neg'].to(device, dtype=torch.float)
					balanced_weights = data['balanced_weights'].to(device, dtype=torch.float)

				if 'pretrained_theta' in data:
					pretrained_theta = data['pretrained_theta']
				else:
					pretrained_theta = None

				predictions, recon_loss, supervised_loss, kld_theta = self.model(bow, normalized_bow, labels, theta=pretrained_theta,
					penalty_bow=self.penalty_bow, penalty_gamma=self.penalty_gamma)

				# if self.is_MMD:
				# 	predictions_z0 = predictions[topics == 0]
				# 	predictions_z1 = predictions[topics == 1]
				#
				# 	mmd_loss = (self.compute_mmd(predictions_z0.reshape(-1, 1).to(torch.float64),
				# 								 predictions_z1.reshape(-1, 1).to(torch.float64)) * self.MMD_pen_coeff)
				#
				# 	total_loss = recon_loss + supervised_loss + self.beta_penalty*kld_theta + mmd_loss
				weighted_mmd_vals = []

				if self.mmd:
					embedding, recon_loss, supervised_loss, kld_theta = self.model(bow, normalized_bow, labels,
																					 theta=pretrained_theta,
																					 penalty_bow=self.penalty_bow,
																					 penalty_gamma=self.penalty_gamma)

					weighted_loss = balanced_weights * supervised_loss
					weighted_loss = torch.sum(weighted_loss) / torch.sum(balanced_weights)

					mmd_val = mmd_loss_weighted(embedding, confounder, balanced_weights_pos, balanced_weights_neg,
												sigma=10)
					weighted_mmd_vals.append(mmd_val[0])
					weighted_mmd = torch.stack(weighted_mmd_vals)
					#print('aqui', weighted_loss, balanced_weights)
					sl = (weighted_loss + (2.0 * weighted_mmd))
					total_loss = sl + recon_loss + self.beta_penalty*kld_theta

				else:
					mmd_loss = 0
					total_loss = recon_loss + supervised_loss + self.beta_penalty * kld_theta

				full_optimizer.zero_grad()
				total_loss.backward()
				full_optimizer.step()
				
				if _%5000==0:
					acc_loss = torch.sum(recon_loss).item()
					acc_kl_theta_loss = torch.sum(kld_theta).item()
					acc_sup_loss = torch.sum(supervised_loss).item()

					if self.mmd:
						acc_mmd_loss = weighted_mmd.item()
						print("Epoch:", epoch, "Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss, "Supervised loss:", acc_sup_loss,"MMD loss:", acc_mmd_loss, "Total loss:", total_loss.item())
					else:
						print("Epoch:", epoch, "Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss, "Supervised loss:", acc_sup_loss)
					sys.stdout.flush()


		if self.do_finetuning:
			supervised_params = [self.model.gammas, 
				# self.model.smoothing,
				self.model.bow_weights.weight,
				self.model.bow_weights.bias,
				self.model.topic_weights.weight,
				self.model.topic_weights.bias]

			supervised_optim = optim.Adam(supervised_params, lr=lr ,weight_decay=weight_decay)

			for epoch in range(extra_epochs):
				for _,data in enumerate(training_loader, 0):
					normalized_bow = data['normalized_bow'].to(device, dtype = torch.float)
					bow = data['bow'].to(device, dtype = torch.long)
					labels = data['label'].to(device, dtype = torch.float)
					recon_loss, supervised_loss, kld_theta = self.model(bow, normalized_bow, labels, 
						penalty_bow=self.penalty_bow, penalty_gamma=self.penalty_gamma)
					total_loss = supervised_loss
					supervised_optim.zero_grad()
					total_loss.backward()
					supervised_optim.step()
					
					if _%5000==0:
						acc_loss = torch.sum(recon_loss).item()
						acc_kl_theta_loss = torch.sum(kld_theta).item()
						acc_sup_loss = torch.sum(supervised_loss).item()
						print("Epoch:", epoch, "Acc. loss:", acc_loss, "KL loss.:", acc_kl_theta_loss, "Supervised loss:", acc_sup_loss)
						sys.stdout.flush()

	def reset_model_parameters(self, data, param_name):
		data_tensor = torch.tensor(data, dtype=torch.float)
		self.model.eval()
		with torch.no_grad():
			if param_name == 'bow_weights':
				self.model.bow_weights.weight.copy_(data_tensor)
			elif param_name == 'gammas':
				self.model.gammas.copy_(data_tensor.t())


	def evaluate_heldout_nll(self, test_counts, theta=None):
		make_eval_metrics = True
		if len(test_counts.shape) == 1:
			normalized_counts= (test_counts/test_counts.sum())[np.newaxis,:]
			make_eval_metrics = False
		else:
			normalized_counts= test_counts/test_counts.sum(axis=1)[:,np.newaxis]

		test_normalized_bow = torch.tensor(normalized_counts, dtype=torch.float).to(device)

		self.model.eval()
		with torch.no_grad():
			
			if theta is not None:
				theta = torch.tensor(theta, dtype=torch.float, device=device)
			else:
				theta, _ = self.model.get_theta(test_normalized_bow)

			predicted_x = self.model.decode(theta).cpu().detach().numpy()
		
		recon_loss = -(predicted_x * test_counts).sum(axis=1).mean()

		return recon_loss

	def evaluate_heldout_prediction(self, test_counts, test_labels, theta=None):
		make_eval_metrics = True
		if len(test_counts.shape) == 1:
			normalized_counts= (test_counts/test_counts.sum())[np.newaxis,:]
			make_eval_metrics = False
		else:
			normalized_counts= test_counts/test_counts.sum(axis=1)[:,np.newaxis]

		test_normalized_bow = torch.tensor(normalized_counts, dtype=torch.float).to(device)

		self.model.eval()
		with torch.no_grad():
			
			if theta is not None:
				theta = torch.tensor(theta, dtype=torch.float, device=device)
			else:
				theta, _ = self.model.get_theta(test_normalized_bow)

			if self.model_name == 'slda':
				_ = self.model.decode(theta)

			predictions = self.model.predict_labels(theta, test_normalized_bow).cpu().detach().numpy()

			if self.model.is_bool:
				predictions = expit(predictions)

		score = None
		if make_eval_metrics:
			if self.model.is_bool:
				score1 = metrics.roc_auc_score(test_labels, predictions)
				score2 = metrics.log_loss(test_labels, predictions, eps=1e-4)
				
				pred = predictions >= 0.5
				score3 = metrics.accuracy_score(test_labels, pred)
				score=(score1,score2,score3)
			else:
				score = metrics.mean_squared_error(test_labels, predictions)
		return score, predictions
