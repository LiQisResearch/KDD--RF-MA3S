
import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utilsdraw.utils import *

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

           

            # Train
            train_score, train_loss ,selected_node_list,aug_reward= self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
  
            selected_node_list=np.reshape(selected_node_list,(-1,self.env_params['problem_size']))
            aug_reward=np.reshape(aug_reward,(-1,1))

            np.savetxt("result//select//MSTSP"+str(self.env_params['problem_name'])+".txt",selected_node_list,fmt='%d')
            np.savetxt("result//distance//MSTSP"+str(self.env_params['problem_name'])+".txt",aug_reward,fmt='%f')


            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)
            
             # LR Decay
            self.scheduler.step()

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        selected_node_list=None
        aug_reward=torch.tensor(1000000000)
        
        trans_baseline=0
        break_flag=0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss,selected_node_list,aug_reward,trans_baseline,break_flag = self._train_one_batch(batch_size,selected_node_list,aug_reward,trans_baseline,break_flag)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 1000:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             avg_score,  avg_loss))
            if break_flag:
                break

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 avg_score, avg_loss))

        return score_AM.avg, loss_AM.avg,selected_node_list.detach().cpu().numpy().squeeze(),aug_reward.detach().cpu().numpy().squeeze()

    def _train_one_batch(self, batch_size,selected_node_list,aug_reward,trans_baseline,break_flag):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        
        prob_list = torch.zeros(size=(self.env_params['decoder_num'], batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)

            state, reward, reward_index, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, :, None]), dim=3)

        if trans_baseline:
            new_baseline=reward.float().mean(dim=2, keepdims=True)
        else:
            new_baseline=reward[reward_index,:,:].float().mean(dim=1, keepdims=True).unsqueeze(dim=0)
        # Loss
        ###############################################
        advantage = reward - new_baseline
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=3)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward[reward_index,:,:].max(dim=1)  # get best results from pomo
        score_mean = -reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        trans_prob=-loss_mean/score_mean
        if trans_prob<0.005:
            trans_baseline=1
            temp_break=(0.0025-trans_prob)/0.0025
            temp=torch.rand(1)
            if temp_break>temp:
                break_flag=1



        new_aug_reward = -reward.reshape(-1,1)
        new_selected_node_list=self.env._get_selected_node_list()

        if new_aug_reward.sum()>aug_reward.sum():
            new_aug_reward=aug_reward
            new_selected_node_list=selected_node_list




        return score_mean.item(), loss_mean.item(),new_selected_node_list,new_aug_reward,trans_baseline,break_flag
