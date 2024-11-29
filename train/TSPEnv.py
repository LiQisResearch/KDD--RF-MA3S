from dataclasses import dataclass
import torch
import math
from TSProblemDef import get_random_problems,augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 3)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.decoder_num = env_params['decoder_num']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.problems = get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 3)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)#
                # shape: (8*batch, problem, 4)
            else:
                raise NotImplementedError
        
        self.problems,self.record=polar_relativisation(self.problems)#RF
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (decoder, batch, pomo)
        self.selected_node_list = torch.zeros((self.decoder_num,self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (decoder, batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.decoder_num,self.batch_size, self.pomo_size, self.problem_size))
        # shape: (decoder, batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected.type(torch.long)
        # shape: (decoder, batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, :, None]), dim=3)
        # shape: (decoder, batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (decoder,batch, pomo)
        reward_whole=torch.tensor([])
        reward_whole_sum=torch.tensor([])
        for i in range(self.decoder_num):
            self.step_state.ninf_mask[i,self.BATCH_IDX, self.POMO_IDX, self.current_node[i,:,:]] = float('-inf')
            # shape: (decoder, batch, pomo, node)

            # returning values
            done = (self.selected_count == self.problem_size)
            if done:
                reward = -self._get_travel_distance(i)  # note the minus sign!
                reward_whole=torch.cat((reward_whole,reward.unsqueeze(0)),dim=0)
                reward_whole_sum=torch.cat((reward_whole_sum,reward.sum().unsqueeze(0)),dim=0)
            else:
                reward = None
                reward_index=None
        if done:
            reward_index=torch.max(reward_whole_sum,dim=0).indices#Find the decoder with the shortest path for the sum of batch and POMO
            
        return self.step_state, reward_whole, reward_index, done



    def _get_travel_distance(self,i):
        
        gathering_index = self.selected_node_list[i].unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)

        seq_expanded = self.record[:, None, :, [0,1]].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        seq_expanded = seq_expanded[:,:,:,[0,1]]

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances



def polar_relativisation(xy):
    #The rectangular coordinate xy is the input size:(batch size,problem size,2)
    #Ensure that the input order is stable
    points = xy.gather(dim=1,index=(xy[:,:,1].sort(stable=True)[1].unsqueeze(2)).repeat(1,1,2))
    sorted_points = points.gather(dim=1,index=(points[:,:,0].sort(stable=True)[1].unsqueeze(2)).repeat(1,1,2))

    #Zero-mean, double for accuracy
    xy_new = sorted_points.double() - sorted_points.double().mean(dim=1).unsqueeze(1)#(batch,points,xy)
    xy_new=xy_new.float()
    #Calculate two elements of polar coordinates
    xy_rho = torch.linalg.norm(xy_new, dim=2)#(batch,points(disance))
    xy_theta = torch.atan(xy_new[:,:,1] / xy_new[:,:,0])#(batch,points(angle))
    xy_theta[xy_new[:,:,0]<0]=xy_theta[xy_new[:,:,0]<0]+math.pi#The polar Angle is extended to the full quadrant
    temp=xy_rho.max(dim=1).values
    xy_rho_new=(xy_rho/(temp.unsqueeze(dim=1)))#Polar coordinates rho normalization(batch,points(disance))
    #sort
    xy_rho_new,sort_indices=xy_rho_new.sort(dim=1,stable=True)
    xy_theta_new=torch.gather(xy_theta,1,sort_indices)#(batch,points(angle))

    xy_rho=xy_rho.sort(dim=1,stable=True).values

    angle = xy_theta_new - xy_theta_new[:,0].unsqueeze(1)#(batch,points(angle),points(angle))
    angle[angle < 0] += 2 * torch.tensor([math.pi])

    xy_result=torch.cat(((xy_rho_new*torch.cos(angle)).unsqueeze(dim=2),(xy_rho_new*torch.sin(angle)).unsqueeze(dim=2)),dim=2)
    record=torch.cat(((xy_rho*torch.cos(xy_theta_new)).unsqueeze(dim=2),(xy_rho*torch.sin(xy_theta_new)).unsqueeze(dim=2)),dim=2)
    return xy_result,record