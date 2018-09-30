import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    """Actor (Policy) Model."""



    def __init__(self, state_size, action_size, seed, fc1_units=254, fc2_units=128,fc3_units=64,fc4_units=32,fc5_units=16):

        """Initialize parameters and build model.

        Params

        ======

            state_size (int): Dimension of each state

            action_size (int): Dimension of each action

            seed (int): Random seed

            fc1_units (int): Number of nodes in first hidden layer

            fc2_units (int): Number of nodes in second hidden layer

        """

        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)

        self.fc2 = nn.Linear(fc1_units, fc2_units)

        #self.fc3 = nn.Linear(fc2_units,fc3_units)

        #self.fc4 = nn.Linear(fc3_units, fc4_units)

        #self.fc5 = nn.Linear(fc4_units, fc5_units)

        #self.fc6 = nn.Linear(fc5_units, action_size)
        self.fc3 = nn.Linear(fc2_units, action_size)



    def forward(self, state):

        """Build a network that maps state -> action values."""

        x = F.relu(self.fc1(state))

        x = F.relu(self.fc2(x))

        #x = F.relu(self.fc3(x))

        #x = F.relu(self.fc4(x))

        #x = F.relu(self.fc5(x))

        #return self.fc6(x)
        return self.fc3(x)
        


class Dueling_DQN(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64,fc3_units=32,fc4_units=32,fc5_units=16):
        super(Dueling_DQN, self).__init__()
        self.action_size =  action_size     
        self.seed = torch.manual_seed(seed)

        self.fc1_adv = nn.Linear(state_size, fc1_units)  
        self.conv2_drop1_adv = nn.Dropout(p=0.1)

        self.fc1_val = nn.Linear(state_size, fc1_units)  
        self.conv2_drop1_val = nn.Dropout(p=0.1) 
 
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.conv2_drop2_adv = nn.Dropout(p=0.1)

        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.conv2_drop2_val = nn.Dropout(p=0.1) 
        
        self.fc3_adv = nn.Linear(fc2_units, action_size)
        self.fc3_val = nn.Linear(fc2_units, 1)
        #self.fc3_val = nn.Linear(fc2_units, action_size)



    def forward(self, state):
        adv = F.relu(self.conv2_drop1_adv(self.fc1_adv(state)))
        val = F.relu(self.conv2_drop1_val(self.fc1_val(state)))

        adv = F.relu(self.conv2_drop2_adv(self.fc2_adv(adv)))
        val = F.relu(self.conv2_drop2_val(self.fc2_adv(val)))

        #adv = F.relu(self.fc1_adv(state))
        #val = F.relu(self.fc1_val(state))

        #adv = F.relu(self.fc2_adv(adv))
        #val = F.relu(self.fc2_adv(val))


        adv = self.fc3_adv(adv)
        val = self.fc3_val(val)

        x = val + adv - adv.mean(1).unsqueeze(1)
        #x = val + adv - adv.mean()
        return x


class Dueling_DQN6(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64,fc3_units=32,fc4_units=32,fc5_units=16):
        super(Dueling_DQN6, self).__init__()
        self.action_size =  action_size     
        self.seed = torch.manual_seed(seed)

        self.fc1_adv = nn.Linear(state_size, fc1_units)  
        self.conv2_drop1_adv = nn.Dropout(p=0.1)

        self.fc1_val = nn.Linear(state_size, fc1_units)  
        self.conv2_drop1_val = nn.Dropout(p=0.1) 
 
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.conv2_drop2_adv = nn.Dropout(p=0.1)

        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.conv2_drop2_val = nn.Dropout(p=0.1) 
 
        self.fc3_adv = nn.Linear(fc2_units, fc3_units)
        self.conv2_drop3_adv = nn.Dropout(p=0.1)

        self.fc3_val = nn.Linear(fc2_units, fc3_units)
        self.conv2_drop3_val = nn.Dropout(p=0.1) 

        self.fc4_adv = nn.Linear(fc3_units, fc4_units)
        self.conv2_drop4_adv = nn.Dropout(p=0.1)

        self.fc4_val = nn.Linear(fc3_units, fc4_units)
        self.conv2_drop4_val = nn.Dropout(p=0.1) 

        self.fc5_adv = nn.Linear(fc4_units, fc5_units)
        self.conv2_drop5_adv = nn.Dropout(p=0.1)

        self.fc5_val = nn.Linear(fc4_units, fc5_units)
        self.conv2_drop5_val = nn.Dropout(p=0.1) 
       
        self.fc6_adv = nn.Linear(fc5_units, action_size)
        #self.conv2_drop6_adv = nn.Dropout(p=0.1)
        self.fc6_val = nn.Linear(fc5_units, 1)
        #self.conv2_drop6_val = nn.Dropout(p=0.1) 
        #self.fc6_val = nn.Linear(fc5_units, action_size)



    def forward(self, state):
        adv = F.relu(self.conv2_drop1_adv(self.fc1_adv(state)))
        val = F.relu(self.conv2_drop1_val(self.fc1_val(state)))

        adv = F.relu(self.conv2_drop2_adv(self.fc2_adv(adv)))
        val = F.relu(self.conv2_drop2_val(self.fc2_adv(val)))

        adv = F.relu(self.conv2_drop3_adv(self.fc3_adv(adv)))
        val = F.relu(self.conv2_drop3_val(self.fc3_val(val)))

        adv = F.relu(self.conv2_drop4_adv(self.fc4_adv(adv)))
        val = F.relu(self.conv2_drop4_val(self.fc4_adv(val)))

        adv = F.relu(self.conv2_drop5_adv(self.fc5_adv(adv)))
        val = F.relu(self.conv2_drop5_val(self.fc5_adv(val)))

        #adv = F.relu(self.fc1_adv(state))
        #val = F.relu(self.fc1_val(state))

        #adv = F.relu(self.fc2_adv(adv))
        #val = F.relu(self.fc2_adv(val))

        #adv = F.relu(self.conv2_drop6_adv(self.fc6_adv(adv)))
        #val = F.relu(self.conv2_drop6_val(self.fc6_adv(val)))

        adv = self.fc6_adv(adv)
        val = self.fc6_val(val)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
        #x = val + adv - adv.mean()
        return x






