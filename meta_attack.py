import torch
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from few_shot.eval import evaluate
from few_shot.datasets import MiniImageNet
from few_shot.core import NShotTaskSampler, create_nshot_task_label
from few_shot.models import FewShotClassifier
from few_shot.maml import meta_gradient_step
from torchvision.utils import save_image
import numpy as np
from numpy import linalg


num_input_channels=256
device = torch.device('cuda')
dataset_class=MiniImageNet
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, 10, n=5, k=1, q=5,
                                   num_tasks=1),
    num_workers=8
)
model = FewShotClassifier(3, 2, 256).to(device, dtype=torch.double)
#print(model)
model.load_state_dict(torch.load('./models/maml/miniImageNet_order=1_n=5_k=2_metabatch=4_train_steps=5_val_steps=10.pth'))
def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k,3, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_

#metrics = evaluate(model, evaluation_taskloader, prepare_meta_batch(5,2,5,4), metrics=['categorical_accuracy'])
#print(metrics)
def imshow(img):
    img = img /2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def get_splits():
    # naive way to generate splits, should fix to be better later
    split = np.array([0,0,0,0,0,0,0,0,0,0])
    temp =  np.array([0,0,0,0,0,0,0,0,0,0])
    for i in range(0,10):
        for j in range(i+1,10):
            for l in range(j+1,10):
              for k in range(l+1,10):
                    for m in range(k+1,10):
                       split[i]=1
                       split[j]=1
                       split[l]=1
                       split[k]=1
                       split[m]=1
                       temp = np.vstack((temp,split))
                       split=[0,0,0,0,0,0,0,0,0,0]
    return np.delete(temp,0,axis=0)

def max_update(model, x, epsilon, eps_optim, splits, loss_fn):
    # train meta learner on support set
    fast_weights = OrderedDict(model.named_parameters())
    for train_steps_inner in range(0,7):
        logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1],fast_weights)
        y = create_nshot_task_label(1,len(x[0][splits==1])).to(device)
        loss = loss_fn(logits,y)
        gradients = torch.autograd.grad(loss,fast_weights.values(),create_graph=True, retain_graph=True)
        fast_weights = OrderedDict((name, param - 0.001 * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))

    # optimize to maximize gap loss
    y_q = create_nshot_task_label(1,len(x[0][splits==0])).to(device)
    q_logits = model.functional_forward(x[0][splits==0]+epsilon[splits==0], fast_weights)
    s_logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1], fast_weights)    


    loss = -1 * torch.sum(loss_fn(q_logits, y_q) - loss_fn(s_logits, y))
    #print('Query loss: ',loss_fn(q_logits, y_q))
    #print('Support los: ',loss_fn(s_logits, y) )
        
    loss.backward()
    eps_optim.step()
    eps_optim.zero_grad()

    # bound epsilons
    epsilon.data = torch.clamp(epsilon,-0.1,0.1)  
    epsilon.data = (torch.clamp(x[0]+ epsilon,0.,1.) - x[0])  

    return epsilon.data 

def update_graph(model, x, epsilon, all_splits, loss_fn, size=10):
    # updates the trajectory of each split

    all_gap_losses = np.zeros((size,))
    for i in range(0,all_splits.shape[0]):
        splits = all_splits[i]
        fast_weights = OrderedDict(model.named_parameters())  
        for train_steps_inner in range(0,7):
            logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1],fast_weights)
            y = create_nshot_task_label(1,len(x[0][splits==1])).to(device)
            loss = loss_fn(logits,y)
            gradients = torch.autograd.grad(loss,fast_weights.values(),create_graph=True, retain_graph=True)
            fast_weights = OrderedDict((name, param - 0.001 * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))

        # Calculate gap loss to update graph
        y_q = create_nshot_task_label(1,len(x[0][splits==0])).to(device)
        q_logits = model.functional_forward(x[0][splits==0]+epsilon[splits==0], fast_weights)
        s_logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1], fast_weights)
        gap_loss = loss_fn(q_logits, y_q)- loss_fn(s_logits, y)
        all_gap_losses[i] = gap_loss
        
        
    return all_gap_losses

def create_adversary(x,y,model):
    loss_fn = nn.CrossEntropyLoss().to(device)
    x = torch.clamp(x,0.,1.)

    epsilon = torch.zeros_like(x[0], requires_grad=True)

    # create bernoulli draw for data splits
    shape = torch.full((10,),0.5)
    splits = torch.bernoulli(shape)
    eps_optim = torch.optim.Adam([epsilon], lr=0.1)
    fast_weights = OrderedDict(model.named_parameters())
    counter = 0
    min_loss = 0.5
    min_splits = np.array([0,0,0,0,0,0,0,0,0,0])
    all_gap_losses = np.zeros((35,))
    min_index = 0

    # Get all splits and then choose which ones we want based on how many data points we know
    # Right now this is just manually inputted, will switch to being an argument to the call

    all_splits = get_splits()

    # 0 is query set and 1 is support set. This just gets all splits with the known point in the specified spot
    
    part1 = all_splits[np.logical_and((all_splits[:,1]==0),(all_splits[:,5]==1))]
    #part1 = all_splits[all_splits[:,2]==0]
    #all_splits = part1[np.logical_and((part1[:,2]==0),(part1[:,8]==1))]
    all_splits = part1[part1[:,7]==1]
    #all_splits = part1
    print(all_splits.shape)
    for i in range(0,all_splits.shape[0]):
        splits = all_splits[i]
        fast_weights = OrderedDict(model.named_parameters())  
        for train_steps_inner in range(0,7):
            logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1],fast_weights)
            y = create_nshot_task_label(1,len(x[0][splits==1])).to(device)
            loss = loss_fn(logits,y)
            gradients = torch.autograd.grad(loss,fast_weights.values(),create_graph=True, retain_graph=True)
            fast_weights = OrderedDict((name, param - 0.001 * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))

        # Find minimum gap loss as first split to minimize over
        y_q = create_nshot_task_label(1,len(x[0][splits==0])).to(device)
        q_logits = model.functional_forward(x[0][splits==0]+epsilon[splits==0], fast_weights)
        s_logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1], fast_weights)
#        print("Gap Loss:")
        gap_loss = loss_fn(q_logits, y_q)- loss_fn(s_logits, y)
        all_gap_losses[i] = gap_loss
        if gap_loss < min_loss:
            min_loss = gap_loss
            min_splits = splits
            min_index = i

    
    all_trajs = all_gap_losses
    # this for loop is when we want to optimize over sets of k worst splits
    # in the semi-agnostic case, there is no benefit to doing multiple sets
    for k in range(0,1):
        #parts = np.argpartition(all_gap_losses,11)
        trajs = all_gap_losses
        #total_splits = all_splits
        #all_splits = all_splits[parts[0:10]]

        # number of splits to minimize over
        # this is manually changed right now, will make it input later
        # change all instances of 126
        for j in range(0, 10):    

            # get minimum split and do the epsilon optimization
            splits = min_splits

            epsilon.data = max_update(model, x, epsilon, eps_optim, splits, loss_fn)

            # Updates min 10 and then all the splits
            #gap_losses = update_graph(model, x, epsilon, all_splits, loss_fn, 10)
            # Updates the trajectories of all splits
            updated_all_losses = update_graph(model, x, epsilon, all_splits, loss_fn, 35)

            #all_gap_losses = np.vstack((all_gap_losses,traj))
            print("All loss gap updates: ")

            # get the new worst split
            #min_index = np.argmin(gap_losses)
            min_index = np.argmin(updated_all_losses)

            # update the trajectories
            #trajs = np.vstack((trajs,gap_losses))
            all_trajs = np.vstack((all_trajs,updated_all_losses))
            print(all_trajs.shape)
            print(min_index)
            
            min_splits = all_splits[min_index]
        all_gap_losses = updated_all_losses
        #all_splits = total_splits
        #splits = total_splits[min_index]



    # save the trajectories of all the splits
    np.save('./two_and_one_0.05_10',all_trajs)
    #np.save('./example_pics', (x+epsilon).detach().cpu().numpy())
    #np.save('./datapoints',x.cpu().numpy())
    #np.save('./trajs',trajs)
    #np.save('./all_splits',total_splits)
    acc1 = 0
    acc2 = 0
    s_acc = []
    q_acc = []
    acc_diff = []
    print("Start of tests")
    
    # test all the possible splits with the known data point pattern
    for l in range(0,all_splits.shape[0]):
        test_splits = all_splits[l]
        test_weights = OrderedDict(model.named_parameters())
        y_q2 = create_nshot_task_label(1,len(x[0][test_splits==0])).to(device)
        y2 = create_nshot_task_label(1,len(x[0][test_splits==1])).to(device)
        for train_steps in range(0,7):
            logits = model.functional_forward(x[0][test_splits==1]+epsilon[test_splits==1],test_weights)
            y2 = create_nshot_task_label(1,len(x[0][test_splits==1])).to(device)
            loss = loss_fn(logits,y2)
            gradients = torch.autograd.grad(loss,test_weights.values(),create_graph=False)
            test_weights = OrderedDict((name, param - 0.001 * grad) for ((name, param), grad) in zip(test_weights.items(), gradients))
       
        q_logits = model.functional_forward(x[0][test_splits==0]+epsilon[test_splits==0], test_weights)
        logits = model.functional_forward(x[0][test_splits==1]+epsilon[test_splits==1],test_weights)
        
        print("Query Loss: ", loss_fn(q_logits,y_q2))
        print("Support Loss: ",loss_fn(logits,y2))

        acc1 += sum(logits.argmax(dim=1)==y2).item()/len(y2)
        acc2 += sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2)
        s_acc.append(sum(logits.argmax(dim=1)==y2).item()/len(y2))
        q_acc.append(sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2))
        acc_diff.append(sum(logits.argmax(dim=1)==y2).item()/len(y2)-sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2))

    print("DONE")
    #print("Support accuracy:", sum(logits.argmax(dim=1)==y2).item()/len(y2))
    #print("Query accuracy:", sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2))
    
    #return sum(logits.argmax(dim=1)==y2).item()/len(y2), sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2)
    return acc1/35, acc2/35, s_acc, q_acc, acc_diff

def split_knowing_attack(x,y,model):
    loss_fn = nn.CrossEntropyLoss().to(device)
    x = torch.clamp(x,0.,1.)

    epsilon = torch.rand(x[0].shape,requires_grad=True,device='cuda',dtype=torch.double)
    
    # create bernoulli draw for data splits
    shape = torch.full((10,),0.5)
    splits = torch.bernoulli(shape)
    fast_weights = OrderedDict(model.named_parameters())
    counter = 0
    all_gap_losses = np.zeros((252,))

    acc1 = 0
    acc2 = 0
    s_acc = []
    q_acc = []
    acc_diff = []

    # Get all splits and then iterate through all of them 
    all_splits = get_splits()
    for i in range(0,all_splits.shape[0]):
        epsilon = torch.zeros_like(x[0], requires_grad=True)
        eps_optim = torch.optim.Adam([epsilon], lr=0.1)
        splits = all_splits[i]
        for j in range(0,1):
            fast_weights = OrderedDict(model.named_parameters())  
            # support training
            for train_steps_inner in range(0,7):
                logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1],fast_weights)
                y = create_nshot_task_label(1,len(x[0][splits==1])).to(device)
                loss = loss_fn(logits,y)
                gradients = torch.autograd.grad(loss,fast_weights.values(),create_graph=True, retain_graph=True)
                fast_weights = OrderedDict((name, param - 0.001 * grad) for ((name, param), grad) in zip(fast_weights.items(), gradients))

            # update epsilon to maximize gap loss
            y_q = create_nshot_task_label(1,len(x[0][splits==0])).to(device)
            q_logits = model.functional_forward(x[0][splits==0]+epsilon[splits==0], fast_weights)
            s_logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1], fast_weights) 

            print("Training Support Accuracy: ", sum(s_logits.argmax(dim=1)==y).item()/len(y))   
            print("Training Query Accuracy: ", sum(q_logits.argmax(dim=1)==y_q).item()/len(y_q)) 

            loss = -1 * loss_fn(q_logits, y_q) - loss_fn(s_logits, y)
            print("Loss: ", loss)
            loss.backward()
            eps_optim.step()
            eps_optim.zero_grad()

            # bound epsilons
            epsilon.data = torch.clamp(epsilon,-0.1,0.1)  
            epsilon.data = (torch.clamp(x[0]+ epsilon,0.,1.) - x[0])  


        print("Testing attack")
        test_weights = OrderedDict(model.named_parameters())
        y_q2 = create_nshot_task_label(1,len(x[0][splits==0])).to(device)
        y2 = create_nshot_task_label(1,len(x[0][splits==1])).to(device)
        for train_steps in range(0,7):
            logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1],test_weights)
            y2 = create_nshot_task_label(1,len(x[0][splits==1])).to(device)
            loss = loss_fn(logits,y2)
            gradients = torch.autograd.grad(loss,test_weights.values(),create_graph=False)
            test_weights = OrderedDict((name, param - 0.001 * grad) for ((name, param), grad) in zip(test_weights.items(), gradients))

        q_logits = model.functional_forward(x[0][splits==0]+epsilon[splits==0], test_weights)
        logits = model.functional_forward(x[0][splits==1]+epsilon[splits==1],test_weights)
        
        #print("Query Loss: ", loss_fn(q_logits,y_q2))
        #print("Support Loss: ",loss_fn(logits,y2))
        print("Testing Support Accuracy: ", sum(logits.argmax(dim=1)==y2).item()/len(y2))
        print("Testing Query Accuracy: ", sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2))

        acc1 += sum(logits.argmax(dim=1)==y2).item()/len(y2)
        acc2 += sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2)
        s_acc.append(sum(logits.argmax(dim=1)==y2).item()/len(y2))
        q_acc.append(sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2))
        acc_diff.append(sum(logits.argmax(dim=1)==y2).item()/len(y2)-sum(q_logits.argmax(dim=1)==y_q2).item()/len(y_q2))

    return acc1/252, acc2/252, s_acc, q_acc, acc_diff

support_acc = 0
query_acc = 0
prepare_batch = prepare_meta_batch(5,1,5,1)
#accs = []
for batch_index, batch in enumerate(evaluation_taskloader):
    x, y = prepare_batch(batch)

    # this is for the (full and partial) split-agnostic attack
    acc1, acc2, std_s, std_q, acc_diff = create_adversary(x,y,model) 

    # this is for split-knowing attack
    #acc1, acc2, std_s, std_q, acc_diff = split_knowing_attack(x,y,model)

 #   accs.append(acc_diff)
    support_acc += acc1
    query_acc += acc2
    print("Support and Query acc: ", acc1, acc2)
    print(std_s, std_q)
    print(acc_diff)
    #print(x.shape)
print("Total Support Accuracy:",support_acc/10)
print("Total Query Accuracy:",query_acc/10)
print("Linf bound:", 0.1)




    
