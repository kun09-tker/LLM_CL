import time
import random
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
import torch
import Tools.utils as utils
import torch.nn as nn
# from apex import amp

import torch.nn.functional as F
from Optimization.Bert import BertAdam
from Loss.contrastive_loss import SupConLoss, DistillKL

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target):
        probs = F.softmax(logits, 1)
        loss = (- target * torch.log(probs)).sum(1).mean()
        return loss

class ApprBase(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model,aux_model=None,logger=None,taskcla=None, args=None):
        # can deal with aux and unaux
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        logger.info("device: {} n_gpu: {}".format(
            self.device, self.n_gpu))

        self.clipgrad=10000

        self.aux_model=aux_model
        self.model=model
        self.logger = logger

        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        self.ce=torch.nn.CrossEntropyLoss()
        self.soft_ce=SoftCrossEntropy()
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)
        self.kd = DistillKL(4)

        self.smax = 400
        self.thres_cosh=50
        self.thres_emb=6
        self.lamb=0.75

        self.mask_pre=None
        self.mask_back=None
        self.aux_mask_pre=None
        self.aux_mask_back=None


        self.tsv_para = \
            ['adapter_capsule_mask.capsule_net.tsv_capsules.route_weights'] + \
            ['adapter_capsule_mask.route_weights'] + \
            ['adapter_capsule_mask.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.weight' for c_t in range(self.model.num_task) ] + \
            ['adapter_capsule_mask.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.bias' for c_t in range(self.model.num_task)] + \
            ['adapter_capsule_mask.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.weight' for c_t in range(self.model.num_task) ] + \
            ['adapter_capsule_mask.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.bias' for c_t in range(self.model.num_task) ] + \
            ['adapter_capsule_mask.fc1.' + str(c_t) + '.weight' for c_t in range(self.model.num_task)] + \
            ['adapter_capsule_mask.fc1.' + str(c_t) + '.bias' for c_t in range(self.model.num_task) ] + \
            ['adapter_capsule_mask.fc2.' + str(c_t) + '.weight' for c_t in range(self.model.num_task)] + \
            ['adapter_capsule_mask.fc2.' + str(c_t) + '.bias' for c_t in range(self.model.num_task) ] + \
            ['adapter_capsule_mask.capsule_net.tsv_capsules.route_weights'] + \
            ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.tsv_capsules.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.route_weights'
            for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc1.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc1.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc2.'+str(c_t)+'.weight'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)] + \
           ['bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule.fc2.'+str(c_t)+'.bias'
            for c_t in range(self.model.num_task) for layer_id in range(self.model.config.num_hidden_layers)]




        print('DIL BERT ADAPTER MASK BASE')

    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        sup_loss = self.sup_con(outputs, targets,args=self.args)
        return sup_loss


    def augment_distill_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        bsz = input_ids.size(0)

        if self.args.distill_head:
            outputs = [output.clone().unsqueeze(1)]
        else:
            outputs = [pooled_rep.clone().unsqueeze(1)]

        with torch.no_grad():
            for pre_t in range(t):
                pre_output_dict = self.model(pre_t,input_ids, segment_ids, input_mask,s=self.smax)
                pre_pooled_rep = pre_output_dict['normalized_pooled_rep']
                pre_output = pre_output_dict['y']
        if self.args.distill_head:
            outputs.append(pre_output.unsqueeze(1).clone())
        else:
            outputs.append(pre_pooled_rep.unsqueeze(1).clone())
        outputs = torch.cat(outputs, dim=1)
        augment_distill_loss= self.sup_con(outputs,args=self.args)

        return augment_distill_loss


    def amix_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s):
        amix_loss = 0
        if self.args.amix_head:
            mix_pooled_reps = [output.clone().unsqueeze(1)]
        else:
            mix_pooled_reps = [pooled_rep.clone().unsqueeze(1)]


        mix_output_dict = self.model(t,input_ids, segment_ids, input_mask,s=s,start_mixup=True)
        mix_output = mix_output_dict['y']
        mix_masks = mix_output_dict['masks']
        mix_pooled_rep = mix_output_dict['normalized_pooled_rep']

        if 'til' in self.args.scenario:
            mix_output = mix_output[t]
        n_loss,_=self.hat_criterion_adapter(mix_output,targets,mix_masks) # it self is also training
        amix_loss+=n_loss # let's first do some pre-training

        if self.args.amix_head:
            mix_pooled_reps.append(mix_output.unsqueeze(1).clone())
        else:
            mix_pooled_reps.append(mix_pooled_rep.unsqueeze(1).clone())


        cur_mix_outputs = torch.cat(mix_pooled_reps, dim=1)

        amix_loss += self.sup_con(cur_mix_outputs, targets,args=self.args) #train attention and contrastive learning at the same time
        return amix_loss


    def hat_criterion_adapter_aux(self,outputs,targets,masks,t=None):
        reg=0
        count=0
        ewc_loss=0

        if self.aux_mask_pre is not None:
            for key in set(masks.keys()) & set(self.aux_mask_pre.keys()):
                m = masks[key]
                mp = self.aux_mask_pre[key]
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m_key,m_value in masks.items():
                reg+=m_value.sum()
                count+=np.prod(m_value.size()).item()
        reg/=count


        return self.ce(outputs,targets)+self.lamb*reg,reg


    def hat_criterion_adapter(self,outputs,targets,masks):
        reg=0
        count=0

        if self.mask_pre is not None:
            # for m,mp in zip(masks,self.mask_pre):
            for key in set(masks.keys()) & set(self.mask_pre.keys()):
                m = masks[key]
                mp = self.mask_pre[key]
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m_key,m_value in masks.items():
                reg+=m_value.sum()
                count+=np.prod(m_value.size()).item()

        reg/=count

        return self.ce(outputs,targets)+self.lamb*reg,reg


    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred,average=average)


    def acc_compute_fn(self,y_true, y_pred):
        try:
            from sklearn.metrics import accuracy_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return accuracy_score(y_true, y_pred)


class Appr(ApprBase):
    def __init__(self,model,logger,taskcla, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)
        print('DIL BERT ADAPTER MASK SUP NCL')

        return

    def train(self,t,train,valid,num_train_steps,train_data,valid_data):

        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)


        best_loss=np.inf
        best_model=utils.get_model(self.model)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step,e)
            clock1=time.time()

            train_loss,train_acc,train_f1_macro=self.eval(t,train,trained_task=t)
            clock2=time.time()
            self.logger.info('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc))

            # print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
            #     1000*self.train_batch_size*(clock1-clock0)/len(train),1000*self.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')

            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid,trained_task=t)
            self.logger.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc))

            # print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        # Activations mask
        # task=torch.autograd.Variable(torch.LongTensor([t]).to(self.device),volatile=False)

        if self.args.multi_gpu and not self.args.distributed:
            mask=self.model.module.mask(t,s=self.smax)
        else:
            mask=self.model.mask(t,s=self.smax)
        for key,value in mask.items():
            mask[key]=torch.autograd.Variable(value.data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for key,value in self.mask_pre.items():
                self.mask_pre[key]=torch.max(self.mask_pre[key],mask[key])

        # Weights mask
        self.mask_back={}
        for n,p in self.model.named_parameters():
            if self.args.multi_gpu and not self.args.distributed:
                vals=self.model.module.get_view_for(n,p,self.mask_pre)
            else:
                vals=self.model.get_view_for(n,p,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals


        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step,e):

        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            # supervised CE loss ===============
            output_dict = self.model(t,input_ids, segment_ids, input_mask,s=s)
            masks = output_dict['masks']
            pooled_rep = output_dict['normalized_pooled_rep']
            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]


            loss,_=self.hat_criterion_adapter(output,targets,masks) #output_ce is the output of head (no softmax)


            # transfer contrastive ===============

            if self.args.amix and t > 0:
                loss += self.amix_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s)

            if self.args.augment_distill and t > 0: #separatlu append
                loss += self.augment_distill_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets, t,s)

            if self.args.sup_loss:
                loss += self.sup_loss(output,pooled_rep,input_ids, segment_ids, input_mask,targets,t,s)


            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

            # break
        return global_step





    def eval(self,t,data,test=None,trained_task=None):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        target_list = []
        pred_list = []

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                if 'dil' in self.args.scenario:

                    if self.args.last_id: # use the last one
                        output_dict = self.model(trained_task,input_ids, segment_ids, input_mask,s=self.smax)
                        output = output_dict['y']
                        masks = output_dict['masks']

                    elif self.args.ent_id: # detect the testing is
                        outputs = []
                        entropies = []

                        if trained_task is None: #training
                            entrop_to_test = range(0, t + 1)
                        else: #testing
                            entrop_to_test = range(0, trained_task + 1)

                        for e in entrop_to_test:
                            output_dict = self.model(e,input_ids, segment_ids, input_mask,s=self.smax)
                            output = output_dict['y']
                            masks = output_dict['masks']
                            outputs.append(output) #shared head

                            Y_hat = F.softmax(output, -1)
                            entropy = -1*torch.sum(Y_hat * torch.log(Y_hat))
                            entropies.append(entropy)
                        inf_task_id = torch.argmin(torch.stack(entropies))
                        # self.logger.info('inf_task_id: '+str(inf_task_id))
                        output=outputs[inf_task_id] #but don't know which one

                elif 'til' in self.args.scenario:
                    task=torch.LongTensor([t]).cuda()
                    output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,s=self.smax)
                    outputs = output_dict['y']
                    masks = output_dict['masks']
                    output = outputs[t]


                loss,_=self.hat_criterion_adapter(output,targets,masks)

                _,pred=output.max(1)
                hits=(pred==targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b

            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')


        return total_loss/total_num,total_acc/total_num,f1




