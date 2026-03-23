import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch
import torch.nn as nn
from ts_benchmark.baselines.pre_train.submodules.TinyTimeMixer.modeling_tinytimemixer import TinyTimeMixerLayer,TinyTimeMixerConfig,TinyTimeMixerChannelFeatureMixerBlock,FeatureMixerBlock,PatchMixerBlock

   
class PredictionHead(nn.Module):
    def __init__(self, individual=False, n_vars=7, hidden = 256, forecast_len=96, head_dropout=0.2, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = hidden

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            # self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)
            # self.linear2 = nn.Linear(forecast_len, forecast_len)
            self.act = nn.GELU()

    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        b,n,d,p = x.shape
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:   
            x = self.dropout(x)
            x = self.linear1(x)      # x: [bs x nvars x forecast_len]

        # top_index
        return x.transpose(2,1)     # [bs x forecast_len x nvars]
    


class ProjectBlock(nn.Module):
    def __init__(self, d_model, n_vars, num_patch, ffn=1, dropout=0.2):
        super().__init__()

        self.norm = self.norm = nn.LayerNorm(d_model)

        num_hidden = d_model * ffn
        self.fc1 = nn.Linear(d_model, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self.weight_layer = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden: torch.Tensor):
        residual = hidden
        hidden = self.norm(hidden)

        hidden = self.dropout1(nn.functional.gelu(self.fc1(hidden)))
        hidden = self.fc2(hidden)
        avg = hidden.mean(-2)
        hidden = self.dropout2(hidden)
        
        weight = self.softmax(self.weight_layer(avg).squeeze())
        hidden = hidden * weight.unsqueeze(-1).unsqueeze(-1)

        out = hidden + residual
        return out
    
class Plugin(nn.Module):
    def __init__(self, backbone=None, pth='', configs=None):
        super().__init__()

        self.d_model = configs.plugin.plugin_dim
        self.n_vars = configs.enc_in
        self.forcast_len = configs.horizon
        dropout = configs.plugin.dropout
        head_dropout = configs.plugin.head_dropout
        
        
        configs.plugin.M = configs.enc_in//10+1

        self.init_fm(backbone, pth)
        model_dim, self.patch_size, self.stride, self.patch_num =  self.fm.get_settings()
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.patch_num * model_dim, configs.horizon)
        
        self.adapter =  nn.ModuleDict({
            'pos': nn.Sequential(nn.Linear(model_dim,model_dim),nn.Dropout(dropout), nn.GELU(), nn.Linear(model_dim,self.d_model)),
            'neg': nn.Sequential(nn.Linear(model_dim,model_dim),nn.Dropout(dropout), nn.GELU(), nn.Linear(model_dim,self.d_model))
        })
        

        n=configs.plugin.num_before
        self.projecions_before = nn.ModuleDict({
            'pos': nn.ModuleList([ProjectBlock(self.d_model, self.n_vars, self.patch_num, dropout=dropout) for _ in range(n)]),
            'neg': nn.ModuleList([ProjectBlock(self.d_model, self.n_vars, self.patch_num, dropout=dropout) for _ in range(n)])
        })

        n=configs.plugin.num_after
        self.projecions_after = nn.ModuleDict({
            'pos': nn.ModuleList([ProjectBlock(self.d_model, self.n_vars, self.patch_num, dropout=dropout) for _ in range(n)]),
            'neg': nn.ModuleList([ProjectBlock(self.d_model, self.n_vars, self.patch_num, dropout=dropout) for _ in range(n)])
        })

        self.contrastive = Channel_contrastive(self.n_vars, model_dim=model_dim, M=configs.plugin.M,
                         K=configs.plugin.K, de=configs.plugin.de, thresold=configs.plugin.thresold)
        
        self.prediction_length = configs.horizon

        self.head = nn.ModuleDict(
            {'pos':PredictionHead(False,self.n_vars,self.d_model,96,head_dropout=head_dropout),
             'neg':PredictionHead(False,self.n_vars,self.d_model,96,head_dropout=head_dropout)})
        self.norm = nn.LayerNorm(self.d_model)
        self.beta = nn.Parameter(torch.tensor([configs.plugin.beta]*self.n_vars))
        self.gama = configs.plugin.gama
        self.label_len = configs.seq_len - 96
        self.output_patch_len = 96
        self.config = configs
    def freeze_backbone(self):

        for param in self.fm.parameters():
            param.requires_grad = False
        
    def unfreeze_backbone(self):
        for param in self.fm.parameters():
            param.requires_grad = True

    def init_fm(self,fm,pth):

        fm.model.use_plugin = True
        self.fm = fm
        # print(f'=================loading {pth}========================')
        # self.fm.load_state_dict(torch.load(pth))
        # self.freeze_backbone()
        
    def forward(self, inputs, dec_inp, x_mark_enc, x_mark_dec,  device=None, num_samples=None):
        B, _, K = inputs.shape

        inputs = rearrange(inputs, 'b l k -> (b k) l 1')
        x_mark_enc = repeat(x_mark_enc, 'b l f -> (b k) l f', k=K)
        x_mark_dec = repeat(x_mark_dec, 'b l f -> (b k) l f', k=K)

        dec_inp = torch.zeros_like(inputs[:, -self.prediction_length:, :]).float()
        dec_inp = torch.cat((inputs[:, -self.label_len:, ...], dec_inp), dim=1).float()

        if self.config.is_train == 0:
            inference_steps = self.prediction_length // self.output_patch_len
            dis = self.prediction_length - inference_steps * self.output_patch_len
            if dis != 0:
                inference_steps += 1

            pred_y = []

            for j in range(inference_steps):
                if len(pred_y) != 0:
                    inputs = torch.cat([inputs[:, self.output_patch_len:, :], pred_y[-1]], dim=1)
                    tmp = x_mark_dec[:, j - 1:j, :]
                    x_mark_enc = torch.cat([x_mark_enc[:, 1:, :], tmp], dim=1)

                outputs = self.forcast(inputs, x_mark_enc, dec_inp, x_mark_dec)
                pred_y.append(outputs[:, -self.output_patch_len:, :])

            pred_y = torch.cat(pred_y, dim=1)
            if dis != 0:
                pred_y = pred_y[:, :-dis, :]
            # pred_y = rearrange(pred_y, '(b k) l 1 -> b l k', b=B, k=K)
            pred_y = pred_y[:, :self.prediction_length, :]
            return pred_y
        else:
            return self.forcast(inputs, x_mark_enc, dec_inp, x_mark_dec)


    # input,embedding,output
    def forcast(self, input, x_mark_enc, dec_inp, x_mark_dec,  device=None, num_samples=None):
        '''
        ts: [batch_size x seqence x n_vars]
        z: [batch_size x n_vars x num_patch x model_dim]
        out: [batch_size x forcast_len n_vars x n_vars]
        '''
        b,t,k = input.shape
        output, embedding = self.fm.forcast_for_plugin_decoder(input, x_mark_enc, dec_inp, x_mark_dec)

        '''
        output: [batch_size x seqence x n_vars]
        embedding: [batch_size x n_vars x num_patch x model_dim]
        '''
        embedding = rearrange(embedding,'(b k) 1 p d -> b k p d', k=self.n_vars)

        embedding = self.dropout(embedding)
        input = rearrange(input,'(b c) (n p) 1-> (b n) c p', p = self.patch_size, c=self.n_vars)

        self.contrastive.cal_corr(input, embedding)
        
        loss={}
        enchance={}
        for polarity in ['pos','neg']:   
            x = self.adapter[polarity](embedding)
            x_mixer = x
            for mixer in self.projecions_before[polarity]:
                x_mixer = mixer(x_mixer)

            cc_loss, A = self.contrastive(rearrange(x_mixer,'b c p d -> (b p) c d'), polarity) 
            loss[polarity] = cc_loss
            for mixer in self.projecions_after[polarity]:
                x_mixer = mixer(x_mixer)
            
            x_mixer = self.norm(x_mixer + x)
            enchance[polarity] = x_mixer

        enchance = self.head[polarity](enchance['neg'] + enchance['pos'])
        enchance = rearrange(enchance, 'b p c t -> b (p t) c')
        output_plugin = (self.fm.denorm_for_plugin(enchance))
        
        loss = loss['neg'] +loss['pos'] 

        output = rearrange(output, '(b c) t 1 -> b t c' , c = self.n_vars)
        output = (output_plugin*self.beta + output * (1-self.beta))

        if self.config.is_train == 0:
            return output
        else:
            if self.training:
                return output,  self.gama * loss 
            else:
                return output
    
class Channel_contrastive(nn.Module):
    def __init__(self, n_vars, model_dim, M=3, K=3, de=3, thresold=0.3):
        super().__init__()
        self.N = n_vars
        self.M = M
        self.K = K
        self.thresold = thresold
        self.q = nn.Parameter(torch.randn(n_vars, M)) 
        self.V1 = nn.Parameter(torch.randn(M, de))
        self.V2 = nn.Parameter(torch.randn(M, de))
        self.f = nn.Linear(model_dim, self.K)
    
    def polynomial(self, embedding):
        '''
            embedding: [batch_size x n_vars x num_patch x model_dim]
        '''
        embedding = rearrange(embedding,'b n p d -> (b p) n d')
        q = self.q
        C = self.f(embedding).unsqueeze(-2).expand(-1,-1, self.M, -1)
        Q = C[...,0] * q
        for i in range(1, self.K):
            q = q * self.q
            Q = C[...,i] * q + Q
        return Q

    def composition(self, ts, Q):
        V = torch.mm(self.V1, self.V2.transpose(0,1)).unsqueeze(0).expand(Q.size(0),-1,-1)
        Corr = torch.sigmoid( torch.bmm (torch.bmm(Q , V) , Q.permute(0, 2, 1)))

        return (self.cal_pearson_corr(ts) + Corr)/2

    def cal_corr(self,ts,embedding):
        Q = self.polynomial(embedding)
        self.A = self.composition(ts, Q)

    def forward(self,features, polarity):
        A = self.A
        if polarity == 'neg':
            A = A*((A<-1*self.thresold ).float()*-1) + A*(A==1).float()
        else:
            A = A*((A>self.thresold ).float())

        A_pos = A 
        dist_pos = self.get_feature_dis(features)
        loss_pos = self.cal_loss(dist_pos,A_pos)
        
        return loss_pos,A
    
    def cal_loss(self,x_dis,A):
        tau=1
        x_dis = torch.exp(tau * x_dis)
        x_dis_sum = torch.sum(x_dis, -1)
        x_dis_sum_pos = torch.sum(x_dis * A, -1)
        return  -torch.log(x_dis_sum_pos * (x_dis_sum**(-1)) + 1e-8).mean()
    
    def cal_pearson_corr(self,x):
        # 获取 x 的形状信息
        _, _, series_length = x.shape

        # 计算均值和标准差
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, unbiased=False, keepdim=True)

        # 标准化 x
        centered_x = x - mean

        # 计算协方差矩阵
        cov_matrix = torch.matmul(centered_x, centered_x.transpose(1, 2)) / series_length

        # 计算标准差的外积
        std_outer = torch.matmul(std, std.transpose(1, 2))

        # 使用 torch.where 避免梯度切断
        # 防止除以零，将 std_outer 为 0 的地方置为一个小的常数
        std_outer_safe = torch.where(std_outer == 0, torch.tensor(1e-8, device=x.device), std_outer)
        
        
        # 计算皮尔逊相关系数
        pearson_corr = cov_matrix / std_outer_safe
        e = torch.eye(pearson_corr.size(1)).unsqueeze(0).expand(pearson_corr.size(0),-1,-1).to(pearson_corr.device)
        pearson_corr = pearson_corr*(1-e) + e
        return pearson_corr
    
    def get_feature_dis(self, x):

        x_dis = torch.matmul(x, x.transpose(-2, -1))  # Equivalent to x @ x.transpose(-2, -1)

        mask = torch.eye(x_dis.shape[-1], device=x.device).unsqueeze(0)  # Shape: [1, n_vars, n_vars]
        

        x_sum = torch.sum(x ** 2, dim=2, keepdim=True)  # Sum of squares
        x_sum = torch.sqrt(x_sum)  # L2 norm
        
        x_sum = torch.matmul(x_sum, x_sum.transpose(-2, -1))  # ||x_i|| * ||x_j||
        
        # 防止除以零（可选）
        epsilon = 1e-8
        x_sum = x_sum + epsilon
        
        x_dis = x_dis / x_sum  # Cosine similarity
        
        x_dis = (1 - mask) * x_dis
        
        return x_dis
    